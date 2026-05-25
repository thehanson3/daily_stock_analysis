# -*- coding: utf-8 -*-
"""Visible conversation history builder for Agent chat requests."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.config import (
    get_agent_context_compression_preset,
    get_effective_agent_primary_model,
)
from src.storage import get_db, persist_llm_usage

logger = logging.getLogger(__name__)

VISIBLE_ROLES = {"user", "assistant"}
SUMMARY_USER_PREFIX = "[系统生成的历史对话摘要，仅供延续本会话]"
SUMMARY_LLM_TIMEOUT_SECONDS = 20

SUMMARY_SYSTEM_PROMPT = """你是股票问答系统的会话压缩器，只能总结已经出现过的用户可见对话内容。

硬性规则：
- 只总结已有对话，不新增行情、新闻、财务数据或投资建议。
- 不推断未出现的事实，不补充新的买卖建议。
- 必须保留标的、持仓成本、周期、风险偏好、策略视角、关键判断、操作条件、止损止盈、数据时效、工具失败和未决问题。
- 输出必须使用 Markdown，并严格包含以下 5 个二级标题：
  ## 会话摘要
  ## 当前关注标的
  ## 用户偏好与约束
  ## 已有判断与操作条件
  ## 风险、数据时效与未决问题
"""


@dataclass(frozen=True)
class VisibleMessage:
    """A persisted user-visible chat message."""

    id: int
    role: str
    content: str
    created_at: Any = None


def build_summary_message(summary_text: str) -> Dict[str, str]:
    """Build the synthetic summary message injected into chat history."""
    return {
        "role": "user",
        "content": f"{SUMMARY_USER_PREFIX}\n{summary_text.strip()}",
    }


def estimate_text_tokens(text: str, config: Any) -> int:
    """Estimate tokens deterministically enough for compression decisions."""
    normalized_text = text or ""
    try:
        import litellm  # type: ignore

        model = get_effective_agent_primary_model(config)
        count = litellm.token_counter(model=model, text=normalized_text)
        return max(0, int(count or 0))
    except Exception as exc:
        logger.debug("Token counter failed; using character heuristic: %s", exc)
        return int(math.ceil(len(normalized_text) / 3))


def estimate_messages_tokens(messages: Sequence[Dict[str, Any]], config: Any) -> int:
    """Estimate tokens for a list of role/content messages."""
    return estimate_text_tokens(_render_messages(messages), config)


def build_summary_messages(
    previous_summary: str,
    messages: Sequence[VisibleMessage],
) -> List[Dict[str, str]]:
    """Build the text-only summary request messages."""
    sections: List[str] = []
    if previous_summary.strip():
        sections.append("已有滚动摘要：\n" + previous_summary.strip())
    sections.append("本次需要纳入摘要的新增对话：")
    sections.append(_render_visible_messages(messages))
    user_payload = "\n\n".join(sections).strip()
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]


def build_visible_chat_history(
    session_id: str,
    llm_adapter: Any,
    config: Any,
) -> List[Dict[str, str]]:
    """Return visible chat history according to the compression state table."""
    db = get_db()
    if not getattr(config, "agent_context_compression_enabled", False):
        return db.get_conversation_history(session_id, limit=20)

    visible_messages = _load_visible_messages(session_id)
    if not visible_messages:
        return []

    summary_record = db.get_conversation_summary(session_id)
    previous_summary = (summary_record or {}).get("summary") or ""
    covered_message_id = _coerce_int((summary_record or {}).get("covered_message_id"), default=0)
    preset = get_agent_context_compression_preset(
        getattr(config, "agent_context_compression_profile", None)
    )
    trigger_tokens = _coerce_int(
        getattr(config, "agent_context_compression_trigger_tokens", preset.trigger_tokens),
        default=preset.trigger_tokens,
    )
    protected_turns = _coerce_int(
        getattr(config, "agent_context_protected_turns", preset.protected_turns),
        default=preset.protected_turns,
    )

    protected_tail = _split_protected_tail(visible_messages, protected_turns)
    protected_ids = {msg.id for msg in protected_tail}
    uncovered_messages = [msg for msg in visible_messages if msg.id > covered_message_id]
    candidate = (
        [build_summary_message(previous_summary)] + _to_chat_messages(uncovered_messages)
        if previous_summary
        else _to_chat_messages(visible_messages)
    )
    candidate_tokens = estimate_messages_tokens(candidate, config)

    if candidate_tokens <= trigger_tokens:
        return candidate

    to_summarize = [
        msg
        for msg in visible_messages
        if msg.id > covered_message_id and msg.id not in protected_ids
    ]
    if not to_summarize:
        if previous_summary:
            logger.warning(
                "Conversation context compression skipped for session %s: protected tail exceeds trigger",
                session_id,
            )
            return [build_summary_message(previous_summary)] + _to_chat_messages(protected_tail)
        logger.warning(
            "Conversation context compression skipped for session %s: all visible history is protected",
            session_id,
        )
        return _to_chat_messages(visible_messages)

    logger.info(
        "Conversation context compression summarizing session %s: %d messages, candidate_tokens=%d, trigger=%d",
        session_id,
        len(to_summarize),
        candidate_tokens,
        trigger_tokens,
    )
    summary_text, response = _generate_summary(
        llm_adapter=llm_adapter,
        config=config,
        previous_summary=previous_summary,
        to_summarize=to_summarize,
        max_tokens=preset.summary_target_tokens,
    )
    if summary_text:
        new_covered_message_id = max(msg.id for msg in to_summarize)
        estimated_tokens = estimate_text_tokens(summary_text, config)
        db.upsert_conversation_summary(
            session_id=session_id,
            summary=summary_text,
            covered_message_id=new_covered_message_id,
            source_message_count=len(to_summarize),
            estimated_tokens=estimated_tokens,
        )
        persist_llm_usage(
            getattr(response, "usage", {}) or {},
            getattr(response, "model", "") or get_effective_agent_primary_model(config) or "unknown",
            call_type="agent",
        )
        return [build_summary_message(summary_text)] + _to_chat_messages(protected_tail)

    logger.warning(
        "Conversation context compression failed for session %s; using state-table fallback",
        session_id,
    )
    if previous_summary:
        return candidate
    return db.get_conversation_history(session_id, limit=20)


def _load_visible_messages(session_id: str) -> List[VisibleMessage]:
    rows = get_db().get_visible_conversation_messages(session_id)
    messages = []
    for row in rows:
        role = str(row.get("role") or "")
        content = str(row.get("content") or "")
        if role not in VISIBLE_ROLES or not content:
            continue
        messages.append(
            VisibleMessage(
                id=_coerce_int(row.get("id"), default=0),
                role=role,
                content=content,
                created_at=row.get("created_at"),
            )
        )
    return [msg for msg in messages if msg.id > 0]


def _split_protected_tail(messages: Sequence[VisibleMessage], protected_turns: int) -> List[VisibleMessage]:
    if not messages:
        return []
    if protected_turns <= 0:
        return []

    user_count = 0
    # If fewer user turns exist than requested, keep start_index=0 and protect
    # the entire visible history. The caller handles over-trigger protected-only
    # sessions without forcing a magic truncate.
    start_index = 0
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            user_count += 1
            if user_count >= protected_turns:
                start_index = index
                break
    return list(messages[start_index:])


def _generate_summary(
    *,
    llm_adapter: Any,
    config: Any,
    previous_summary: str,
    to_summarize: Sequence[VisibleMessage],
    max_tokens: int,
) -> Tuple[Optional[str], Any]:
    try:
        response = llm_adapter.call_text(
            build_summary_messages(previous_summary, to_summarize),
            temperature=0,
            max_tokens=max_tokens,
            timeout=SUMMARY_LLM_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning("Conversation summary LLM call raised: %s", exc)
        return None, None

    content = (getattr(response, "content", None) or "").strip()
    if getattr(response, "provider", "") == "error" or not content:
        return None, response
    return content, response


def _to_chat_messages(messages: Iterable[VisibleMessage]) -> List[Dict[str, str]]:
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def _render_messages(messages: Sequence[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"{msg.get('role', '')}:\n{msg.get('content', '')}"
        for msg in messages
    )


def _render_visible_messages(messages: Sequence[VisibleMessage]) -> str:
    return "\n\n".join(f"{msg.role}:\n{msg.content}" for msg in messages)


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
