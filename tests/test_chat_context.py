# -*- coding: utf-8 -*-
import math
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.chat_context import (  # noqa: E402
    SUMMARY_USER_PREFIX,
    VisibleMessage,
    _split_protected_tail,
    build_visible_chat_history,
    estimate_text_tokens,
)
from src.config import Config  # noqa: E402
from src.storage import DatabaseManager  # noqa: E402


def _reset_db() -> DatabaseManager:
    DatabaseManager.reset_instance()
    Config.reset_instance()
    return DatabaseManager(db_url="sqlite:///:memory:")


def _config(
    *,
    enabled: bool = True,
    trigger: int = 12000,
    protected: int = 1,
    profile: str = "balanced",
) -> SimpleNamespace:
    return SimpleNamespace(
        agent_context_compression_enabled=enabled,
        agent_context_compression_profile=profile,
        agent_context_compression_trigger_tokens=trigger,
        agent_context_protected_turns=protected,
        llm_model_list=[],
        agent_litellm_model="openai/test-model",
        litellm_model="openai/test-model",
        litellm_fallback_models=[],
    )


def _add_messages(db: DatabaseManager, session_id: str, messages: list[tuple[str, str]]) -> None:
    for role, content in messages:
        db.save_conversation_message(session_id, role, content)


def teardown_function() -> None:
    DatabaseManager.reset_instance()
    Config.reset_instance()


def test_disabled_compression_returns_recent_20_messages() -> None:
    db = _reset_db()
    session_id = "chat-disabled"
    _add_messages(db, session_id, [("user", f"msg-{idx}") for idx in range(25)])

    history = build_visible_chat_history(session_id, MagicMock(), _config(enabled=False))

    assert len(history) == 20
    assert history[0]["content"] == "msg-5"
    assert history[-1]["content"] == "msg-24"


def test_enabled_under_trigger_without_summary_returns_full_history_over_20() -> None:
    db = _reset_db()
    session_id = "chat-full-raw"
    _add_messages(db, session_id, [("user", f"msg-{idx}") for idx in range(25)])

    history = build_visible_chat_history(session_id, MagicMock(), _config(trigger=999999))

    assert len(history) == 25
    assert history[0]["content"] == "msg-0"
    assert history[-1]["content"] == "msg-24"


def test_existing_summary_under_trigger_returns_summary_and_uncovered_messages() -> None:
    db = _reset_db()
    session_id = "chat-summary-under"
    _add_messages(
        db,
        session_id,
        [
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
            ("assistant", "a2"),
        ],
    )
    db.upsert_conversation_summary(session_id, "old summary", 2, 2, 10)

    history = build_visible_chat_history(session_id, MagicMock(), _config(trigger=999999))

    assert history[0]["role"] == "user"
    assert history[0]["content"].startswith(SUMMARY_USER_PREFIX)
    assert [msg["content"] for msg in history[1:]] == ["u2", "a2"]


def test_over_trigger_generates_summary_and_updates_covered_message_id() -> None:
    db = _reset_db()
    session_id = "chat-summarize"
    _add_messages(
        db,
        session_id,
        [
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
            ("assistant", "a2"),
            ("user", "u3"),
        ],
    )
    adapter = MagicMock()
    adapter.call_text.return_value = SimpleNamespace(
        content="## 会话摘要\n新摘要",
        provider="openai",
        model="openai/test-model",
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )

    with patch("src.agent.chat_context.estimate_messages_tokens", return_value=999999):
        history = build_visible_chat_history(session_id, adapter, _config(trigger=1, protected=1))

    summary = db.get_conversation_summary(session_id)
    assert summary is not None
    assert summary["covered_message_id"] == 4
    assert summary["source_message_count"] == 4
    assert history[0]["content"].startswith(SUMMARY_USER_PREFIX)
    assert [msg["content"] for msg in history[1:]] == ["u3"]


def test_second_request_only_summarizes_incremental_unprotected_messages() -> None:
    db = _reset_db()
    session_id = "chat-incremental"
    _add_messages(
        db,
        session_id,
        [
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
            ("assistant", "a2"),
            ("user", "u3"),
        ],
    )
    db.upsert_conversation_summary(session_id, "old summary", 2, 2, 10)
    adapter = MagicMock()
    adapter.call_text.return_value = SimpleNamespace(content="new summary", provider="openai", model="m", usage={})

    with patch("src.agent.chat_context.estimate_messages_tokens", return_value=999999):
        build_visible_chat_history(session_id, adapter, _config(trigger=1, protected=1))

    payload = adapter.call_text.call_args.args[0][1]["content"]
    assert "old summary" in payload
    assert "u2" in payload
    assert "a2" in payload
    assert "u1" not in payload
    assert "a1" not in payload
    assert "u3" not in payload
    assert db.get_conversation_summary(session_id)["covered_message_id"] == 4


def test_protected_tail_counts_recent_user_turns_and_keeps_following_messages() -> None:
    messages = [
        VisibleMessage(1, "user", "u1"),
        VisibleMessage(2, "assistant", "a1"),
        VisibleMessage(3, "assistant", "a-orphan"),
        VisibleMessage(4, "user", "u2"),
        VisibleMessage(5, "assistant", "a2"),
        VisibleMessage(6, "user", "u3"),
    ]

    tail = _split_protected_tail(messages, protected_turns=2)

    assert [msg.id for msg in tail] == [4, 5, 6]


def test_empty_to_summarize_warns_and_does_not_call_llm() -> None:
    db = _reset_db()
    session_id = "chat-protected-only"
    _add_messages(db, session_id, [("user", "u1"), ("assistant", "a1")])
    db.upsert_conversation_summary(session_id, "old summary", 2, 2, 10)
    adapter = MagicMock()

    with patch("src.agent.chat_context.estimate_messages_tokens", return_value=999999):
        with patch("src.agent.chat_context.logger.warning") as warning:
            history = build_visible_chat_history(session_id, adapter, _config(trigger=1, protected=1))

    adapter.call_text.assert_not_called()
    assert warning.called
    assert history[0]["content"].startswith(SUMMARY_USER_PREFIX)
    assert [msg["content"] for msg in history[1:]] == ["u1", "a1"]


def test_empty_to_summarize_without_summary_returns_full_history_and_does_not_call_llm() -> None:
    db = _reset_db()
    session_id = "chat-protected-only-no-summary"
    _add_messages(db, session_id, [("user", "u1"), ("assistant", "a1")])
    adapter = MagicMock()

    with patch("src.agent.chat_context.estimate_messages_tokens", return_value=999999):
        with patch("src.agent.chat_context.logger.warning") as warning:
            history = build_visible_chat_history(session_id, adapter, _config(trigger=1, protected=1))

    adapter.call_text.assert_not_called()
    assert warning.called
    assert history == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_summary_failure_falls_back_to_recent_20_without_old_summary() -> None:
    db = _reset_db()
    session_id = "chat-summary-fails"
    _add_messages(db, session_id, [("user", f"msg-{idx}") for idx in range(25)])
    adapter = MagicMock()
    adapter.call_text.return_value = SimpleNamespace(content="", provider="error", model="", usage={})

    with patch("src.agent.chat_context.estimate_messages_tokens", return_value=999999):
        history = build_visible_chat_history(session_id, adapter, _config(trigger=1, protected=1))

    assert len(history) == 20
    assert history[0]["content"] == "msg-5"


def test_summary_failure_with_old_summary_returns_candidate() -> None:
    db = _reset_db()
    session_id = "chat-summary-fails-old"
    _add_messages(db, session_id, [("user", "u1"), ("assistant", "a1"), ("user", "u2")])
    db.upsert_conversation_summary(session_id, "old summary", 2, 2, 10)
    adapter = MagicMock()
    adapter.call_text.return_value = SimpleNamespace(content="", provider="error", model="", usage={})

    with patch("src.agent.chat_context.estimate_messages_tokens", return_value=999999):
        history = build_visible_chat_history(session_id, adapter, _config(trigger=1, protected=1))

    assert history[0]["content"].startswith(SUMMARY_USER_PREFIX)
    assert [msg["content"] for msg in history[1:]] == ["u2"]


def test_token_estimator_falls_back_to_character_heuristic() -> None:
    with patch("src.agent.chat_context.get_effective_agent_primary_model", side_effect=RuntimeError("no model")):
        assert estimate_text_tokens("abcdefg", _config()) == math.ceil(7 / 3)
