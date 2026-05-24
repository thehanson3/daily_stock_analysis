# -*- coding: utf-8 -*-
"""Contract checks for the AnalysisContextPack P0 inventory doc."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = PROJECT_ROOT / "docs" / "analysis-context-pack.md"


def _read_doc() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def _section(doc: str, heading: str) -> str:
    marker = f"## {heading}"
    assert marker in doc
    return doc.split(marker, 1)[1].split("\n## ", 1)[0]


def test_analysis_context_pack_doc_has_required_sections() -> None:
    doc = _read_doc()

    for heading in (
        "## 术语与边界",
        "## P0 范围与非目标",
        "## 字段质量状态",
        "## 现有状态映射",
        "## 七路径盘点",
        "## 源码锚点",
        "## 兼容与安全边界",
    ):
        assert heading in doc


def test_analysis_context_pack_doc_disambiguates_context_surfaces() -> None:
    section = _section(_read_doc(), "术语与边界")

    for token in (
        "`storage.get_analysis_context()`",
        "`enhanced_context`",
        "`analysis_history.context_snapshot`",
        "Agent executor message context",
        "Agent orchestrator `AgentContext`",
        "`AGENT_ARCH=single`",
        "`AGENT_ARCH=multi`",
    ):
        assert token in section


def test_analysis_context_pack_doc_defines_p0_quality_states() -> None:
    section = _section(_read_doc(), "字段质量状态")

    for state in (
        "`available`",
        "`missing`",
        "`not_supported`",
        "`fallback`",
        "`stale`",
        "`estimated`",
        "`partial`",
    ):
        assert state in section
    assert "`fetch_failed`" not in section


def test_analysis_context_pack_doc_covers_seven_paths() -> None:
    section = _section(_read_doc(), "七路径盘点")

    for heading in (
        "### 普通分析",
        "### Agent",
        "### 告警",
        "### 持仓",
        "### 回测",
        "### 历史",
        "### 通知",
    ):
        assert heading in section


def test_analysis_context_pack_doc_records_agent_context_visibility() -> None:
    section = _section(_read_doc(), "七路径盘点")

    for token in (
        "`initial_context`",
        "`fundamental_context`",
        "不显式注入 `fundamental_context` 或 `trend_result`",
        "pre-fetched data",
        "不预注入 `fundamental_context`",
    ):
        assert token in section


def test_analysis_context_pack_doc_records_non_goals_and_safety_boundaries() -> None:
    doc = _read_doc()

    for token in (
        "不定义 `AnalysisContextPack` schema",
        "不新增 builder",
        "不接入 runtime",
        "不 pack 化 `market_review`",
        "`market_light`",
        "`fetch_failed` 与 `not_supported` 的细分留到 P5",
        "`analysis_history.context_snapshot.enhanced_context.date`",
        "完整 pack 不默认公开",
        "API key",
        "token",
        "cookie",
        "完整 webhook URL",
        "邮箱密码",
    ):
        assert token in doc


def test_analysis_context_pack_doc_maps_existing_status_terms() -> None:
    section = _section(_read_doc(), "现有状态映射")

    for token in (
        "`degraded`",
        "`insufficient_data`",
        "`partial_failed`",
        "`data_missing`",
        "`price_stale`",
        "`data_quality=ok/partial/unavailable`",
        "不映射",
    ):
        assert token in section


def test_analysis_context_pack_doc_lists_source_anchors() -> None:
    section = _section(_read_doc(), "源码锚点")

    for path in (
        "src/core/pipeline.py",
        "src/storage.py",
        "src/analyzer.py",
        "src/agent/orchestrator.py",
        "src/agent/executor.py",
        "src/agent/tools/data_tools.py",
        "src/services/alert_worker.py",
        "src/services/portfolio_service.py",
        "src/services/backtest_service.py",
        "src/repositories/backtest_repo.py",
        "src/services/history_service.py",
        "api/v1/endpoints/history.py",
        "api/v1/endpoints/analysis.py",
        "api/v1/schemas/history.py",
        "api/v1/schemas/portfolio.py",
        "src/notification.py",
        "docs/alerts.md",
        "docs/notifications.md",
    ):
        assert path in section


def test_analysis_context_pack_doc_updates_indexes_and_changelog() -> None:
    index = (PROJECT_ROOT / "docs" / "INDEX.md").read_text(encoding="utf-8")
    index_en = (PROJECT_ROOT / "docs" / "INDEX_EN.md").read_text(encoding="utf-8")
    changelog = (PROJECT_ROOT / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "[分析上下文包 P0 盘点](analysis-context-pack.md)" in index
    assert (
        "[Analysis Context Pack P0 Inventory](analysis-context-pack.md) "
        "<sub><sub>![P0 Badge](https://img.shields.io/badge/P0-red?style=flat)</sub></sub> "
        "(Chinese-only)"
    ) in index_en
    assert "[文档] 新增 AnalysisContextPack P0 上下文盘点" in changelog
