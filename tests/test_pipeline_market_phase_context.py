# -*- coding: utf-8 -*-
"""Regression tests for Issue #1386 P1a market phase context plumbing."""

import os
import sys
import unittest
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.analyzer import AnalysisResult
from src.core.pipeline import StockAnalysisPipeline
from src.enums import ReportType
from src.services.run_diagnostics import (
    activate_run_diagnostic_context,
    current_diagnostic_snapshot,
    reset_run_diagnostic_context,
)


def _analysis_result() -> AnalysisResult:
    return AnalysisResult(
        code="600519",
        name="贵州茅台",
        sentiment_score=62,
        trend_prediction="震荡",
        operation_advice="持有",
        decision_type="hold",
    )


def _phase_payload() -> dict:
    return {
        "market": "cn",
        "phase": "intraday",
        "market_local_time": "2026-03-27T10:00:00+08:00",
        "session_date": "2026-03-27",
        "effective_daily_bar_date": "2026-03-26",
        "is_trading_day": True,
        "is_market_open_now": True,
        "is_partial_bar": True,
        "minutes_to_open": None,
        "minutes_to_close": 300,
        "trigger_source": "system",
        "analysis_intent": "auto",
        "warnings": [],
    }


def _make_pipeline(*, agent_mode: bool = False, save_context_snapshot: bool = True) -> StockAnalysisPipeline:
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.config = SimpleNamespace(
        enable_realtime_quote=False,
        enable_chip_distribution=False,
        realtime_source_priority=[],
        agent_mode=agent_mode,
        agent_skills=[],
        save_context_snapshot=save_context_snapshot,
        report_language="zh",
        report_integrity_enabled=False,
        fundamental_stage_timeout_seconds=1,
    )
    pipeline.source_message = None
    pipeline.query_id = None
    pipeline.query_source = "system"
    pipeline.save_context_snapshot = save_context_snapshot
    pipeline.progress_callback = None
    pipeline.analysis_skills = None
    pipeline.social_sentiment_service = None

    pipeline.fetcher_manager = MagicMock()
    pipeline.fetcher_manager.get_stock_name.return_value = "贵州茅台"
    pipeline.fetcher_manager.get_realtime_quote.return_value = None
    pipeline.fetcher_manager.get_chip_distribution.return_value = None
    pipeline.fetcher_manager.get_fundamental_context.return_value = {
        "market": "cn",
        "coverage": {"boards": "not_supported"},
        "source_chain": [],
    }
    pipeline.fetcher_manager.build_failed_fundamental_context.return_value = {
        "market": "cn",
        "coverage": {"boards": "not_supported"},
        "source_chain": [],
    }

    pipeline.db = MagicMock()
    pipeline.db.get_data_range.return_value = []
    pipeline.db.get_analysis_context.return_value = {
        "code": "600519",
        "stock_name": "贵州茅台",
        "date": "2026-03-26",
        "today": {},
        "yesterday": {},
    }

    pipeline.trend_analyzer = MagicMock()
    pipeline.analyzer = MagicMock()
    pipeline.analyzer.analyze.return_value = _analysis_result()
    pipeline.search_service = MagicMock()
    pipeline.search_service.is_available = False
    pipeline.search_service.news_window_days = 3
    pipeline._emit_progress = MagicMock()
    return pipeline


class PipelineMarketPhaseContextTestCase(unittest.TestCase):
    def test_process_single_stock_propagates_current_time_to_analyze_stock(self):
        pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
        pipeline.query_id = None
        pipeline._emit_progress = MagicMock()
        pipeline._resolve_resume_target_date = MagicMock(return_value=date(2026, 3, 26))
        pipeline.fetch_and_save_stock_data = MagicMock(return_value=(True, None))
        pipeline.analyze_stock = MagicMock(
            return_value=SimpleNamespace(
                success=True,
                operation_advice="持有",
                sentiment_score=60,
            )
        )
        frozen_time = datetime(2026, 3, 27, 10, 0)

        pipeline.process_single_stock(
            "600519",
            report_type=ReportType.SIMPLE,
            analysis_query_id="q-frozen",
            current_time=frozen_time,
        )

        pipeline.analyze_stock.assert_called_once_with(
            "600519",
            ReportType.SIMPLE,
            query_id="q-frozen",
            current_time=frozen_time,
        )

    def test_legacy_pipeline_passes_market_phase_context_to_analyzer_only(self):
        pipeline = _make_pipeline(agent_mode=False, save_context_snapshot=True)
        phase_payload = _phase_payload()
        phase_context = SimpleNamespace(to_dict=MagicMock(return_value=phase_payload))

        with patch("src.core.pipeline.build_market_phase_context", return_value=phase_context) as mock_build:
            result = pipeline.analyze_stock(
                "600519",
                ReportType.SIMPLE,
                "q-runtime",
                current_time=datetime(2026, 3, 27, 10, 0),
            )

        self.assertIsNotNone(result)
        mock_build.assert_called_once()
        enhanced_context = pipeline.analyzer.analyze.call_args.args[0]
        self.assertEqual(enhanced_context["market_phase_context"], phase_payload)

        save_kwargs = pipeline.db.save_analysis_history.call_args.kwargs
        self.assertTrue(save_kwargs["save_snapshot"])
        snapshot = save_kwargs["context_snapshot"]
        self.assertNotIn("market_phase_context", snapshot["enhanced_context"])

    def test_agent_legacy_context_gets_runtime_key_but_history_snapshot_strips_it(self):
        pipeline = _make_pipeline(agent_mode=True, save_context_snapshot=True)
        pipeline._ensure_agent_history = MagicMock()
        phase_payload = _phase_payload()

        from src.agent.executor import AgentResult

        agent_result = AgentResult(
            success=True,
            content="{}",
            dashboard={
                "stock_name": "贵州茅台",
                "sentiment_score": 66,
                "trend_prediction": "震荡",
                "operation_advice": "持有",
                "decision_type": "hold",
            },
            provider="test",
        )
        executor = MagicMock()
        executor.run.return_value = agent_result

        with patch("src.agent.factory.build_agent_executor", return_value=executor):
            result = pipeline._analyze_with_agent(
                code="600519",
                report_type=ReportType.SIMPLE,
                query_id="q-agent",
                stock_name="贵州茅台",
                realtime_quote=None,
                chip_data=None,
                fundamental_context={"market": "cn"},
                trend_result=None,
                market_phase_context=phase_payload,
            )

        self.assertIsNotNone(result)
        run_context = executor.run.call_args.kwargs["context"]
        self.assertEqual(run_context["market_phase_context"], phase_payload)

        save_kwargs = pipeline.db.save_analysis_history.call_args.kwargs
        self.assertTrue(save_kwargs["save_snapshot"])
        self.assertNotIn("market_phase_context", save_kwargs["context_snapshot"])
        enhanced_context = save_kwargs["context_snapshot"]["enhanced_context"]
        self.assertEqual(enhanced_context["stock_name"], "贵州茅台")

    def test_agent_history_snapshot_contains_diagnostics_context_when_active(self):
        pipeline = _make_pipeline(agent_mode=True, save_context_snapshot=True)
        pipeline._ensure_agent_history = MagicMock()
        phase_payload = _phase_payload()
        token = activate_run_diagnostic_context(
            trace_id="trace-agent",
            query_id="q-agent",
            stock_code="600519",
            trigger_source="api",
        )
        try:
            from src.agent.executor import AgentResult

            agent_result = AgentResult(
                success=True,
                content="{}",
                dashboard={
                    "stock_name": "贵州茅台",
                    "sentiment_score": 70,
                    "trend_prediction": "震荡",
                    "operation_advice": "持有",
                    "decision_type": "hold",
                },
                provider="test",
            )
            executor = MagicMock()
            executor.run.return_value = agent_result

            with patch("src.agent.factory.build_agent_executor", return_value=executor):
                result = pipeline._analyze_with_agent(
                    code="600519",
                    report_type=ReportType.SIMPLE,
                    query_id="q-agent",
                    stock_name="贵州茅台",
                    realtime_quote=None,
                    chip_data=None,
                    fundamental_context={"market": "cn"},
                    trend_result=None,
                    market_phase_context=phase_payload,
                )

            self.assertIsNotNone(result)
            save_kwargs = pipeline.db.save_analysis_history.call_args.kwargs
            self.assertTrue(save_kwargs["save_snapshot"])
            snapshot = save_kwargs["context_snapshot"]
            self.assertIn("diagnostics", snapshot)
            diagnostics = snapshot["diagnostics"]
            self.assertIsNotNone(diagnostics)
            self.assertEqual(diagnostics["trace_id"], "trace-agent")
            self.assertEqual(diagnostics["query_id"], "q-agent")
            self.assertEqual(diagnostics["provider_runs"], [])
            self.assertEqual(current_diagnostic_snapshot(), diagnostics)
        finally:
            reset_run_diagnostic_context(token)


if __name__ == "__main__":
    unittest.main()
