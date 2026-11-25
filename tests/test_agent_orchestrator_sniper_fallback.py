# -*- coding: utf-8 -*-
"""Regression tests for AgentOrchestrator sniper point fallbacks."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.agent.orchestrator import AgentOrchestrator
from src.agent.protocols import AgentContext


class TestAgentOrchestratorSniperFallback(unittest.TestCase):
    def test_finalization_preserves_explicit_exit_rules(self):
        orch = AgentOrchestrator(tool_registry=MagicMock(), llm_adapter=MagicMock())
        ctx = AgentContext(query="test", stock_code="301308", stock_name="测试")
        strategy = {
            "sell_condition": "收盘价达到12.50后考虑卖出一部分。",
            "stop_loss_condition": "收盘价低于10.80才考虑退出，盘中跌破不直接卖出。",
            "validity": "仅2026-11-25交易时段有效。",
        }
        payload = {
            "decision_type": "hold", "analysis_summary": "继续观察",
            "dashboard": {"battle_plan": {"position_strategy": strategy}},
        }
        normalized = orch._finalize_dashboard_payload(payload, ctx)
        self.assertEqual(normalized["dashboard"]["battle_plan"]["position_strategy"], strategy)

    def test_secondary_buy_does_not_duplicate_ideal_buy(self):
        orch = AgentOrchestrator(
            tool_registry=MagicMock(),
            llm_adapter=MagicMock(),
        )
        ctx = AgentContext(query="test", stock_code="301308", stock_name="江波龙")

        payload = {
            "decision_type": "buy",
            "analysis_summary": "趋势仍强，等待回踩。",
            "dashboard": {
                "key_levels": {
                    "support": 301.61,
                    "stop_loss": 295.0,
                    "resistance": 340.44,
                }
            },
        }

        normalized = orch._finalize_dashboard_payload(payload, ctx)

        self.assertIsNotNone(normalized)
        sniper = normalized["dashboard"]["battle_plan"]["sniper_points"]
        self.assertEqual(sniper["ideal_buy"], 301.61)
        self.assertEqual(sniper["secondary_buy"], "N/A")

    def test_secondary_buy_numeric_string_does_not_duplicate_ideal_buy(self):
        orch = AgentOrchestrator(
            tool_registry=MagicMock(),
            llm_adapter=MagicMock(),
        )
        ctx = AgentContext(query="test", stock_code="301308", stock_name="江波龙")

        payload = {
            "decision_type": "buy",
            "analysis_summary": "趋势仍强，等待回踩。",
            "dashboard": {
                "battle_plan": {
                    "sniper_points": {
                        "secondary_buy": "301.61",
                    }
                },
                "key_levels": {
                    "support": 301.61,
                    "stop_loss": 295.0,
                    "resistance": 340.44,
                },
            },
        }

        normalized = orch._finalize_dashboard_payload(payload, ctx)

        self.assertIsNotNone(normalized)
        sniper = normalized["dashboard"]["battle_plan"]["sniper_points"]
        self.assertEqual(sniper["ideal_buy"], 301.61)
        self.assertEqual(sniper["secondary_buy"], "N/A")


if __name__ == "__main__":
    unittest.main()
