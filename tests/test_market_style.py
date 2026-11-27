# -*- coding: utf-8 -*-
"""Regression coverage for dated style signals and the actual notification renderer."""

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.core.market_style import (
    BENCHMARK, SECTOR_POOL, STYLE_INDICES, build_market_brief, classify_style, prepare_daily_bars,
    relative_strength,
)
from src.data.market_fund_catalog import FUND_FAMILIES, FUND_REVIEW_MAX_DAYS, get_fund_reference
from src.core.trading_calendar import get_trading_dates
from src.market_analyzer import MarketOverview
from src.market_analyzer import MarketAnalyzer
from src.notification import NotificationService
from src.services.stock_list_parser import ParseStatus


DAY = date(2026, 3, 6)
NOW = datetime(2026, 3, 6, 15, 25, tzinfo=ZoneInfo("Asia/Shanghai"))


def bars(prices=None, *, end=DAY):
    prices = prices if prices is not None else [100 + index * 0.5 for index in range(80)]
    dates = get_trading_dates("cn", date(end.year - 1, end.month, 1), end)
    assert dates is not None
    frame = pd.DataFrame({
        "date": dates[-len(prices):], "close": prices,
        "high": [price + 0.5 for price in prices], "low": [price - 0.5 for price in prices],
        "volume": [100.0] * len(prices),
    })
    return frame


class HistoryManager:
    def __init__(self, histories=None):
        self.histories = histories or {}
        self.calls = []

    def get_daily_data(self, code, **kwargs):
        self.calls.append((code, kwargs))
        assert kwargs["analysis_target"].asset_type == ParseStatus.INDEX
        return self.histories.get(code, bars()), "fixture"


def test_style_classification_distinguishes_rising_falling_top_and_bottom():
    cases = [
        ([100 + index * 0.5 for index in range(80)], "rising"),
        ([160 - index * 0.5 for index in range(80)], "falling"),
        ([100 + index * 0.5 for index in range(70)] + [140, 138, 136, 132, 128, 125, 122, 120, 118, 116], "top_risk"),
        ([160 - index * 0.5 for index in range(70)] + [120, 116, 112, 110, 108, 109, 110, 111, 112, 114], "bottom_watch"),
    ]
    for prices, expected in cases:
        signal = classify_style(prepare_daily_bars(bars(prices), DAY))
        assert signal["stage"] == expected
        assert signal["support"] == min(prices[-21:-1]) - 0.5
        assert signal["resistance"] == max(prices[-21:-1]) + 0.5


def test_recovery_requires_volume_and_prior_decline():
    frame = bars([160] * 35 + [120] * 44 + [150])
    frame.loc[79, "volume"] = 150
    assert classify_style(prepare_daily_bars(frame, DAY))["stage"] == "recovering"
    frame.loc[79, "volume"] = 0
    assert classify_style(prepare_daily_bars(frame, DAY))["stage"] != "recovering"


@pytest.mark.parametrize("problem", ["stale", "duplicate", "short", "infinite", "negative", "price", "missing", "gap"])
def test_bad_history_never_becomes_a_current_style_signal(problem):
    frame = bars()
    if problem == "stale":
        frame = frame.iloc[:-1]
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.tail(1)])
    elif problem == "short":
        frame = frame.tail(60)
    elif problem == "infinite":
        frame.loc[79, "close"] = float("inf")
    elif problem == "negative":
        frame.loc[79, "volume"] = -1
    elif problem == "price":
        frame.loc[79, "high"] = 1
    elif problem == "gap":
        frame = frame.drop(index=60)
    else:
        frame = frame.drop(columns="high")
    with pytest.raises(ValueError):
        prepare_daily_bars(frame, DAY)


def test_future_or_partial_bars_do_not_enter_the_closed_daily_signal():
    frame = bars(end=date(2026, 3, 9))
    frame.loc[79, "close"] = 100000
    closed = prepare_daily_bars(frame, DAY)
    assert closed["date"].iloc[-1] == DAY
    assert closed["close"].max() < 100000


def test_relative_strength_requires_identical_session_dates():
    left = prepare_daily_bars(bars(), DAY)
    right = prepare_daily_bars(bars(), DAY)
    assert relative_strength(left, right) == 0
    right.loc[74, "date"] = date(2026, 1, 1)
    assert relative_strength(left, right) is None


def test_real_collector_rejects_stale_sources_and_does_not_invent_breadth():
    manager = HistoryManager({STYLE_INDICES["technology"][0]: bars().iloc[:-1]})
    result = build_market_brief(manager, MarketOverview(date=DAY.isoformat()), {}, now=NOW)
    assert not result["styles"]["technology"]["available"]
    assert "尚未更新" in result["markdown"]
    assert "暂无法判断" in result["markdown"]
    assert "未核验组内上涨比例" in result["markdown"]
    assert "上涨股票增多" not in result["markdown"]
    assert "相比上次" not in result["markdown"]
    assert "比上一交易日" in result["markdown"]
    assert result["calendar_verified"]


def test_intraday_collection_excludes_today_and_labels_previous_complete_day():
    now = NOW.replace(hour=14, minute=20)
    previous_day = date(2026, 3, 5)
    manager = HistoryManager({code: bars(end=previous_day) for code in [BENCHMARK, "sh000688", "sh000922"]})
    result = build_market_brief(manager, MarketOverview(date=DAY.isoformat()), {}, now=now)
    assert result["data_date"] == previous_day.isoformat()
    assert result["is_partial_bar"] is True
    assert "盘中观察" in result["markdown"]
    assert all(call[1]["end_date"] == previous_day.isoformat() for call in manager.calls)


def test_sector_recommendation_requires_history_strength_volume_and_registered_identity():
    bank = bars([100 + index for index in range(80)])
    bank.loc[79, "volume"] = 120
    manager = HistoryManager({"sz399986": bank})
    overview = MarketOverview(date=DAY.isoformat(), top_sectors=[
        {"name": "银行", "change_pct": 3}, {"name": "未知题材", "change_pct": 20},
    ])
    result = build_market_brief(manager, overview, {}, now=NOW)
    assert result["sectors"][0]["verified"]
    assert not result["sectors"][1]["verified"]
    assert "还没核实持续性" in result["sectors"][1]["reason"]
    assert all(code != "未知题材" for code, _ in manager.calls)
    bank.loc[79, "volume"] = 10
    result = build_market_brief(manager, overview, {}, now=NOW)
    assert not result["sectors"][0]["verified"]
    assert "不凑推荐" in result["markdown"]


def test_missing_benchmark_does_not_create_rotation_or_recommendations():
    manager = HistoryManager({BENCHMARK: pd.DataFrame()})
    result = build_market_brief(manager, MarketOverview(date=DAY.isoformat(), top_sectors=[{"name": "银行"}]), {}, now=NOW)
    assert result["styles"]["technology"]["available"]
    assert "暂时看不清谁更强" in result["markdown"]
    assert not result["sectors"][0]["verified"]


def test_push_uses_brief_while_files_history_and_merged_report_remain_complete():
    from src.core import market_review as runtime

    manager = HistoryManager()
    brief = build_market_brief(manager, MarketOverview(date=DAY.isoformat()), {}, now=NOW)
    payload = {
        "kind": "market_review", "region": "cn", "language": "zh", "brief": brief,
        "title": "完整大盘复盘", "sections": [{"key": "full_review", "title": "详情", "markdown": "完整分析与新闻"}],
        "markdown_report": "完整分析与新闻", "sectors": {"top": [{"name": "完整榜单", "change_pct": 3}]},
    }
    notifier = create_autospec(NotificationService, instance=True)
    notifier.is_available.return_value = True
    notifier.send.return_value = True
    analyzer = MagicMock()
    analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
        report="完整分析与新闻", structured_payload=payload, market_light_snapshot={},
    )
    config = SimpleNamespace(report_language="zh", market_review_region="cn")
    with patch.object(runtime, "MarketAnalyzer", return_value=analyzer), patch.object(runtime, "_persist_market_review_history") as persist:
        result = runtime.run_market_review(notifier, config=config, send_notification=True)
    assert result == "完整分析与新闻"
    analyzer.run_daily_review_with_snapshot.assert_called_once_with(include_brief=True)
    assert "完整分析与新闻" in notifier.save_report_to_file.call_args.args[0]
    assert "完整分析与新闻" in persist.call_args.kwargs["markdown_report"]
    sent = notifier.send.call_args.args[0]
    assert "大盘简报" in sent and "科技成长" in sent
    assert "完整榜单" not in sent and "完整分析与新闻" not in sent
    push_payload = notifier.send.call_args.kwargs["structured_payload"]
    assert push_payload["presentation"] == "brief"
    assert "presentation" not in payload

    from src.share_image import build_share_image_html

    html = build_share_image_html(sent, structured_payload=push_payload)
    assert "科技成长" in html and "传统价值" in html
    assert "完整榜单" not in html and "完整分析与新闻" not in html


def test_legacy_and_other_market_rendering_are_preserved():
    from src.core.market_review import _render_market_review_payload_markdown

    payload = {"region": "us", "markdown_report": "Existing English report"}
    assert "Existing English report" in _render_market_review_payload_markdown(payload, prefer_brief=True)


@pytest.mark.parametrize("language,include_brief,expected_calls", [("zh", False, 0), ("zh", True, 10), ("en", True, 0), ("ko", True, 0)])
def test_analyzer_only_collects_extra_history_for_requested_chinese_pushes(language, include_brief, expected_calls):
    manager = HistoryManager()
    config = SimpleNamespace(report_language=language)
    with patch("src.market_analyzer.DataFetcherManager", return_value=manager):
        analyzer = MarketAnalyzer(config=config)
    overview = MarketOverview(date=DAY.isoformat())
    with patch.object(analyzer, "get_market_overview", return_value=overview), patch.object(
        analyzer, "search_market_news", return_value=[],
    ), patch.object(analyzer, "_merge_persisted_market_intelligence", return_value=[]), patch.object(
        analyzer, "generate_market_review", return_value="## Full report\n\nExisting analysis",
    ), patch("src.core.market_style.get_market_now", return_value=NOW):
        result = analyzer.run_daily_review_with_snapshot(include_brief=include_brief)
    assert len(manager.calls) == expected_calls
    assert ("brief" in result.structured_payload) == bool(expected_calls)
    assert result.report == "## Full report\n\nExisting analysis"


def test_unverified_calendar_abstains_without_fetching_or_guessing_dates():
    manager = HistoryManager()
    phase = SimpleNamespace(
        effective_daily_bar_date=DAY, warnings=["calendar_unavailable"],
        phase=SimpleNamespace(value="unknown"), is_partial_bar=None,
    )
    with patch("src.core.market_style.build_market_phase_context", return_value=phase):
        result = build_market_brief(manager, MarketOverview(date=DAY.isoformat()), {}, now=NOW)
    assert manager.calls == []
    assert not result["calendar_verified"]
    assert "交易日期无法核验" in result["markdown"]
    assert "交易时点未核验" in result["markdown"]


def test_index_search_codes_are_visible_even_when_style_data_is_unavailable():
    manager = HistoryManager({"sh000688": pd.DataFrame(), "sh000922": pd.DataFrame()})
    result = build_market_brief(manager, MarketOverview(date=DAY.isoformat()), {}, now=NOW)
    assert "科创50（1B0688.SH）" in result["markdown"]
    assert "中证红利（000922.CSI）" in result["markdown"]
    assert "暂无法判断" in result["markdown"]


def test_missing_daily_leaderboard_still_scans_the_bounded_pool_and_deduplicates_indices():
    bank = bars([100 + index for index in range(80)])
    manager = HistoryManager({"sz399986": bank})
    overview = MarketOverview(date=DAY.isoformat(), top_sectors=[])
    result = build_market_brief(manager, overview, {}, now=NOW)
    assert result["sector_scan"] == {"total": 7, "available": 7}
    assert next(item for item in result["sectors"] if item.get("code") == "sz399986")["verified"]
    called_codes = [code for code, _ in manager.calls]
    assert set(called_codes) == {BENCHMARK, "sh000688", "sh000922", *SECTOR_POOL}
    assert len(called_codes) == len(set(called_codes)) == 10

    overview.top_sectors = [{"name": "银行"}, {"name": "银行行业"}, {"name": "未知题材"}]
    result = build_market_brief(HistoryManager({"sz399986": bank}), overview, {}, now=NOW)
    assert len([item for item in result["sectors"] if item.get("code") == "sz399986"]) == 1
    assert not next(item for item in result["sectors"] if item["name"] == "未知题材")["verified"]


def test_selected_sector_uses_an_exact_reviewed_fund_mapping_in_notification_and_image():
    from dataclasses import replace
    from src.core.market_review import _render_market_review_payload_markdown
    from src.share_image import build_share_image_html

    bank = bars([100 + index for index in range(80)])
    reviewed_family = replace(FUND_FAMILIES["sz399986"], reviewed_on=DAY)
    with patch.dict(FUND_FAMILIES, {"sz399986": reviewed_family}):
        result = build_market_brief(HistoryManager({"sz399986": bank}), MarketOverview(date=DAY.isoformat()), {}, now=NOW)
    selected = next(item for item in result["sectors"] if item.get("code") == "sz399986")
    reference = selected["fund_reference"]
    assert reference["index_code"] == selected["code"]
    assert reference["status"] == "reviewed"
    assert reference["platform_availability_verified"] is False
    assert {fund["share_class"] for fund in reference["funds"]} == {"A", "C"}
    payload = {"kind": "market_review", "region": "cn", "brief": result, "presentation": "brief"}
    markdown = _render_market_review_payload_markdown(payload, prefer_brief=True)
    html = build_share_image_html(markdown, structured_payload=payload)
    for content in (markdown, html):
        assert "天弘中证银行ETF联接A" in content
        assert "001594" in content and "001595" in content
        assert "1B0688.SH" in content and "000922.CSI" in content
        assert "上涨延续观察" in content
        assert "以页面为准" in content
        assert DAY.isoformat() in content


def test_watch_or_extended_sectors_do_not_become_fund_recommendations():
    extended = bars([100.0] * 79 + [120.0])
    manager = HistoryManager({"sz399986": extended})
    result = build_market_brief(manager, MarketOverview(date=DAY.isoformat()), {}, now=NOW)
    bank = next(item for item in result["sectors"] if item.get("code") == "sz399986")
    assert bank["history_available"] and not bank["verified"]
    assert "不追涨" in bank["reason"]
    assert "fund_reference" not in bank
    assert "001595" not in result["markdown"]
    assert "不凑推荐" in result["markdown"]


def test_sector_order_uses_relative_strength_and_limits_recommendations_to_three():
    histories = {code: bars([100 + index * slope for index in range(80)])
                 for code, slope in zip(SECTOR_POOL, [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3])}
    result = build_market_brief(HistoryManager(histories), MarketOverview(date=DAY.isoformat()), {}, now=NOW)
    selected = sorted(result["sectors"], key=lambda item: item["relative_5d"], reverse=True)[:3]
    assert [item["name"] for item in selected] == ["钢铁", "白酒", "医疗"]
    markdown = result["markdown"]
    assert markdown.index("**钢铁｜") < markdown.index("**白酒｜") < markdown.index("**医疗｜")
    assert "**银行｜" not in markdown and "**军工｜" not in markdown


@pytest.mark.parametrize("offset", [-1, FUND_REVIEW_MAX_DAYS + 1])
def test_unreviewed_or_expired_fund_identity_is_not_exposed_as_current(offset):
    family = FUND_FAMILIES["sz399986"]
    reference = get_fund_reference("sz399986", family.reviewed_on + timedelta(days=offset))
    assert reference["status"] == "review_required"
    assert reference["funds"] == []
    assert reference["platform_availability_verified"] is False


def test_same_named_theme_does_not_substitute_a_different_or_liquidated_fund():
    reference = get_fund_reference("sh000941", date(2026, 11, 27))
    assert reference["status"] == "unmapped" and reference["funds"] == []
    assert "暂无已核对" in reference["reason"]
    assert "huaan.com.cn" in reference["source_url"]
    # Medical equipment/services index is not the all-share pharmaceuticals index.
    medical = get_fund_reference("sz399989", date(2026, 11, 27))
    assert all("医药卫生" not in fund["name"] for fund in medical["funds"])
    assert get_fund_reference("000941.SH", date(2026, 11, 27))["funds"] == []
