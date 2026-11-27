# -*- coding: utf-8 -*-
"""A-share style observations derived from completed, dated index bars."""

import logging
import math
from datetime import date, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from data_provider.base import DataFetchError
from src.core.trading_calendar import build_market_phase_context, get_market_now, get_trading_dates
from src.data.market_fund_catalog import get_fund_reference
from src.services.stock_list_parser import ParseStatus, parse_analysis_target

logger = logging.getLogger(__name__)

STYLE_INDICES = {
    "technology": ("sh000688", "科创50", "科技成长"),
    "value": ("sh000922", "中证红利", "传统价值／红利"),
}
STYLE_DISPLAY_CODES = {"technology": "1B0688.SH", "value": "000922.CSI"}
BENCHMARK = "sh000300"
# Only registered industry indices are eligible for historical verification.
SECTOR_INDICES = {
    "银行": "sz399986", "银行行业": "sz399986",
    "证券": "sz399975", "证券公司": "sz399975", "证券行业": "sz399975",
    "新能源": "sh000941", "军工": "sz399967", "国防军工": "sz399967",
    "医疗": "sz399989", "医疗服务": "sz399989", "白酒": "sz399997",
    "钢铁": "930606.CSI", "钢铁行业": "930606.CSI",
}
# Scan each registered direction even when the daily leaderboard is unavailable.
SECTOR_POOL = {}
for _sector_name, _sector_code in SECTOR_INDICES.items():
    SECTOR_POOL.setdefault(_sector_code, _sector_name)
STAGES = {
    "top_risk": ("🟠 顶部预警，尚未确认", "之前涨过一段，现在开始涨不动，要留意继续回落。"),
    "bottom_watch": ("🟡 筑底观察，尚未确认", "跌了之后有回稳迹象，还需要多观察，不能说已经跌完。"),
    "recovering": ("🟢 企稳信号增强", "从低位反弹后放量突破近期高位，但仍要留意再次回落。"),
    "rising": ("🟢 上涨延续", "近期走势还算稳，先看能不能继续保持。"),
    "falling": ("🔴 仍在走弱", "近期仍然偏弱，暂时还没看到足够的企稳迹象。"),
    "range": ("🟡 方向未明", "上涨和下跌的信号还不一致，先观察。"),
}


def prepare_daily_bars(frame: pd.DataFrame, expected_date: date) -> pd.DataFrame:
    """Reject bad/stale bars; never consume an incomplete or future session."""
    required = {"date", "close", "high", "low", "volume"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError("日线字段不完整")
    bars = frame[list(required)].copy()
    parsed = pd.to_datetime(bars["date"], errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError("日线日期无法核验")
    bars["date"] = parsed.dt.date
    bars = bars[bars["date"] <= expected_date].sort_values("date").reset_index(drop=True)
    if bars.empty or bars["date"].iloc[-1] != expected_date:
        raise ValueError(f"日线尚未更新到 {expected_date.isoformat()}")
    if bars["date"].duplicated().any():
        raise ValueError("同一日期有重复日线")
    bars = bars.tail(90).reset_index(drop=True)
    if len(bars) < 61:
        raise ValueError("不足 61 根完整日线，暂无法判断")
    sessions = get_trading_dates("cn", bars["date"].iloc[0], expected_date)
    if sessions is None or bars["date"].tail(61).tolist() != list(sessions.date)[-61:]:
        raise ValueError("日线交易日期不连续或无法核验")
    for column in ("close", "high", "low", "volume"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
        if not bars[column].map(lambda value: pd.notna(value) and math.isfinite(value)).all():
            raise ValueError("日线包含无效数值")
    if (
        (bars[["close", "high", "low"]] <= 0).any().any()
        or (bars["volume"] < 0).any()
        or (bars["high"] < bars["close"]).any()
        or (bars["low"] > bars["close"]).any()
    ):
        raise ValueError("日线价格或成交量异常")
    return bars


def classify_style(bars: pd.DataFrame) -> Dict[str, Any]:
    """Classify a 20/60-bar price phase without claiming an absolute top/bottom."""
    close = bars["close"]
    previous = bars.iloc[:-1]
    price = float(close.iloc[-1])
    ma5 = float(close.tail(5).mean())
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    previous_ma5 = float(previous["close"].tail(5).mean())
    previous_ma20 = float(previous["close"].tail(20).mean())
    support = float(previous["low"].tail(20).min())
    resistance = float(previous["high"].tail(20).max())
    volume_base = float(previous["volume"].tail(20).mean())
    volume_ratio = float(bars["volume"].iloc[-1]) / volume_base if volume_base > 0 else None
    recent = close.tail(60).reset_index(drop=True)
    near_peak = int(recent.idxmax()) >= len(recent) - 10
    near_trough = int(recent.idxmin()) >= len(recent) - 10
    drawdown = price / float(recent.max()) - 1
    rebound = price / float(recent.min()) - 1
    stage = "range"
    if near_peak and drawdown <= -0.03 and price < ma20 and ma5 < previous_ma5:
        stage = "top_risk"
    elif near_trough and rebound >= 0.03 and price > ma5 >= previous_ma5 and price < ma20:
        stage = "bottom_watch"
    elif (
        price > resistance and volume_ratio is not None and volume_ratio >= 1.3
        and ma20 > previous_ma20 and float(recent.min()) / float(recent.max()) <= 0.9
        and int(recent.idxmin()) > int(recent.idxmax())
    ):
        stage = "recovering"
    elif price > ma20 > ma60 and ma20 > previous_ma20:
        stage = "rising"
    elif price < ma20 < ma60 and ma20 < previous_ma20:
        stage = "falling"
    return {
        "stage": stage, "price": round(price, 2), "date": bars["date"].iloc[-1].isoformat(),
        "support": round(support, 2), "resistance": round(resistance, 2),
        "ma20": round(ma20, 2), "ma60": round(ma60, 2),
        "return_5d": round((price / float(close.iloc[-6]) - 1) * 100, 2),
        "return_20d": round((price / float(close.iloc[-21]) - 1) * 100, 2),
        "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
    }


def relative_strength(bars: pd.DataFrame, benchmark: pd.DataFrame, periods: int = 5) -> Optional[float]:
    """Compare identical dated sessions, not mismatched row offsets."""
    left, right = bars.tail(periods + 1), benchmark.tail(periods + 1)
    if len(left) != periods + 1 or left["date"].tolist() != right["date"].tolist():
        return None
    return round(
        (float(left["close"].iloc[-1]) / float(left["close"].iloc[0])
         - float(right["close"].iloc[-1]) / float(right["close"].iloc[0])) * 100, 2,
    )


def build_market_brief(manager: Any, overview: Any, light: Dict[str, Any], *, now=None) -> Dict[str, Any]:
    """Collect bounded histories through the existing source chain; disclose each missing input."""
    now = now or get_market_now("cn")
    phase = build_market_phase_context(market="cn", current_time=now)
    expected = phase.effective_daily_bar_date
    calendar_ok = not phase.warnings and phase.phase.value != "unknown"
    histories: Dict[str, pd.DataFrame] = {}
    sources: Dict[str, str] = {}
    index_names: Dict[str, str] = {}
    failures: Dict[str, str] = {}

    def load(code: str) -> Optional[pd.DataFrame]:
        if code in histories:
            return histories[code]
        if code in failures:
            return None
        if not calendar_ok:
            failures[code] = "交易日期无法核验，暂无法判断"
            return None
        try:
            target = parse_analysis_target(code)
            if target.asset_type != ParseStatus.INDEX:
                raise ValueError("代表指数未登记，暂无法判断")
            frame, source = manager.get_daily_data(
                code, start_date=(expected - timedelta(days=200)).isoformat(),
                end_date=expected.isoformat(), days=120, analysis_target=target,
            )
            histories[code] = prepare_daily_bars(frame, expected)
            sources[code] = str(source)
            index_names[code] = target.matched_index.display_name if target.matched_index else code
            return histories[code]
        except (DataFetchError, ValueError, TimeoutError, ConnectionError) as exc:
            logger.warning("Market style history unavailable: code=%s error_type=%s", code, type(exc).__name__)
            failures[code] = str(exc) if isinstance(exc, ValueError) else "行情读取失败，暂无法判断"
            return None

    benchmark = load(BENCHMARK)
    styles = {}
    for key, (code, name, label) in STYLE_INDICES.items():
        bars = load(code)
        observation = {
            "code": code, "name": name, "label": label, "available": bars is not None,
            "display_code": STYLE_DISPLAY_CODES[key],
        }
        if bars is None:
            observation["reason"] = failures[code]
        else:
            observation.update(classify_style(bars))
            observation["previous_stage"] = classify_style(bars.iloc[:-1])["stage"]
            observation["source"] = sources[code]
            observation["relative_5d"] = relative_strength(bars, benchmark) if benchmark is not None else None
        styles[key] = observation

    candidates = []
    seen = set()
    # A daily leaderboard is a candidate list, never sufficient for a recommendation.
    sector_inputs = list(overview.top_sectors or [])[:5] + [{"name": name} for name in SECTOR_POOL.values()]
    for sector in sector_inputs:
        if not isinstance(sector, dict):
            continue
        name = str(sector.get("name") or "").strip()
        code = SECTOR_INDICES.get(name)
        identity = code or name
        if not name or identity in seen:
            continue
        seen.add(identity)
        item = {
            "name": name, "verified": False, "history_available": False,
            "reason": "只知道当天涨得靠前，还没核实持续性",
        }
        if code:
            item["code"] = code
            bars = load(code) if benchmark is not None else None
            if bars is not None and benchmark is not None:
                signal = classify_style(bars)
                strength = relative_strength(bars, benchmark)
                extended = float(bars["close"].iloc[-1]) / float(bars["close"].tail(20).mean()) - 1 >= 0.1
                verified = (
                    signal["stage"] in {"rising", "recovering"}
                    and signal["price"] > signal["ma20"]
                    and signal["volume_ratio"] is not None and signal["volume_ratio"] >= 1.0
                    and strength is not None and strength > 0
                    and not extended
                )
                item.update(signal, relative_5d=strength, verified=verified, history_available=True)
                item["index_name"] = index_names[code]
                item["reason"] = "走势和成交量还没同时确认，先观察"
                if extended:
                    item["reason"] = "已高于近20日均价一成以上，先等回落站稳，不追涨"
                elif verified:
                    item["outlook"] = "企稳转强观察" if signal["stage"] == "recovering" else "上涨延续观察"
                    item["reason"] = (
                        "低位后放量突破，留意转强能否持续" if signal["stage"] == "recovering"
                        else "走势向上，成交量也跟上了，留意强势能否延续"
                    )
                    item["fund_reference"] = get_fund_reference(code, now.date())
                elif signal["stage"] == "bottom_watch":
                    item["reason"] = "低位开始回稳，但还没确认转强，先等放量突破"
                item["source"] = sources[code]
            elif code in failures:
                item["reason"] = failures[code]
            elif benchmark is None:
                item["reason"] = "缺少同期沪深300数据，先观察"
        candidates.append(item)

    result = {
        "generated_at": now.isoformat(), "data_date": expected.isoformat(),
        "is_partial_bar": phase.is_partial_bar, "calendar_verified": calendar_ok,
        "phase": phase.phase.value,
        "styles": styles, "sectors": candidates,
        "benchmark_available": benchmark is not None,
        "sector_scan": {
            "total": len(SECTOR_POOL),
            "available": sum(item["history_available"] for item in candidates if item.get("code")),
        },
    }
    result["markdown"] = render_market_brief(result, light)
    return result


def render_market_brief(brief: Dict[str, Any], light: Dict[str, Any]) -> str:
    """Keep the push plain, compact, and traceable to actual index observations."""
    generated = pd.Timestamp(brief["generated_at"])
    if light.get("data_quality") == "unavailable" or not light:
        market_line = "大盘数据不够，目前整体强弱暂时看不清。"
    else:
        status = light.get("status")
        market_line = {
            "green": "盘面整体偏强，但还要看强势能不能持续。",
            "yellow": "盘面涨跌比较分散，方向还不一致。",
            "red": "盘面整体偏弱，先留意能不能止住跌势。",
        }.get(status, "目前方向还不明确，先观察。")
    phase_label = {
        "premarket": "盘前观察", "intraday": "盘中观察", "lunch_break": "午间观察",
        "closing_auction": "临近收盘观察", "postmarket": "收盘后观察",
        "non_trading": "非交易日观察",
    }.get(brief.get("phase"), "交易时点未核验")
    if not brief.get("calendar_verified"):
        phase_label = "交易时点未核验"
    lines = [
        f"## 📊 大盘简报｜{generated:%Y-%m-%d %H:%M}（北京时间）",
        f"{phase_label} · 根据收盘数据判断，截至 {brief['data_date']}", "",
        f"**一句话：{market_line}**", "",
    ]
    changes = []
    for key, icon in (("technology", "💻"), ("value", "🏦")):
        item = brief["styles"][key]
        index_label = f"{item['name']}（{item.get('display_code', STYLE_DISPLAY_CODES[key])}）"
        if not item["available"]:
            lines.extend([f"**{icon} {item['label']}：暂无法判断**", f"看 {index_label}：{item['reason']}", ""])
            continue
        stage_title, explanation = STAGES[item["stage"]]
        support_text = (
            f"已经低于此前低位 **{item['support']:.2f} 点**，需要更小心"
            if item["price"] < item["support"]
            else f"跌破近期低位 **{item['support']:.2f} 点**要更小心"
        )
        resistance_text = (
            f"目前高于此前高位 **{item['resistance']:.2f} 点**，还要看能不能保持住"
            if item["price"] > item["resistance"]
            else f"站上近期高位 **{item['resistance']:.2f} 点**且成交放大，转强信号才更足"
        )
        lines.extend([
            f"**{icon} {item['label']}：{stage_title}**", explanation,
            f"看 {index_label}：最近收盘 {item['price']:.2f} 点；近五日 {item['return_5d']:+.2f}%。",
            f"{support_text}；{resistance_text}。", "",
        ])
        previous = item["previous_stage"]
        current = item["stage"]
        if previous == current:
            changes.append(f"{item['label']}维持“{STAGES[current][0].split(' ', 1)[-1]}”")
        else:
            changes.append(f"{item['label']}从“{STAGES[previous][0].split(' ', 1)[-1]}”转为“{stage_title.split(' ', 1)[-1]}”")
    tech, value = brief["styles"]["technology"], brief["styles"]["value"]
    if (
        tech.get("available") and value.get("available")
        and tech.get("relative_5d") is not None and value.get("relative_5d") is not None
    ):
        difference = tech["relative_5d"] - value["relative_5d"]
        if abs(difference) < 1:
            rotation = "近五日两边差别不大，还看不出明显切换。"
        else:
            leader = "科技" if difference > 0 else "传统价值／红利"
            rotation = f"近五日{leader}相对更强，还不能据此说资金已经转向。"
    else:
        rotation = "两边的同期日线没有齐全，暂时看不清谁更强。"
    lines.extend(["**🔄 谁更强？**", rotation, "", "**⭐ 哪些板块值得留意？**"])
    recommendations = sorted(
        [item for item in brief["sectors"] if item["verified"]],
        key=lambda item: item["relative_5d"], reverse=True,
    )[:3]
    fund_review_dates = set()
    if recommendations:
        for item in recommendations:
            lines.extend([
                f"**{item['name']}｜{item.get('outlook', '上涨延续观察')}**",
                f"{item['reason']}；近五日比沪深300多涨 {item['relative_5d']:.2f} 个百分点。",
                f"看{item['index_name']}能否放量站稳 {item['resistance']:.2f} 点，跌破 {item['support']:.2f} 点则重新评估。",
            ])
            reference = item.get("fund_reference", {})
            if reference.get("status") == "reviewed":
                funds_text = "／".join(f"{fund['name']}（{fund['code']}）" for fund in reference["funds"])
                lines.append(f"对应基金参考，支付宝搜代码：{funds_text}。")
                fund_review_dates.add(reference["reviewed_on"])
            else:
                lines.append(reference.get("reason", "暂无已核对的同指数场外基金") + "。")
            lines.append("")
    else:
        lines.append("暂没有趋势和量能都符合条件的板块，不凑推荐。")
    watch = [item for item in brief["sectors"] if not item["verified"]][:2]
    if watch:
        lines.extend(["", "先观察：" + "；".join(f"{item['name']}（{item['reason']}）" for item in watch) + "。"])
    lines.extend(["", "**📌 比上一交易日**", "；".join(changes) + "。" if changes else "完整日线不足，暂时不能比较。", ""])
    lines.append("科创50、中证红利只是两类风格的代表；未核验组内上涨比例。关键点位来自此前20个完整交易日，仅供观察。")
    scan = brief.get("sector_scan")
    if scan:
        lines.append(f"本次检查银行、证券、新能源、军工、医疗、白酒、钢铁，{scan['available']}/{scan['total']} 个方向日线有效；不是全市场排名，也不保证上涨。")
    if fund_review_dates:
        lines.append(
            f"基金资料核对：{'、'.join(sorted(fund_review_dates))}。A/C是同系列不同收费份额；"
            "支付宝上架、申购状态及费率以页面为准，留意持有期和赎回费。"
        )
    return "\n".join(lines)
