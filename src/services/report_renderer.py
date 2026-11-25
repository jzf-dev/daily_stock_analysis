# -*- coding: utf-8 -*-
"""
===================================
Report Engine - Jinja2 Report Renderer
===================================

Renders reports from Jinja2 templates. Falls back to caller's logic on template
missing or render error. Template path is relative to project root.
Any expensive data preparation should be injected by the caller via extra_context.
"""

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analyzer import AnalysisResult
from src.config import get_config
from src.market_phase_summary import format_public_market_status_line, format_public_phase_pack_excerpt
from src.report_language import (
    format_market_snapshot_heading,
    format_report_metric,
    format_report_price,
    format_volume_analysis,
    get_localized_stock_name,
    get_report_labels,
    get_signal_level,
    get_chip_unavailable_reason,
    is_chip_structure_unavailable,
    localize_chip_health,
    localize_conflict_severity,
    localize_consensus_level,
    localize_operation_advice,
    localize_strategy_signal,
    localize_strategy_skill,
    localize_strategy_conflict_description,
    localize_strategy_synthesis_summary,
    localize_trend_prediction,
    normalize_report_language,
    normalize_strategy_synthesis_payload,
    strategy_invalid_opinion_count,
)
from src.schemas.decision_action import (
    display_action_fields_for_result,
    display_decision_type_for_result,
    display_operation_advice_for_result,
    localize_action_label,
)
from src.utils.data_processing import (
    normalize_model_used,
    signal_attribution_has_content,
    signal_attribution_weight_items,
)

logger = logging.getLogger(__name__)


def _brief_text(value: Any, language: Optional[str] = None) -> str:
    """Keep complete statements, including negations and conditions, on one line."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return ""
    if isinstance(value, (int, float)) and not math.isfinite(value):
        return ""
    text = " ".join(str(value).split()).replace("**", "").replace("__", "")
    if text.lower() in {"", "none", "null", "nan", "inf", "n/a", "暂无", "无", "-"}:
        return ""
    if language and normalize_report_language(language) == "zh":
        # Explain existing terms without dropping negations or inventing thresholds.
        for term, explanation in (
            ("检查减仓计划", "判断是否卖出一部分"),
            ("确认承接情况", "判断买盘能否支撑价格"),
            ("承接", "买盘能否支撑价格"),
            ("兑现条件", "卖出条件"),
            ("减仓", "卖出一部分"),
            ("加仓", "追加买入"),
        ):
            text = text.replace(term, explanation)
    return text


def build_brief_stock_body(result: AnalysisResult, report_language: str) -> str:
    """Share the beginner-readable brief between built-in and template reports.

    This is a presentation of the existing decision, not a new signal evaluator.
    No market requests, model calls, inferred order sizes or price triggers occur.
    """
    from src.analysis_context_pack_prompt import CORE_DEGRADED_STATUSES
    from src.phase_decision_guardrail import CORE_DATA_BLOCKS

    labels = get_report_labels(report_language)
    if not getattr(result, "success", True):
        return labels["brief_failed"]

    def block(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
        value = parent.get(key)
        return value if isinstance(value, dict) else {}

    def statements(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return list(dict.fromkeys(text for item in value if (text := _brief_text(item, report_language))))

    dashboard = result.dashboard if isinstance(result.dashboard, dict) else {}
    core = block(dashboard, "core_conclusion")
    phase = block(dashboard, "phase_decision")
    battle = block(dashboard, "battle_plan")
    sniper = block(battle, "sniper_points")
    strategy = block(battle, "position_strategy")
    overview = getattr(result, "analysis_context_pack_overview", None)
    overview = overview if isinstance(overview, dict) else {}
    blocks = overview.get("blocks")
    blocks = [item for item in blocks if isinstance(item, dict)] if isinstance(blocks, list) else []
    core_degraded = any(
        item.get("key") in CORE_DATA_BLOCKS and item.get("status") in CORE_DEGRADED_STATUSES
        for item in blocks
    ) or block(overview, "data_quality").get("level") == "poor"

    action_fields = display_action_fields_for_result(result, report_language=report_language)
    action = action_fields["action"]
    no_position_key = {
        "buy": "brief_entry_wait", "add": "brief_add_wait",
    }.get(action, "brief_no_buy")
    has_position_key = {
        "buy": "brief_holder_watch", "add": "brief_holder_add",
        "reduce": "brief_holder_reduce", "sell": "brief_holder_sell",
        "avoid": "brief_holder_risk", "alert": "brief_holder_risk",
    }.get(action, "brief_holder_watch")
    # Quote time and the corresponding price lead each stock's message. A daily
    # close or a retrieval timestamp must not masquerade as a realtime quote.
    quote = next((item for item in blocks if item.get("key") == "quote"), {})
    quote_time = _brief_text(quote.get("provider_timestamp"))
    if quote_time:
        try:
            if "T" not in quote_time:
                raise ValueError("Missing datetime separator")
            quote_time = datetime.fromisoformat(quote_time.replace("Z", "+00:00")).isoformat(
                sep=" ", timespec="minutes",
            )
        except ValueError:
            quote_time = ""
    try:
        price = float(result.current_price)
    except (TypeError, ValueError):
        price = 0.0
    has_quote_price = bool(quote_time) and math.isfinite(price) and price > 0
    lines: List[str] = [
        f"🕒 {labels['brief_quote_time']}：{quote_time or labels['brief_unknown_time']}",
        f"💰 {labels['brief_price']}："
        f"{format_report_price(result.current_price, result.code) if has_quote_price else labels['brief_unknown_price']}",
    ]
    snapshot = getattr(result, "market_snapshot", None)
    if not has_quote_price and isinstance(snapshot, dict) and _brief_text(snapshot.get("date")):
        bar_price = format_report_price(snapshot.get("close"), result.code)
        lines.append(f"{labels['brief_bar']}：{bar_price} / {_brief_text(snapshot['date'])}")
    score = format_report_metric(result.sentiment_score, precision=0)
    score_text = score if score == "N/A" else f"{score}/100"
    advice = labels.get(f"brief_action_{action}") or display_operation_advice_for_result(
        result, report_language=report_language,
    )
    decision_type = display_decision_type_for_result(result, report_language=report_language)
    advice_icon = {"buy": "🟢", "hold": "🟡", "sell": "🔴"}[decision_type]
    lines.extend(["", f"{advice_icon} {labels['brief_advice']}：{advice}"])
    lines.append(f"{labels['brief_score']}：{score_text}")
    lines.append("")
    if core_degraded:
        lines.append(labels["brief_data_warning"])
        no_position_key = "brief_no_buy"
        has_position_key = "brief_holder_risk"
    lines.extend([
        f"{labels['brief_no_position']}：{labels[no_position_key]}",
        f"{labels['brief_has_position']}：{labels[has_position_key]}",
    ])

    def reference_price(key: str) -> str:
        value = _brief_text(sniper.get(key), report_language)
        if not value:
            return ""
        cleaned = _clean_sniper_value(value)
        try:
            number = float(cleaned)
        except ValueError:
            return cleaned  # Preserve complete conditional price descriptions.
        if not math.isfinite(number) or number <= 0:
            return ""
        return format_report_price(cleaned, result.code)

    # A price in an old or contradictory plan must not become a new buy signal.
    if action in {"buy", "add"} and not core_degraded:
        entry = reference_price("ideal_buy")
        if entry:
            lines.extend(["", f"{labels['brief_entry']}：{entry}"])

    for rule_key, point_key, label_key, missing_key, icon in (
        ("sell_condition", "take_profit", "brief_sell_condition", "brief_missing_sell", "💸"),
        ("stop_loss_condition", "stop_loss", "brief_stop_condition", "brief_missing_stop", "⚠️"),
    ):
        rule = _brief_text(strategy.get(rule_key), report_language)
        if not rule:
            rule = labels[missing_key]
            reference = reference_price(point_key)
            if reference:
                rule = f"{rule} {labels['brief_reference'].format(price=reference)}"
        lines.extend(["", f"{icon} {labels[label_key]}：{rule}"])

    # Keep additional explicit constraints, especially negations. Never replace
    # them with a generic reassessment instruction or a price-only trigger.
    conditions = statements(phase.get("watch_conditions"))
    if not conditions and not any(strategy.get(key) for key in ("sell_condition", "stop_loss_condition")):
        conditions = statements(battle.get("action_checklist"))
    if conditions:
        lines.extend(["", f"{labels['brief_conditions']}：{'；'.join(conditions[:3])}"])

    reason = _brief_text(core.get("one_sentence"), report_language) or _brief_text(
        result.analysis_summary, report_language,
    )
    if reason:
        lines.extend(["", f"{labels['brief_reason']}：{reason}"])
    risks = statements(block(dashboard, "intelligence").get("risk_alerts"))
    if not risks and _brief_text(result.risk_warning, report_language):
        risks = [_brief_text(result.risk_warning, report_language)]
    limitations = statements(phase.get("data_limitations"))
    risk_notes = list(dict.fromkeys(risks[:3] + limitations[:3]))
    if risk_notes:
        lines.extend(["", f"{labels['brief_risks']}：{'；'.join(risk_notes)}"])
    validity = _brief_text(strategy.get("validity"), report_language) or labels["brief_missing_validity"]
    lines.extend(["", f"{labels['brief_validity']}：{validity}"])
    return "\n".join(lines)


def _escape_md(text: str) -> str:
    """Escape markdown special chars (*ST etc)."""
    if not text:
        return ""
    return text.replace("*", "\\*").replace("_", "\\_")


def _clean_sniper_value(val: Any) -> str:
    """Format sniper point value for display (strip label prefixes)."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val).strip() if val else ""
    if not s or s == "N/A":
        return s or "N/A"
    prefixes = [
        "理想买入点：", "次优买入点：", "止损位：", "目标位：",
        "理想买入点:", "次优买入点:", "止损位:", "目标位:",
        "Ideal Entry:", "Secondary Entry:", "Stop Loss:", "Target:",
    ]
    for prefix in prefixes:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _resolve_templates_dir() -> Path:
    """Resolve template directory relative to project root."""
    config = get_config()
    base = Path(__file__).resolve().parent.parent.parent
    templates_dir = Path(config.report_templates_dir)
    if not templates_dir.is_absolute():
        return base / templates_dir
    return templates_dir


from src.services.empty_news import empty_news_disclosure


def render(
    platform: str,
    results: List[AnalysisResult],
    report_date: Optional[str] = None,
    summary_only: bool = False,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Render report using Jinja2 template.

    Args:
        platform: One of: markdown, wechat, brief
        results: List of AnalysisResult
        report_date: Report date string (default: today)
        summary_only: Whether to output summary only
        extra_context: Additional template context

    Returns:
        Rendered string, or None on error (caller should fallback).
    """
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        logger.warning("jinja2 not installed, report renderer disabled")
        return None

    if report_date is None:
        report_date = datetime.now().strftime("%Y-%m-%d")

    templates_dir = _resolve_templates_dir()
    template_name = f"report_{platform}.j2"
    template_path = templates_dir / template_name
    if not template_path.exists():
        logger.debug("Report template not found: %s", template_path)
        return None

    report_language = normalize_report_language(
        (extra_context or {}).get("report_language")
        or next(
            (getattr(result, "report_language", None) for result in results if getattr(result, "report_language", None)),
            None,
        )
        or getattr(get_config(), "report_language", "zh")
    )
    labels = get_report_labels(report_language)

    # Build template context with pre-computed signal levels (sorted by score)
    sorted_results = sorted(results, key=lambda x: x.sentiment_score, reverse=True)
    sorted_enriched = []
    for r in sorted_results:
        display_action = display_action_fields_for_result(
            r,
            report_language=report_language,
        )["action"]
        display_advice = display_operation_advice_for_result(
            r,
            report_language=report_language,
        )
        signal_action = {
            "buy": "buy",
            "add": "buy",
            "hold": "hold",
            "reduce": "reduce",
            "sell": "sell",
            "watch": "watch",
            "avoid": "hold",
            "alert": "sell",
        }.get(display_action, display_action)
        _, se, _ = get_signal_level(signal_action or display_advice, r.sentiment_score, report_language)
        rn = get_localized_stock_name(r.name, r.code, report_language)
        sorted_enriched.append({
            "result": r,
            "empty_news_disclosure": empty_news_disclosure(r, report_language),
            "signal_text": display_advice,
            "signal_emoji": se,
            "stock_name": _escape_md(rn),
            "localized_operation_advice": display_advice,
            "localized_trend_prediction": localize_trend_prediction(r.trend_prediction, report_language),
            "brief_body": build_brief_stock_body(r, report_language) if platform == "brief" else "",
        })

    display_buckets = [
        display_decision_type_for_result(r, report_language=report_language)
        for r in results
    ]
    buy_count = sum(1 for bucket in display_buckets if bucket == "buy")
    sell_count = sum(1 for bucket in display_buckets if bucket == "sell")
    hold_count = len(display_buckets) - buy_count - sell_count
    show_llm_model = bool(getattr(get_config(), "report_show_llm_model", True))
    models_used: List[str] = []
    if show_llm_model:
        for result in results:
            model = normalize_model_used(getattr(result, "model_used", None))
            if model:
                models_used.append(model)
        models_used = list(dict.fromkeys(models_used))

    report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def failed_checks(checklist: List[str]) -> List[str]:
        return [c for c in (checklist or []) if c.startswith("❌") or c.startswith("⚠️")]

    def phase_pack_excerpt(result: AnalysisResult) -> str:
        return format_public_phase_pack_excerpt(
            getattr(result, "market_phase_summary", None),
            getattr(result, "analysis_context_pack_overview", None),
            source=getattr(result, "analysis_visibility_source", None) or "evaluator_snapshot",
            report_language=report_language,
        )

    def market_status_line() -> str:
        for source_results in (results or [], sorted_results):
            for result in source_results:
                line = format_public_market_status_line(
                    getattr(result, "market_phase_summary", None),
                    report_language=report_language,
                )
                if line:
                    return line
        return ""

    context: Dict[str, Any] = {
        "report_date": report_date,
        "report_timestamp": report_timestamp,
        "results": sorted_results,
        "enriched": sorted_enriched,  # Sorted by sentiment_score desc
        "summary_only": summary_only,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "labels": labels,
        "format_market_snapshot_heading": format_market_snapshot_heading,
        "format_report_metric": format_report_metric,
        "format_report_price": format_report_price,
        "format_volume_analysis": format_volume_analysis,
        "report_language": report_language,
        "models_used": models_used,
        "show_llm_model": show_llm_model,
        "market_status_line": market_status_line(),
        "escape_md": _escape_md,
        "clean_sniper": _clean_sniper_value,
        "failed_checks": failed_checks,
        "phase_pack_excerpt": phase_pack_excerpt,
        "history_by_code": {},
        "get_chip_unavailable_reason": get_chip_unavailable_reason,
        "is_chip_structure_unavailable": is_chip_structure_unavailable,
        "localize_operation_advice": localize_operation_advice,
        "localize_action_label": localize_action_label,
        "localize_trend_prediction": localize_trend_prediction,
        "localize_chip_health": localize_chip_health,
        "localize_strategy_signal": localize_strategy_signal,
        "localize_strategy_skill": localize_strategy_skill,
        "localize_strategy_conflict_description": localize_strategy_conflict_description,
        "localize_strategy_synthesis_summary": localize_strategy_synthesis_summary,
        "localize_consensus_level": localize_consensus_level,
        "localize_conflict_severity": localize_conflict_severity,
        "normalize_strategy_synthesis_payload": normalize_strategy_synthesis_payload,
        "strategy_invalid_opinion_count": strategy_invalid_opinion_count,
        "signal_attribution_has_content": signal_attribution_has_content,
        "signal_attribution_weight_items": signal_attribution_weight_items,
    }
    if extra_context:
        safe_extra_context = dict(extra_context)
        safe_extra_context.pop("labels", None)
        safe_extra_context.pop("report_language", None)
        context.update(safe_extra_context)

    try:
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(default=False),
        )
        template = env.get_template(template_name)
        return template.render(**context)
    except Exception as e:
        logger.warning("Report render failed for %s: %s", template_name, e)
        return None
