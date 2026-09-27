"""
app/components/prediction_card.py
Reusable result / prediction display components.
"""
import streamlit as st
from app.styles.theme import COLORS, severity_badge


def severity_banner(category: str, score: float, priority: int) -> str:
    """Return HTML banner for severity result."""
    bg_map = {
        "Low":    COLORS["success"],
        "Medium": COLORS["warning"],
        "High":   COLORS["danger"],
    }
    text_map = {
        "Low":    "#1B5E3B",
        "Medium": "#7A4700",
        "High":   "#7A1523",
    }
    icon_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    bg   = bg_map.get(category, COLORS["success"])
    tc   = text_map.get(category, COLORS["text"])
    icon = icon_map.get(category, "⚪")
    return f"""
    <div style='background:{bg};border-radius:12px;padding:16px 20px;
                margin:8px 0;border:1px solid rgba(0,0,0,0.06);'>
        <div style='color:{tc};font-size:1.3rem;font-weight:800;'>
            {icon} {category.upper()} SEVERITY
        </div>
        <div style='color:{tc};font-size:0.88rem;margin-top:4px;opacity:0.85;'>
            Score: <b>{score:.1f} / 100</b> &nbsp;·&nbsp; Repair Priority: <b>{priority} / 10</b>
        </div>
    </div>"""


def condition_card(label: str) -> str:
    """Return styled overall condition card."""
    icon_map = {"good": "✅", "moderate": "⚠️", "fair": "🔶", "poor": "🚨"}
    icon = "ℹ️"
    for k, v in icon_map.items():
        if k in label.lower():
            icon = v
            break
    return f"""
    <div style='background:{COLORS["surface"]};border:1px solid {COLORS["border"]};
                border-radius:10px;padding:14px 16px;margin-top:10px;'>
        <div style='color:{COLORS["text_muted"]};font-size:0.75rem;font-weight:600;
                    text-transform:uppercase;letter-spacing:0.05em;'>Overall Condition</div>
        <div style='color:{COLORS["text"]};font-size:1rem;font-weight:700;margin-top:6px;'>
            {icon} {label}
        </div>
    </div>"""


def detection_row_html(index: int, damage_type: str, confidence: float,
                        bbox: list) -> str:
    """Return a styled detection row."""
    conf_pct = confidence * 100
    conf_color = COLORS["sev_low"] if conf_pct >= 70 else (
        COLORS["sev_medium"] if conf_pct >= 40 else COLORS["sev_high"])
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return f"""
    <div style='background:{COLORS["surface"]};border:1px solid {COLORS["border"]};
                border-radius:8px;padding:10px 14px;margin:4px 0;
                display:flex;align-items:center;gap:12px;'>
        <span style='background:{COLORS["primary"]};color:{COLORS["text"]};
                     border-radius:50%;width:24px;height:24px;display:inline-flex;
                     align-items:center;justify-content:center;
                     font-size:0.75rem;font-weight:700;flex-shrink:0;'>{index}</span>
        <div style='flex:1;'>
            <div style='color:{COLORS["text"]};font-weight:600;font-size:0.9rem;'>{damage_type}</div>
            <div style='color:{COLORS["text_muted"]};font-size:0.78rem;'>
                Box: ({x1},{y1}) → ({x2},{y2})
            </div>
        </div>
        <span style='color:{conf_color};font-weight:700;font-size:0.9rem;'>
            {conf_pct:.1f}%
        </span>
    </div>"""
