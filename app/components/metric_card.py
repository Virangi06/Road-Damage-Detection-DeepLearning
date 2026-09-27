"""
app/components/metric_card.py
Reusable pastel metric/KPI card helpers.
"""
import streamlit as st
from app.styles.theme import COLORS


def kpi_card(label: str, value: str, delta: str = "", icon: str = "", color: str = None):
    """Render a styled KPI card using st.metric (styled by global CSS)."""
    display_label = f"{icon}  {label}" if icon else label
    st.metric(label=display_label, value=value, delta=delta if delta else None)


def stat_card_html(icon: str, label: str, value: str, sub: str = "",
                   border_color: str = None) -> str:
    """Return a self-contained HTML stat card string."""
    bc = border_color or COLORS["primary"]
    return f"""
    <div style='background:{COLORS["surface"]};border:1px solid {COLORS["border"]};
                border-top:3px solid {bc};border-radius:12px;padding:16px 18px;
                box-shadow:0 2px 8px rgba(0,0,0,0.04);'>
        <div style='font-size:1.6rem;margin-bottom:4px;'>{icon}</div>
        <div style='color:{COLORS["text_muted"]};font-size:0.75rem;font-weight:600;
                    text-transform:uppercase;letter-spacing:0.05em;'>{label}</div>
        <div style='color:{COLORS["text"]};font-size:1.55rem;font-weight:700;
                    margin-top:2px;'>{value}</div>
        <div style='color:{COLORS["text_light"]};font-size:0.76rem;
                    margin-top:3px;'>{sub}</div>
    </div>"""


def model_badge_html(name: str, acc: str, f1: str,
                     color: str = None, best: bool = False) -> str:
    """Return HTML for a model summary badge card."""
    bc = color or COLORS["primary"]
    crown = " 👑" if best else ""
    return f"""
    <div style='background:{COLORS["surface"]};border:1px solid {COLORS["border"]};
                border-left:4px solid {bc};border-radius:10px;padding:14px 16px;
                margin:4px 0;'>
        <div style='color:{COLORS["text"]};font-weight:700;font-size:0.95rem;'>{name}{crown}</div>
        <div style='margin-top:6px;display:flex;gap:12px;'>
            <span style='background:{COLORS["secondary"]};color:{COLORS["text"]};
                         border-radius:6px;padding:2px 10px;font-size:0.8rem;font-weight:600;'>
                🎯 {acc}</span>
            <span style='background:{COLORS["accent"]};color:{COLORS["text"]};
                         border-radius:6px;padding:2px 10px;font-size:0.8rem;font-weight:600;'>
                📐 F1: {f1}</span>
        </div>
    </div>"""
