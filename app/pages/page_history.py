"""
app/pages/page_history.py
==========================
Analysis History page — filterable, sortable, with timeline chart.
Streamlit 1.64.0 + Plotly 7.1.0 compatible.
Python 3.11 — no backslashes inside f-string expressions (uses C alias).
"""

import os
import json
import datetime
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from collections import Counter

from app.styles.theme import (
    COLORS, CHART_COLORS, inject_css,
    page_header, section_title, pastel_layout, empty_state,
)

# ── Colour alias — safe in f-strings ─────────────────────────────────────────
C = COLORS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HISTORY_FILE = os.path.join(PROJECT_ROOT, "app", "analysis_history.json")


def _load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _clear_history():
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
        except Exception:
            pass


def _pl(**kw) -> dict:
    return pastel_layout(**kw)


def render():
    inject_css()

    page_header(
        "Analysis History",
        "All previous road image analyses — automatically saved after each inference.",
        "🕒",
    )
    st.divider()

    history = _load_history()

    if not history:
        st.markdown(empty_state(
            "📂",
            "No analysis history yet",
            "Analyse a road image on the Analyze Image page to see results here.",
        ), unsafe_allow_html=True)
        return

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    total   = len(history)
    damaged = sum(1 for h in history if h.get("damage_detected"))
    highs   = sum(1 for h in history if h.get("severity_category") == "High")
    avg_sc  = sum(h.get("severity_score", 0) for h in history) / max(total, 1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Analyses",     total)
    k2.metric("Damage Detected",    damaged, f"{damaged/total*100:.0f}%")
    k3.metric("High Severity",      highs)
    k4.metric("Avg Severity Score", f"{avg_sc:.1f}/100")

    st.divider()

    # ── Filter bar ────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        filter_sev = st.multiselect(
            "Filter by severity",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
        )
    with f2:
        filter_dmg = st.selectbox(
            "Filter by damage detected",
            ["All", "Damaged only", "No damage only"],
        )
    with f3:
        sort_by = st.selectbox(
            "Sort by",
            ["Newest first", "Oldest first", "Highest severity", "Lowest severity"],
        )

    # Apply filters
    filtered = [h for h in history
                if h.get("severity_category", "Low") in filter_sev]
    if filter_dmg == "Damaged only":
        filtered = [h for h in filtered if h.get("damage_detected")]
    elif filter_dmg == "No damage only":
        filtered = [h for h in filtered if not h.get("damage_detected")]

    if sort_by == "Oldest first":
        filtered = list(reversed(filtered))
    elif sort_by == "Highest severity":
        filtered = sorted(filtered, key=lambda x: x.get("severity_score", 0), reverse=True)
    elif sort_by == "Lowest severity":
        filtered = sorted(filtered, key=lambda x: x.get("severity_score", 0))

    n_filtered = len(filtered)
    text_muted = C['text_muted']
    st.markdown(
        f"<div style='color:{text_muted};font-size:0.82rem;margin-bottom:8px;'>"
        f"<b>{n_filtered}</b> records match current filters</div>",
        unsafe_allow_html=True,
    )

    # ── History table ─────────────────────────────────────────────────────────
    _SEV = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    rows = []
    for h in filtered:
        ts = h.get("timestamp", "")
        try:
            dt = datetime.datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            dt = ts[:16] if len(ts) >= 16 else ts
        sev = h.get("severity_category", "Low")
        rows.append({
            "Date/Time":  dt,
            "Image":      h.get("image_name", "?"),
            "Damage":     "Yes" if h.get("damage_detected") else "No",
            "Regions":    h.get("regions", 0),
            "Severity":   _SEV.get(sev, "⚪") + " " + sev,
            "Score":      f"{h.get('severity_score',0):.1f}/100",
            "Priority":   f"{h.get('repair_priority',1)}/10",
            "Condition":  h.get("overall_condition", "—"),
            "ms":         f"{h.get('inference_ms',0):.0f}",
        })

    if rows:
        row_h = min(40 + len(rows) * 35, 420)
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            height=row_h,
        )

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    section_title("📊 History Analytics")

    ch1, ch2 = st.columns(2)

    with ch1:
        sev_counts = Counter(h.get("severity_category", "Low") for h in history)
        fig_sev = go.Figure(go.Pie(
            labels=list(sev_counts.keys()),
            values=list(sev_counts.values()),
            marker=dict(
                colors=[C['success'], C['warning'], C['danger']],
                line=dict(color=C['surface'], width=2),
            ),
            hole=0.48,
            textfont=dict(color=C['text']),
        ))
        fig_sev.update_layout(title="Severity Distribution", height=280,
                               **_pl())
        st.plotly_chart(fig_sev, use_container_width=True)

    with ch2:
        all_types = []
        for h in history:
            all_types.extend(h.get("damage_types") or [])
        if all_types:
            dt_counts = Counter(t for t in all_types if t)
            fig_dt = go.Figure(go.Bar(
                x=list(dt_counts.values()),
                y=list(dt_counts.keys()),
                orientation="h",
                marker=dict(
                    color=CHART_COLORS[1],
                    line=dict(color="rgba(0,0,0,0.05)", width=1),
                ),
                text=list(dt_counts.values()),
                textposition="outside",
                textfont=dict(color=C['text']),
            ))
            fig_dt.update_layout(title="Damage Type Frequency", height=280,
                                  **_pl())
            st.plotly_chart(fig_dt, use_container_width=True)
        else:
            st.info("No damage type data available yet.")

    # Timeline (last 20 chronological)
    if len(history) > 1:
        recent = list(reversed(history[-20:]))
        tl_labels, tl_scores = [], []
        for h in recent:
            ts = h.get("timestamp", "")
            try:
                lbl = datetime.datetime.fromisoformat(ts).strftime("%m-%d %H:%M")
            except Exception:
                lbl = ts[:10]
            tl_labels.append(lbl)
            tl_scores.append(h.get("severity_score", 0))

        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(
            x=tl_labels, y=tl_scores,
            mode="lines+markers",
            line=dict(color=CHART_COLORS[0], width=2),
            marker=dict(size=8, color=CHART_COLORS[0],
                        line=dict(color=C['surface'], width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(110,198,202,0.12)",   # valid numeric rgba
        ))
        fig_tl.add_hline(
            y=30, line_dash="dot", line_color=C['sev_medium'],
            annotation_text="Low/Medium",
            annotation_position="bottom right",
            annotation_font=dict(color=C['sev_medium'], size=10),
        )
        fig_tl.add_hline(
            y=65, line_dash="dot", line_color=C['sev_high'],
            annotation_text="Medium/High",
            annotation_position="bottom right",
            annotation_font=dict(color=C['sev_high'], size=10),
        )
        fig_tl.update_layout(
            title="Severity Score Timeline (Last 20 Analyses)",
            xaxis_title="Analysis Time",
            yaxis_title="Severity Score",
            yaxis_range=[0, 108],
            height=320,
            **_pl(),
        )
        st.plotly_chart(fig_tl, use_container_width=True)

    # ── Clear button ──────────────────────────────────────────────────────────
    st.divider()
    if st.button("🗑️ Clear All History", type="secondary"):
        _clear_history()
        st.success("History cleared.")
        st.rerun()
