"""
app/pages/page_history.py
==========================
Analysis History page — shows previous analysis sessions stored in
app/analysis_history.json with ability to filter, sort, and re-view.
"""

import os
import json
import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HISTORY_FILE = os.path.join(PROJECT_ROOT, 'app', 'analysis_history.json')


def _load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except Exception:
            return []


def _clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


def _dark_layout(**kwargs) -> dict:
    base = dict(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1', size=12),
        xaxis=dict(showgrid=True, gridcolor='#1e293b'),
        yaxis=dict(showgrid=True, gridcolor='#1e293b'),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    base.update(kwargs)
    return base


def render():
    st.markdown("""
    <h2 style='background:linear-gradient(90deg,#0ea5e9,#6366f1);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               font-size:1.9rem;font-weight:800;'>
        🕒 Analysis History
    </h2>
    <p style='color:#94a3b8;'>
        All previous image analysis sessions are stored here automatically.
        Each entry shows the image analysed, damage detected, and severity assessment.
    </p>
    """, unsafe_allow_html=True)
    st.divider()

    history = _load_history()

    if not history:
        st.markdown("""
        <div style='border:2px dashed #334155;border-radius:14px;padding:48px;
                    text-align:center;background:#0f172a;'>
            <div style='font-size:3rem;'>📂</div>
            <div style='color:#64748b;font-size:1rem;margin-top:12px;'>
                No analysis sessions yet.<br>
                Go to <b>🔍 Analyze Image</b> to run your first analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    total   = len(history)
    damaged = sum(1 for h in history if h.get('damage_detected'))
    highs   = sum(1 for h in history if h.get('severity_category') == 'High')
    avg_score = sum(h.get('severity_score', 0) for h in history) / max(total, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Analyses", total)
    c2.metric("Damage Detected", damaged, f"{damaged/total*100:.0f}%")
    c3.metric("High Severity", highs)
    c4.metric("Avg Severity Score", f"{avg_score:.1f}/100")

    st.divider()

    # ── Filter controls ───────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_sev = st.multiselect(
            "Filter by severity",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
        )
    with col_f2:
        filter_dmg = st.selectbox("Filter by damage", ["All", "Damaged only", "No damage only"])
    with col_f3:
        sort_by = st.selectbox("Sort by", ["Newest first", "Oldest first",
                                            "Highest severity", "Lowest severity"])

    # Apply filters
    filtered = [h for h in history if h.get('severity_category', 'Low') in filter_sev]
    if filter_dmg == "Damaged only":
        filtered = [h for h in filtered if h.get('damage_detected')]
    elif filter_dmg == "No damage only":
        filtered = [h for h in filtered if not h.get('damage_detected')]

    if sort_by == "Oldest first":
        filtered = list(reversed(filtered))
    elif sort_by == "Highest severity":
        filtered = sorted(filtered, key=lambda x: x.get('severity_score', 0), reverse=True)
    elif sort_by == "Lowest severity":
        filtered = sorted(filtered, key=lambda x: x.get('severity_score', 0))

    st.markdown(f"**{len(filtered)} records** matching filters:")

    # ── History table ─────────────────────────────────────────────────────────
    _SEV_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    rows = []
    for h in filtered:
        ts  = h.get('timestamp', '')
        try:
            dt = datetime.datetime.fromisoformat(ts).strftime('%Y-%m-%d %H:%M')
        except Exception:
            dt = ts[:16] if len(ts) >= 16 else ts

        sev_cat = h.get('severity_category', 'Low')
        rows.append({
            "Date & Time":      dt,
            "Image":            h.get('image_name', 'Unknown'),
            "Damage":           "✅ Yes" if h.get('damage_detected') else "❌ No",
            "Regions":          h.get('regions', 0),
            "Severity":         f"{_SEV_EMOJI.get(sev_cat,'⚪')} {sev_cat}",
            "Score":            f"{h.get('severity_score',0):.1f}/100",
            "Priority":         f"{h.get('repair_priority',1)}/10",
            "Condition":        h.get('overall_condition', 'Unknown'),
            "Inference (ms)":   f"{h.get('inference_ms',0):.0f}",
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True, height=400)

    # ── History charts ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📊 History Analytics")

    chart_l, chart_r = st.columns(2)

    with chart_l:
        sev_counts = {"Low": 0, "Medium": 0, "High": 0}
        for h in history:
            cat = h.get('severity_category', 'Low')
            sev_counts[cat] = sev_counts.get(cat, 0) + 1

        fig_sev = go.Figure(go.Pie(
            labels=list(sev_counts.keys()),
            values=list(sev_counts.values()),
            marker_colors=['#22c55e', '#f59e0b', '#ef4444'],
            hole=0.45,
        ))
        fig_sev.update_layout(title="Severity Distribution", height=280, **_dark_layout())
        st.plotly_chart(fig_sev, use_container_width=True)

    with chart_r:
        damage_types_all = []
        for h in history:
            damage_types_all.extend(h.get('damage_types', []))

        if damage_types_all:
            from collections import Counter
            dt_counts = Counter(damage_types_all)
            fig_dt = go.Figure(go.Bar(
                x=list(dt_counts.values()),
                y=list(dt_counts.keys()),
                orientation='h',
                marker_color='#6366f1',
                text=list(dt_counts.values()),
                textposition='outside',
            ))
            fig_dt.update_layout(title="Damage Type Frequency", height=280, **_dark_layout())
            st.plotly_chart(fig_dt, use_container_width=True)
        else:
            st.info("No damage type data available yet.")

    # Severity score timeline
    if len(history) > 1:
        timestamps = []
        scores     = []
        for h in reversed(history[-20:]):  # last 20 entries chronological
            ts = h.get('timestamp', '')
            try:
                dt = datetime.datetime.fromisoformat(ts).strftime('%m-%d %H:%M')
            except Exception:
                dt = ts[:10]
            timestamps.append(dt)
            scores.append(h.get('severity_score', 0))

        fig_timeline = go.Figure(go.Scatter(
            x=timestamps, y=scores,
            mode='lines+markers',
            line=dict(color='#0ea5e9', width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(14,165,233,0.1)',
        ))
        fig_timeline.add_hline(y=30, line_dash='dot', line_color='#f59e0b',
                                annotation_text="Low/Medium threshold",
                                annotation_position="bottom right")
        fig_timeline.add_hline(y=65, line_dash='dot', line_color='#ef4444',
                                annotation_text="Medium/High threshold",
                                annotation_position="bottom right")
        fig_timeline.update_layout(
            title="Severity Score Timeline (Last 20 Analyses)",
            xaxis_title="Analysis Time", yaxis_title="Severity Score",
            yaxis_range=[0, 105], height=320, **_dark_layout(),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    # ── Clear history ─────────────────────────────────────────────────────────
    st.divider()
    if st.button("🗑️ Clear All History", type="secondary"):
        _clear_history()
        st.success("History cleared.")
        st.rerun()
