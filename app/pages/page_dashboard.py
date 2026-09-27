"""
app/pages/page_dashboard.py
============================
Dashboard — project KPIs, dataset stats, model comparison.
Streamlit 1.64.0 + Plotly 7.1.0 compatible.
All fillcolor / rgba values use proper numeric format.
"""

import os
import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from app.styles.theme import (
    COLORS, CHART_COLORS, PLOTLY_FILL,
    inject_css, page_header, section_title, pastel_layout,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _load_json(filename: str):
    path = os.path.join(PROJECT_ROOT, filename)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ── Radar fill colors — VALID Plotly rgba strings (numeric, not hex) ─────────
# chart_1 = #6EC6CA → rgb(110,198,202)
# chart_2 = #7EB8F7 → rgb(126,184,247)
# chart_3 = #C09BD8 → rgb(192,155,216)
_RADAR_LINE  = [CHART_COLORS[0], CHART_COLORS[1], CHART_COLORS[2]]
_RADAR_FILL  = [PLOTLY_FILL["chart_1"], PLOTLY_FILL["chart_2"], PLOTLY_FILL["chart_3"]]


def render():
    inject_css()

    # ── Page header ───────────────────────────────────────────────────────────
    page_header(
        "Road Damage Detection & Severity Assessment",
        "Intelligent AI system for automated road condition monitoring — RDD2022 dataset",
        "🚧",
    )
    st.divider()

    # ── Load data ─────────────────────────────────────────────────────────────
    eda = _load_json("phase1_eda_stats.json")

    total_train = eda.get("train", {}).get("num_images", 26869)
    total_val   = eda.get("val",   {}).get("num_images", 5758)
    total_test  = eda.get("test",  {}).get("num_images", 5758)
    total_images = total_train + total_val + total_test
    total_boxes  = (
        eda.get("train", {}).get("total_boxes", 46296) +
        eda.get("val",   {}).get("total_boxes", 9741)  +
        eda.get("test",  {}).get("total_boxes", 9675)
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📸 Total Images",       f"{total_images:,}")
    k2.metric("📦 Bounding Boxes",     f"{total_boxes:,}")
    k3.metric("🎯 Best Accuracy",      "92.0%",   "ViT Baseline")
    k4.metric("📐 Best Macro F1",      "0.6953",  "ViT Baseline")
    k5.metric("⚡ Fastest Inference",  "80.6 FPS", "CNN Baseline")

    st.divider()

    # ── Main 2-column layout ──────────────────────────────────────────────────
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    # ── Left: charts ──────────────────────────────────────────────────────────
    with col_left:
        section_title("📊 Training Set Class Distribution")

        class_counts = eda.get("train", {}).get("class_counts", {
            "D00 (Longitudinal Crack)": 18201,
            "D10 (Transverse Crack)":   8386,
            "D20 (Alligator Crack)":    7527,
            "D40 (Pothole)":            7554,
            "D43/D44 (Other Damage)":   4628,
        })
        classes = list(class_counts.keys())
        counts  = list(class_counts.values())

        fig_bar = go.Figure(go.Bar(
            x=counts,
            y=classes,
            orientation="h",
            marker=dict(
                color=CHART_COLORS[:len(classes)],
                line=dict(color="rgba(0,0,0,0.05)", width=1),
            ),
            text=[f"{c:,}" for c in counts],
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=11),
        ))
        fig_bar.update_layout(
            height=280,
            **pastel_layout(
                margin=dict(l=0, r=50, t=10, b=10),
                xaxis=dict(showgrid=True, gridcolor=COLORS["border"], title=""),
                yaxis=dict(showgrid=False),
            ),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        section_title("🌍 Image Source Countries")

        country_counts = eda.get("train", {}).get("country_counts", {
            "China": 3051, "Czech": 1962, "India": 5368,
            "Japan": 7432, "Norway": 5708, "United States": 3348,
        })
        fig_pie = go.Figure(go.Pie(
            labels=list(country_counts.keys()),
            values=list(country_counts.values()),
            hole=0.48,
            marker=dict(
                colors=CHART_COLORS[:len(country_counts)],
                line=dict(color=COLORS["surface"], width=2),
            ),
            textfont=dict(size=11, color=COLORS["text"]),
        ))
        fig_pie.update_layout(
            height=260,
            **pastel_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    orientation="h", y=-0.18,
                    font=dict(color=COLORS["text_muted"], size=11),
                    bgcolor=COLORS["surface"],
                ),
            ),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Right: model comparison ───────────────────────────────────────────────
    with col_right:
        section_title("🤖 Model Performance Radar")

        model_names = ["CNN Baseline", "ViT Baseline", "Hybrid CNN+ViT"]
        accuracies  = [89.0, 92.0, 87.0]
        macro_f1s   = [63.3, 69.5, 60.2]    # ×100 for radar scale
        precisions  = [72.6, 71.8, 60.0]
        recalls     = [62.1, 68.2, 60.5]
        categories  = ["Accuracy", "Macro F1", "Precision", "Recall"]

        fig_radar = go.Figure()
        for i, (name, acc, f1, p, r) in enumerate(
                zip(model_names, accuracies, macro_f1s, precisions, recalls)):
            values = [acc, f1, p, r, acc]   # close polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=name,
                line=dict(color=_RADAR_LINE[i], width=2),
                fillcolor=_RADAR_FILL[i],   # ← VALID rgba string from theme
                opacity=0.85,
            ))

        fig_radar.update_layout(
            height=320,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor=COLORS["border"],
                    tickfont=dict(color=COLORS["text_muted"], size=10),
                    tickcolor=COLORS["border"],
                ),
                angularaxis=dict(
                    tickfont=dict(color=COLORS["text"], size=12, family="Inter, sans-serif"),
                    gridcolor=COLORS["border"],
                ),
                bgcolor=COLORS["bg"],
            ),
            paper_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"], family="Inter, sans-serif"),
            legend=dict(
                orientation="h", y=-0.18,
                font=dict(color=COLORS["text"], size=11),
                bgcolor=COLORS["surface"],
            ),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        section_title("📋 Quick Results Table")
        df_show = pd.DataFrame({
            "Model":     model_names,
            "Accuracy":  [f"{a:.0f}%" for a in accuracies],
            "Macro F1":  ["0.6330", "0.6953", "0.6023"],
            "Params":    ["4.01 M",  "5.72 M",  "9.73 M"],
            "Speed":     ["12.4 ms", "18.7 ms", "31.2 ms"],
        })
        st.dataframe(df_show, hide_index=True, use_container_width=True)

        # Model badges
        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
        from app.components.metric_card import model_badge_html
        for name, acc, f1, color, best in [
            ("CNN Baseline (EfficientNet-B0)",  "89.0%", "0.633",  CHART_COLORS[0], False),
            ("ViT Baseline (vit_tiny)",          "92.0%", "0.695",  CHART_COLORS[1], True),
            ("Hybrid CNN+ViT (Proposed)",        "87.0%", "0.602",  CHART_COLORS[2], False),
        ]:
            st.markdown(model_badge_html(name, acc, f1, color, best),
                        unsafe_allow_html=True)

    st.divider()

    # ── How it works ──────────────────────────────────────────────────────────
    section_title("⚙️ How the System Works")

    steps = [
        ("📸", "Upload Image",    "Provide a road photograph"),
        ("🔍", "Detect Damage",   "Faster R-CNN finds bounding boxes"),
        ("🏷️", "Classify Type",   "Hybrid CNN+ViT identifies damage"),
        ("📏", "Assess Severity", "0-100 severity score computed"),
        ("📊", "View Results",    "Annotated image + metrics shown"),
    ]
    step_cols = st.columns(5)
    for col, (icon, title, desc) in zip(step_cols, steps):
        col.markdown(
            f"""<div class='rd-step-card'>
                    <div style='font-size:1.8rem;'>{icon}</div>
                    <div style='font-weight:700;color:{COLORS["text"]};
                                font-size:0.88rem;margin:6px 0 3px;'>{title}</div>
                    <div style='color:{COLORS["text_muted"]};font-size:0.75rem;
                                line-height:1.3;'>{desc}</div>
               </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Severity guide ────────────────────────────────────────────────────────
    section_title("🚦 Severity Classification Guide")
    s1, s2, s3 = st.columns(3)

    def _sev_card(col, bg, border, title_color, title, score_range, desc):
        col.markdown(
            f"""<div style='background:{bg};border:1.5px solid {border};
                            border-radius:12px;padding:18px;'>
                    <div style='color:{title_color};font-size:1.1rem;
                                font-weight:800;'>{title}</div>
                    <div style='color:{title_color};font-size:0.82rem;
                                margin-top:4px;opacity:0.8;'>Score: {score_range}</div>
                    <div style='color:{COLORS["text"]};font-size:0.82rem;
                                margin-top:8px;line-height:1.4;'>{desc}</div>
               </div>""",
            unsafe_allow_html=True,
        )

    _sev_card(s1, COLORS["success"], COLORS["sev_low"],  COLORS["sev_low"],
              "🟢 LOW SEVERITY",    "0 – 29.9",
              "Minor surface cracks. Monitor and schedule routine maintenance.")
    _sev_card(s2, COLORS["warning"], COLORS["sev_medium"], COLORS["sev_medium"],
              "🟡 MEDIUM SEVERITY", "30 – 64.9",
              "Multiple cracks or moderate damage. Schedule maintenance soon.")
    _sev_card(s3, COLORS["danger"],  COLORS["sev_high"],   COLORS["sev_high"],
              "🔴 HIGH SEVERITY",   "65 – 100",
              "Potholes or extensive damage. Immediate repair required.")
