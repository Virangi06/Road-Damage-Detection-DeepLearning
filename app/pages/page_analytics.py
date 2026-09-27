"""
app/pages/page_analytics.py
============================
Analytics — model comparison, training curves, per-class F1, efficiency.
Streamlit 1.64.0 + Plotly 7.1.0 compatible.
All rgba / color values use valid numeric format via theme.py.
"""

import os
import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from app.styles.theme import (
    COLORS, CHART_COLORS, inject_css,
    page_header, section_title, pastel_layout,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _load_json(fn: str) -> dict:
    path = os.path.join(PROJECT_ROOT, fn)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ── Pastel layout helper (light background) ───────────────────────────────────
def _pl(**kw) -> dict:
    return pastel_layout(**kw)


def render():
    inject_css()

    page_header(
        "Model Analytics & Performance",
        "Quantitative benchmark comparison of CNN, ViT, and Hybrid CNN+ViT architectures.",
        "📊",
    )
    st.divider()

    eda = _load_json("phase1_eda_stats.json")

    tabs = st.tabs([
        "📈 Model Comparison",
        "📉 Training History",
        "🗂️ Per-Class F1",
        "⚡ Efficiency",
        "🗃️ Dataset Stats",
    ])

    # ── Tab 1: Model Comparison ───────────────────────────────────────────────
    with tabs[0]:
        section_title("Quantitative Benchmark Comparison")

        accuracies  = [89.0, 92.0, 87.0]
        macro_f1s   = [0.633, 0.6953, 0.6023]
        labels      = ["CNN\n(EfficientNet)", "ViT\n(vit_tiny)", "Hybrid\nCNN+ViT"]

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Bar(
                x=labels, y=accuracies,
                marker=dict(
                    color=CHART_COLORS[:3],
                    line=dict(color="rgba(0,0,0,0.05)", width=1),
                ),
                text=[f"{a:.0f}%" for a in accuracies],
                textposition="outside",
                textfont=dict(color=COLORS["text"]),
            ))
            fig.update_layout(
                title="Validation Accuracy (%)", yaxis_range=[0, 105],
                height=320, **_pl()
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure(go.Bar(
                x=labels, y=macro_f1s,
                marker=dict(
                    color=CHART_COLORS[:3],
                    line=dict(color="rgba(0,0,0,0.05)", width=1),
                ),
                text=[f"{f:.4f}" for f in macro_f1s],
                textposition="outside",
                textfont=dict(color=COLORS["text"]),
            ))
            fig2.update_layout(
                title="Macro F1-Score", yaxis_range=[0, 0.85],
                height=320, **_pl()
            )
            st.plotly_chart(fig2, use_container_width=True)

        section_title("Full Metrics Table")
        st.dataframe(pd.DataFrame({
            "Architecture":    ["CNN Baseline", "ViT Baseline", "Hybrid CNN+ViT"],
            "Accuracy (%)":    [89.0, 92.0, 87.0],
            "Macro Precision": [0.7261, 0.7175, 0.6000],
            "Macro Recall":    [0.6209, 0.6818, 0.6046],
            "Macro F1":        [0.6330, 0.6953, 0.6023],
            "Params (M)":      [4.008,  5.718,  9.727],
            "Latency (ms)":    [12.4,   18.7,   31.2],
            "FPS":             [80.6,   53.5,   32.1],
        }), hide_index=True, use_container_width=True)

    # ── Tab 2: Training History ───────────────────────────────────────────────
    with tabs[1]:
        section_title("Training & Validation Curves — Phase 3")

        models_data = {
            "CNN Baseline":   {
                "epochs":    [1, 2, 3],
                "train_loss":[1.1091, 0.644, 0.4348],
                "val_loss":  [0.6137, 0.4139, 0.3785],
                "train_acc": [67.0, 78.33, 86.0],
                "val_acc":   [86.0, 88.0, 89.0],
            },
            "ViT Baseline":   {
                "epochs":    [1, 2, 3],
                "train_loss":[0.9383, 0.5702, 0.4151],
                "val_loss":  [0.5179, 0.3741, 0.3896],
                "train_acc": [72.67, 78.67, 87.33],
                "val_acc":   [84.0, 92.0, 85.0],
            },
            "Hybrid CNN+ViT": {
                "epochs":    [1, 2, 3],
                "train_loss":[0.874, 0.5464, 0.3406],
                "val_loss":  [0.6258, 0.3527, 0.364],
                "train_acc": [69.0, 80.33, 89.67],
                "val_acc":   [85.0, 87.0, 87.0],
            },
        }
        colors3 = CHART_COLORS[:3]

        cl, cr = st.columns(2)
        with cl:
            fig_loss = go.Figure()
            for (name, data), color in zip(models_data.items(), colors3):
                fig_loss.add_trace(go.Scatter(
                    x=data["epochs"], y=data["val_loss"],
                    name=f"{name} (Val)",
                    line=dict(color=color, width=2),
                    mode="lines+markers",
                    marker=dict(size=7),
                ))
                fig_loss.add_trace(go.Scatter(
                    x=data["epochs"], y=data["train_loss"],
                    name=f"{name} (Train)",
                    line=dict(color=color, width=2, dash="dot"),
                    mode="lines+markers",
                    marker=dict(size=5),
                ))
            fig_loss.update_layout(
                title="Loss Curves", xaxis_title="Epoch",
                yaxis_title="Cross-Entropy Loss", height=360, **_pl()
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with cr:
            fig_acc = go.Figure()
            for (name, data), color in zip(models_data.items(), colors3):
                fig_acc.add_trace(go.Scatter(
                    x=data["epochs"], y=data["val_acc"],
                    name=f"{name} (Val)",
                    line=dict(color=color, width=2),
                    mode="lines+markers",
                    marker=dict(size=7),
                ))
                fig_acc.add_trace(go.Scatter(
                    x=data["epochs"], y=data["train_acc"],
                    name=f"{name} (Train)",
                    line=dict(color=color, width=2, dash="dot"),
                    mode="lines+markers",
                    marker=dict(size=5),
                ))
            fig_acc.update_layout(
                title="Accuracy Curves", xaxis_title="Epoch",
                yaxis_title="Accuracy (%)", height=360, **_pl()
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        section_title("Phase 2 — Faster R-CNN Detection Training")
        st.dataframe(pd.DataFrame({
            "Epoch":        [1, 2],
            "Train Loss":   [0.5440, 0.3075],
            "Val Loss":     [0.3123, 0.2661],
            "LR":           ["3.00e-4", "2.25e-4"],
            "Cls Loss":     [0.1912, 0.1023],
            "Box Reg Loss": [0.0521, 0.0614],
            "Checkpoint":   ["Saved", "✅ Best"],
        }), hide_index=True, use_container_width=True)

    # ── Tab 3: Per-Class F1 ───────────────────────────────────────────────────
    with tabs[2]:
        section_title("Per-Class F1-Score Comparison")

        cls_labels = ["D00\nLongitudinal", "D10\nTransverse",
                      "D20\nAlligator",    "D40\nPothole", "D43-44\nOther"]
        cnn_f1    = [0.9254, 0.9383, 0.4444, 0.8571, 0.0]
        vit_f1    = [0.9677, 0.9639, 0.6667, 0.8780, 0.0]
        hybrid_f1 = [0.9375, 0.9383, 0.2857, 0.8500, 0.0]

        fig_f1 = go.Figure()
        for f1s, name, color in [
            (cnn_f1,    "CNN Baseline",   CHART_COLORS[0]),
            (vit_f1,    "ViT Baseline",   CHART_COLORS[1]),
            (hybrid_f1, "Hybrid CNN+ViT", CHART_COLORS[2]),
        ]:
            fig_f1.add_trace(go.Bar(
                name=name, x=cls_labels, y=f1s,
                marker=dict(
                    color=color,
                    line=dict(color="rgba(0,0,0,0.05)", width=1),
                ),
                text=[f"{v:.3f}" for v in f1s],
                textposition="outside",
                textfont=dict(color=COLORS["text"], size=10),
            ))

        fig_f1.update_layout(
            barmode="group",
            title="Per-Class F1 — All 5 Damage Categories",
            yaxis_range=[0, 1.15], yaxis_title="F1-Score",
            height=420, **_pl()
        )
        st.plotly_chart(fig_f1, use_container_width=True)

        st.info(
            "**ℹ️ D43/D44 (Other Damage): F1 = 0.0 across all models.**  "
            "Root causes: class imbalance (4,628 vs 18,201 samples for D00), "
            "high intra-class visual heterogeneity, and 3-epoch training limit."
        )

        section_title("Detailed Per-Class Metrics")
        cls_full = ["D00 Longitudinal","D10 Transverse",
                    "D20 Alligator","D40 Pothole","D43/D44 Other"]
        st.dataframe(pd.DataFrame({
            "Class":      cls_full,
            "CNN P":      [0.8857,0.9268,1.0000,0.8182,0.0],
            "CNN R":      [0.9688,0.9500,0.2857,0.9000,0.0],
            "CNN F1":     cnn_f1,
            "ViT P":      [1.0000,0.9302,0.8000,0.8571,0.0],
            "ViT R":      [0.9375,1.0000,0.5714,0.9000,0.0],
            "ViT F1":     vit_f1,
            "Hybrid P":   [0.9375,0.9268,0.2857,0.8500,0.0],
            "Hybrid R":   [0.9375,0.9500,0.2857,0.8500,0.0],
            "Hybrid F1":  hybrid_f1,
        }), hide_index=True, use_container_width=True)

    # ── Tab 4: Efficiency ─────────────────────────────────────────────────────
    with tabs[3]:
        section_title("Architecture Efficiency & Accuracy Trade-off")

        names      = ["CNN Baseline", "ViT Baseline", "Hybrid CNN+ViT"]
        accuracies = [89.0, 92.0, 87.0]
        latencies  = [12.4, 18.7, 31.2]
        params     = [4.008, 5.718, 9.727]

        el, er = st.columns(2)
        with el:
            fig_lat = go.Figure()
            for i, name in enumerate(names):
                fig_lat.add_trace(go.Scatter(
                    x=[latencies[i]], y=[accuracies[i]],
                    mode="markers+text",
                    marker=dict(size=20, color=CHART_COLORS[i],
                                line=dict(color="white", width=2)),
                    text=[name], textposition="top center",
                    textfont=dict(color=COLORS["text"], size=10),
                    name=name,
                ))
            fig_lat.update_layout(
                title="Accuracy vs. Latency",
                xaxis_title="Latency (ms/crop)",
                yaxis_title="Accuracy (%)",
                yaxis_range=[84, 95],
                height=360, **_pl()
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        with er:
            fig_par = go.Figure()
            for i, name in enumerate(names):
                fig_par.add_trace(go.Scatter(
                    x=[params[i]], y=[accuracies[i]],
                    mode="markers+text",
                    marker=dict(size=20, color=CHART_COLORS[i],
                                line=dict(color="white", width=2)),
                    text=[name], textposition="top center",
                    textfont=dict(color=COLORS["text"], size=10),
                    name=name,
                ))
            fig_par.update_layout(
                title="Accuracy vs. Parameters",
                xaxis_title="Parameters (Millions)",
                yaxis_title="Accuracy (%)",
                yaxis_range=[84, 95],
                height=360, **_pl()
            )
            st.plotly_chart(fig_par, use_container_width=True)

        st.dataframe(pd.DataFrame({
            "Architecture":  names,
            "Accuracy (%)":  accuracies,
            "Params (M)":    params,
            "File (MB)":     [16.2, 22.8, 38.9],
            "Latency (ms)":  latencies,
            "FPS":           [80.6, 53.5, 32.1],
        }), hide_index=True, use_container_width=True)

    # ── Tab 5: Dataset Stats ──────────────────────────────────────────────────
    with tabs[4]:
        section_title("RDD2022 Dataset Statistics")

        train = eda.get("train", {})
        val   = eda.get("val",   {})
        test  = eda.get("test",  {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Train Images",       f"{train.get('num_images',26869):,}")
        c2.metric("Val Images",         f"{val.get('num_images',5758):,}")
        c3.metric("Test Images",        f"{test.get('num_images',5758):,}")
        c4, c5, c6 = st.columns(3)
        c4.metric("Train Annotations",  f"{train.get('total_boxes',46296):,}")
        c5.metric("Val Annotations",    f"{val.get('total_boxes',9741):,}")
        c6.metric("Test Annotations",   f"{test.get('total_boxes',9675):,}")

        # Split bar
        fig_split = go.Figure(go.Bar(
            x=["Train", "Validation", "Test"],
            y=[26869, 5758, 5758],
            marker=dict(
                color=CHART_COLORS[:3],
                line=dict(color="rgba(0,0,0,0.05)", width=1),
            ),
            text=["26,869", "5,758", "5,758"],
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
        ))
        fig_split.update_layout(
            title="Dataset Split Distribution", yaxis_title="Images",
            height=300, **_pl()
        )
        st.plotly_chart(fig_split, use_container_width=True)

        # Class distribution
        cc_train = train.get("class_counts", {
            "D00 (Longitudinal Crack)": 18201,
            "D10 (Transverse Crack)": 8386,
            "D20 (Alligator Crack)": 7527,
            "D40 (Pothole)": 7554,
            "D43/D44 (Other Damage)": 4628,
        })
        fig_cls = go.Figure(go.Bar(
            x=list(cc_train.values()),
            y=list(cc_train.keys()),
            orientation="h",
            marker=dict(
                color=CHART_COLORS[:5],
                line=dict(color="rgba(0,0,0,0.05)", width=1),
            ),
            text=[f"{v:,}" for v in cc_train.values()],
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
        ))
        fig_cls.update_layout(
            title="Training Set Class Distribution",
            xaxis_title="Annotations", height=280, **_pl()
        )
        st.plotly_chart(fig_cls, use_container_width=True)

        # Box size donut
        sizes_d = train.get("box_sizes", {
            "small (<32×32)": 9893, "medium (32-96×96)": 21869, "large (>96×96)": 14534,
        })
        fig_sz = go.Figure(go.Pie(
            labels=list(sizes_d.keys()),
            values=list(sizes_d.values()),
            hole=0.45,
            marker=dict(
                colors=CHART_COLORS[:3],
                line=dict(color=COLORS["surface"], width=2),
            ),
        ))
        fig_sz.update_layout(
            title="Bounding Box Size Distribution", height=280, **_pl()
        )
        st.plotly_chart(fig_sz, use_container_width=True)
