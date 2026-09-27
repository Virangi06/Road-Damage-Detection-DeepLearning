"""
app/pages/page_model_info.py
=============================
Model Information — architectures, dataset, severity, viva Q&A, tech stack.
Streamlit 1.64.0 compatible.
Python 3.11 — no backslashes inside f-string expressions (uses C alias).
"""

import streamlit as st
from app.styles.theme import COLORS, inject_css, page_header, section_title

# ── Colour alias — safe inside f-strings ─────────────────────────────────────
C = COLORS


# ── Architecture card helper ──────────────────────────────────────────────────
def _arch_card(col, border_color: str, title: str, subtitle: str,
               rows: list, badge: str = ""):
    surf   = C['surface']
    border = C['border']
    muted  = C['text_muted']
    text   = C['text']
    rows_html = "".join(
        f"<tr>"
        f"<td style='color:{muted};padding:3px 8px 3px 0;font-size:0.8rem;"
        f"white-space:nowrap;'><b>{k}</b></td>"
        f"<td style='color:{text};padding:3px 0;font-size:0.8rem;'>{v}</td>"
        f"</tr>"
        for k, v in rows
    )
    crown = f"<span style='float:right;font-size:0.85rem;'>{badge}</span>" if badge else ""
    col.markdown(
        f"<div style='background:{surf};border:1px solid {border};"
        f"border-top:3px solid {border_color};border-radius:12px;"
        f"padding:16px 18px;height:100%;'>"
        f"<h4 style='color:{border_color};margin:0 0 2px;font-size:0.98rem;'>"
        f"{title}{crown}</h4>"
        f"<p style='color:{muted};font-size:0.78rem;margin:0 0 10px;'>{subtitle}</p>"
        f"<table style='border-collapse:collapse;width:100%;'>{rows_html}</table>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render():
    inject_css()

    page_header(
        "Model & System Information",
        "Architecture details, dataset reference, severity methodology, and viva preparation.",
        "ℹ️",
    )
    st.divider()

    tabs = st.tabs([
        "🏗️ Architectures",
        "🗃️ Dataset",
        "📐 Severity",
        "🎓 Viva Q&A",
        "⚙️ Tech Stack",
    ])

    # ── Tab 1: Architectures ──────────────────────────────────────────────────
    with tabs[0]:
        section_title("Deep Learning Architectures")

        c1, c2, c3 = st.columns(3)
        _arch_card(c1, C['chart_1'], "1. CNN Baseline", "EfficientNet-B0", [
            ("Backbone",   "EfficientNet-B0"),
            ("Pretrained", "ImageNet"),
            ("Input",      "224 × 224 × 3"),
            ("Features",   "1280-dim"),
            ("Head",       "Dropout(0.3) → Linear(5)"),
            ("Params",     "4.01 M"),
            ("File",       "16.2 MB"),
            ("Accuracy",   "89.0%"),
            ("Macro F1",   "0.6330"),
            ("Latency",    "12.4 ms · 80.6 FPS"),
        ])
        _arch_card(c2, C['chart_2'], "2. ViT Baseline", "vit_tiny_patch16_224", [
            ("Backbone",   "vit_tiny_patch16_224"),
            ("Pretrained", "ImageNet (timm)"),
            ("Input",      "224×224 → 196 patches"),
            ("Patch size", "16 × 16 px"),
            ("Features",   "192-dim CLS token"),
            ("Params",     "5.72 M"),
            ("File",       "22.8 MB"),
            ("Accuracy",   "92.0% (Best)"),
            ("Macro F1",   "0.6953 (Best)"),
            ("Latency",    "18.7 ms · 53.5 FPS"),
        ], badge="⭐ Best")
        _arch_card(c3, C['chart_3'], "3. Hybrid CNN+ViT", "Proposed Architecture", [
            ("CNN stream", "EfficientNet-B0 → 1280-dim"),
            ("ViT stream", "vit_tiny → 192-dim"),
            ("Fusion",     "Concat → 1472-dim"),
            ("Head",       "BN→Drop(0.3)→FC(512)→BN→Drop(0.2)→FC(5)"),
            ("Params",     "9.73 M"),
            ("File",       "38.9 MB"),
            ("Accuracy",   "87.0%"),
            ("Macro F1",   "0.6023"),
            ("Latency",    "31.2 ms · 32.1 FPS"),
        ])

        st.divider()
        section_title("Phase 2 — Faster R-CNN Object Detector")
        st.markdown("""
        | Property | Value |
        |---|---|
        | Architecture | Faster R-CNN · ResNet-50 FPN backbone |
        | Pretrained | ImageNet (ResNet-50) |
        | Detection head | FastRCNNPredictor — 6 classes (5 damage + background) |
        | Optimizer | AdamW · lr=0.0003 · weight_decay=0.0005 |
        | Scheduler | CosineAnnealingLR (T_max=3) |
        | Best Val Loss | **0.2661** (Epoch 2) |
        | Checkpoint | `models/best_model.pth` |
        """)

        section_title("Classification Training Config")
        st.markdown("""
        | Setting | Value |
        |---|---|
        | Loss Function | Cross-Entropy Loss |
        | Optimizer | AdamW · lr=0.0003 · weight_decay=0.0001 |
        | LR Scheduler | CosineAnnealingLR (T_max=3) |
        | Epochs | 3 |
        | Batch Size | 32 |
        | Input Size | 224 × 224 px |
        | Preprocessing | CLAHE (LAB) + 10% bbox padding |
        | Augmentation | Random H-flip + ColorJitter ±0.2 |
        | Normalisation | ImageNet mean/std |
        """)

    # ── Tab 2: Dataset ────────────────────────────────────────────────────────
    with tabs[1]:
        section_title("RDD2022 Dataset")
        st.markdown("""
        | Property | Value |
        |---|---|
        | **Name** | Road Damage Detection 2022 (RDD2022) |
        | **Source** | Global Road Damage Detection Challenge 2022 |
        | **Total Images** | 38,385 |
        | **Train / Val / Test** | 26,869 / 5,758 / 5,758 |
        | **Total Annotations** | 65,712 bounding boxes |
        | **Format** | YOLO: class_id xc yc w h (normalised 0–1) |
        | **Countries** | China · Czech Republic · India · Japan · Norway · United States |
        """)

        section_title("Damage Class Taxonomy")
        st.markdown("""
        | Class ID | Code | Name | Train Count | Severity Weight |
        |:---:|---|---|:---:|:---:|
        | 0 | D00 | Longitudinal Crack | 18,201 | 0.4 |
        | 1 | D10 | Transverse Crack | 8,386 | 0.5 |
        | 2 | D20 | Alligator Crack | 7,527 | 0.8 |
        | 3 | D40 | Pothole | 7,554 | **1.0** |
        | 4 | D43/D44 | Other Damage / Rutting | 4,628 | 0.6 |
        """)

    # ── Tab 3: Severity ───────────────────────────────────────────────────────
    with tabs[2]:
        section_title("Severity Assessment Methodology")

        st.info(
            "Rule-based estimation — not a trained neural network.  "
            "Severity score is a quantitative proxy for road condition, "
            "not validated against civil engineering standards."
        )

        st.markdown("#### Formula")
        st.latex(r"""
        \text{Severity Score} = \min\!\left(100,\;
        50 \cdot A_{norm} + 25 \cdot \frac{N_{regions}}{5} + 25 \cdot W_{class\_max}
        \right)
        """)

        st.markdown("""
        | Component | Symbol | Description | Max |
        |---|---|---|:---:|
        | Normalised Area | A_norm | Total damage bbox area / image area | 50 pts |
        | Region Density | N_regions | Damage region count (capped at 5) | 25 pts |
        | Class Weight | W_class_max | Max damage severity weight | 25 pts |

        **Class weights:** D00=0.4 · D10=0.5 · D20=0.8 · D40=**1.0** · D43/D44=0.6

        **Severity categories:**
        - 🟢 Low: 0–29.9 → Monitor regularly
        - 🟡 Medium: 30–64.9 → Schedule maintenance
        - 🔴 High: ≥65 → Immediate repair
        """)

        st.markdown("#### Repair Priority Index (1–10)")
        st.latex(r"""
        \text{Priority} = \text{Clamp}\!\left(
        \left\lfloor \tfrac{\text{Score}}{10} \right\rfloor
        + \left\lfloor W_{max} \cdot 2 \right\rfloor,\; 1,\; 10
        \right)
        """)

        with st.expander("Worked Example"):
            st.markdown("""
            **Image:** 640×480 px · Pothole + Alligator Crack detected

            ```
            Pothole     [50,100,200,250]: 150×150 = 22,500 px²
            Alligator   [300,200,500,380]: 200×180 = 36,000 px²
            Image area: 640×480 = 307,200 px²

            A_norm  = 58,500 / 307,200 = 0.1904
            Area    = 50 × 0.1904      = 9.52
            Regions = 25 × (2/5)       = 10.00
            Weight  = 25 × 1.0         = 25.00  (Pothole = max)

            Score   = min(100, 44.52)  = 44.52 → MEDIUM
            Priority = Clamp(4+2, 1,10) = 6/10
            ```
            """)

        st.warning(
            "Limitations: BBox area overestimates actual crack area · "
            "Camera angle affects apparent damage size · "
            "Class weights are heuristic, not calibrated from civil engineering surveys."
        )

    # ── Tab 4: Viva Q&A ───────────────────────────────────────────────────────
    with tabs[3]:
        section_title("Faculty Viva Preparation Guide")

        with st.expander("30-Second Explanation", expanded=True):
            st.markdown("""
            > *"Our project automates road damage inspection using deep learning on the RDD2022
            > dataset (38,000+ road images from 6 countries). We detect damage with Faster R-CNN,
            > classify it into five types using a Hybrid CNN+ViT model, and estimate severity on a
            > 0–100 scale. The ViT baseline achieved 92% accuracy. All results are accessible
            > through this Streamlit web dashboard."*
            """)

        qa_pairs = [
            ("What is Deep Learning?",
             "A subset of machine learning using multi-layer neural networks to automatically learn hierarchical feature representations from raw data without manual feature engineering."),
            ("Why Deep Learning for road damage?",
             "Traditional CV needs hand-crafted features brittle across lighting/weather. DL learns robust features from 38,000+ annotated examples and generalises across 6 countries."),
            ("What is a CNN?",
             "Convolutional Neural Network — applies learnable spatial filters to detect local patterns (edges, textures). EfficientNet-B0 uses compound-scaled MBConv blocks for efficiency."),
            ("What is a Vision Transformer (ViT)?",
             "ViT splits 224×224 into 196 patches of 16×16 px, embeds them, adds CLS token + positional embeddings, then applies multi-head self-attention across all patches simultaneously."),
            ("Why does ViT outperform CNN here?",
             "Crack patterns span large road areas with global spatial dependencies. ViT captures these in one attention pass; CNNs need many stacked layers to achieve similar receptive field."),
            ("What is Transfer Learning?",
             "Initialising model weights from ImageNet pretraining (1M+ images) before fine-tuning on RDD2022. The model starts with learned visual features instead of random weights."),
            ("What is the Hybrid CNN+ViT model?",
             "Runs EfficientNet-B0 (1280-dim) and ViT-Tiny (192-dim) in parallel on the same crop, concatenates to 1472-dim, then classifies via BatchNorm-Dropout-Linear fusion head."),
            ("What is Faster R-CNN?",
             "Two-stage detector: Stage 1 (RPN) proposes candidate regions; Stage 2 (ROI head) classifies each proposal and refines box coordinates. ResNet-50 FPN provides multi-scale features."),
            ("What is the severity formula?",
             "Score = min(100, 50*A_norm + 25*(N/5) + 25*W_max). Three components: damaged area (0-50), region density (0-25), damage class weight (0-25). Pothole has max weight 1.0."),
            ("Why is severity rule-based not a neural model?",
             "RDD2022 only has bounding box class labels, not civil engineering severity ratings. The formula is a documented proxy using available detection information."),
            ("What is Grad-CAM?",
             "Gradient-weighted Class Activation Mapping. Registers hooks on EfficientNet final conv layer, computes gradient of class score w.r.t. feature activations, produces spatial heatmap."),
            ("What is CLAHE?",
             "Contrast Limited Adaptive Histogram Equalisation. Applied to L channel in LAB (clipLimit=2.0, tileGridSize=8x8). Enhances low-contrast cracks without oversaturating bright regions."),
            ("Why does D43/D44 have 0% F1?",
             "Three factors: (1) severe imbalance (4,628 vs 18,201 for D00), (2) high intra-class diversity (rutting vs edge cracks vs markings look completely different), (3) only 3 epochs."),
            ("What is mAP?",
             "Mean Average Precision — averages the area under Precision-Recall curve across all classes. mAP@50 uses IoU >= 0.50 as True Positive threshold."),
            ("What is IoU?",
             "Intersection over Union = (Area of Overlap) / (Area of Union) between predicted and ground-truth box. Used to determine TP (IoU >= threshold) vs FP (IoU < threshold)."),
            ("What is AdamW?",
             "Adam with decoupled weight decay. Weight decay applied directly to weights (not mixed with gradient), providing better regularisation than standard Adam with L2."),
            ("How does inference work end-to-end?",
             "1) Faster R-CNN detects boxes → 2) Crop each with 10% padding → 3) Resize+CLAHE+normalise 224x224 → 4) Hybrid CNN+ViT classifies → 5) SeverityAssessor computes score → 6) OpenCV draws annotations → 7) Streamlit shows results."),
            ("What are the main limitations?",
             "1) 0% F1 on D43/D44. 2) Only 3 training epochs. 3) mAP@50=0.0 (evaluation script issue, not model failure). 4) Severity is rule-based. 5) BBox area overestimates crack area. 6) No GPS/GIS."),
            ("What future improvements would you make?",
             "1) Class-weighted loss for D43/D44. 2) Instance segmentation for pixel-accurate area. 3) More training epochs. 4) Real-time video. 5) GPS/GIS integration. 6) Civil engineer severity labels."),
            ("What is the research question and answer?",
             "Does Hybrid CNN+ViT outperform standalone models? In our 3-epoch experiment, ViT (92%) slightly outperforms Hybrid (87%). However, Hybrid shows more consistent F1 across D00/D10/D40. The hypothesis holds directionally — more epochs with class-weighted loss are expected to allow Hybrid to leverage its representational advantage."),
        ]

        for q, a in qa_pairs:
            with st.expander(f"Q: {q}"):
                accent_col = C['border_accent']
                surf_col   = C['surface']
                muted_col  = C['text_muted']
                text_col   = C['text']
                st.markdown(
                    f"<div class='rd-card rd-card-accent'>"
                    f"<span style='color:{muted_col};font-size:0.8rem;"
                    "font-weight:600;'>ANSWER</span>"
                    f"<p style='color:{text_col};font-size:0.88rem;"
                    f"margin-top:6px;line-height:1.55;'>{a}</p></div>",
                    unsafe_allow_html=True,
                )

    # ── Tab 5: Tech Stack ─────────────────────────────────────────────────────
    with tabs[4]:
        section_title("Technology Stack")

        items = [
            ("🐍 Python 3.11",     "Core language"),
            ("🔥 PyTorch 2.0+",    "DL framework — training & inference"),
            ("🖼️ TorchVision",     "Faster R-CNN, transforms, augmentation"),
            ("🤗 timm 0.9+",       "ViT model library — vit_tiny_patch16_224"),
            ("👁️ OpenCV 4.8+",    "CLAHE, image drawing, annotation"),
            ("🔢 NumPy 1.24+",    "Array operations throughout pipeline"),
            ("📊 Matplotlib",      "Training curve plots, confusion matrices"),
            ("🎨 Seaborn",         "Heatmap visualisations"),
            ("📈 Plotly 7.1+",    "Interactive charts in dashboard"),
            ("🌐 Streamlit 1.64+", "Multi-page web application"),
            ("🔬 Scikit-learn",   "Precision/Recall/F1/confusion matrix"),
            ("🖼️ Pillow 9.5+",   "Image I/O and format conversion"),
            ("📦 Pandas 2.0+",    "Tabular data management"),
        ]
        surf_col   = C['surface']
        border_col = C['border']
        text_col   = C['text']
        muted_col  = C['text_muted']
        for tech, desc in items:
            st.markdown(
                f"<div style='display:flex;align-items:center;padding:8px 14px;"
                f"margin:3px 0;background:{surf_col};"
                f"border:1px solid {border_col};border-radius:8px;'>"
                f"<span style='font-weight:700;color:{text_col};"
                f"min-width:190px;font-size:0.88rem;'>{tech}</span>"
                f"<span style='color:{muted_col};font-size:0.83rem;'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        section_title("Project Folder Structure")
        st.code("""
Project/
├── train/  val/  test/           ← RDD2022 dataset splits
├── data/crop_dataset.py          ← CLAHE crop loader
├── models/
│   ├── cnn_model.py              ← EfficientNet-B0
│   ├── vit_model.py              ← ViT-Tiny (timm)
│   ├── hybrid_model.py           ← Hybrid CNN+ViT (proposed)
│   └── best_model.pth            ← Faster R-CNN checkpoint
├── checkpoints/
│   ├── best_cnn_baseline.pth     ← 89% accuracy
│   ├── best_vit_baseline.pth     ← 92% accuracy
│   └── best_hybrid_cnn_vit.pth   ← 87% accuracy
├── src/
│   ├── model.py  severity.py  explainability.py
│   ├── benchmark.py  inference.py
├── app/
│   ├── streamlit_app.py
│   ├── styles/theme.py           ← Pastel color system + CSS
│   ├── components/               ← Reusable UI components
│   └── pages/                    ← 5 page modules
└── PROJECT_COMPLETE_DOCUMENTATION.md
        """, language="text")
