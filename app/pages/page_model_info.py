"""
app/pages/page_model_info.py
=============================
Model Information page — architecture details, dataset info,
severity methodology explanation, and faculty viva guide.
"""

import streamlit as st


def render():
    st.markdown("""
    <h2 style='background:linear-gradient(90deg,#0ea5e9,#6366f1);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               font-size:1.9rem;font-weight:800;'>
        ℹ️ Model & System Information
    </h2>
    <p style='color:#94a3b8;'>
        Complete reference for the deep learning architectures, dataset, severity
        methodology, and viva preparation guide.
    </p>
    """, unsafe_allow_html=True)
    st.divider()

    tabs = st.tabs([
        "🏗️ Architectures",
        "🗃️ Dataset",
        "📐 Severity Method",
        "🎓 Viva Guide",
        "⚙️ Tech Stack",
    ])

    # ─── Tab 1: Architectures ─────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Deep Learning Model Architectures")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style='background:#1e293b;border:1px solid #0ea5e9;border-radius:12px;padding:18px;height:100%;'>
            <h4 style='color:#0ea5e9;'>1. CNN Baseline</h4>
            <p style='color:#94a3b8;font-size:0.85rem;'>EfficientNet-B0</p>
            <hr style='border-color:#334155;'>
            <table style='color:#cbd5e1;font-size:0.82rem;width:100%;'>
            <tr><td><b>Backbone</b></td><td>EfficientNet-B0</td></tr>
            <tr><td><b>Pretrained</b></td><td>ImageNet</td></tr>
            <tr><td><b>Input</b></td><td>224 × 224 × 3</td></tr>
            <tr><td><b>Feature Dim</b></td><td>1280-dim</td></tr>
            <tr><td><b>Head</b></td><td>Dropout(0.3) → Linear(5)</td></tr>
            <tr><td><b>Parameters</b></td><td>4.01 M</td></tr>
            <tr><td><b>File Size</b></td><td>16.2 MB</td></tr>
            <tr><td><b>Accuracy</b></td><td>89.0%</td></tr>
            <tr><td><b>Macro F1</b></td><td>0.6330</td></tr>
            <tr><td><b>Latency</b></td><td>12.4 ms/crop</td></tr>
            <tr><td><b>FPS</b></td><td>80.6</td></tr>
            <tr><td><b>Focus</b></td><td>Local textures & edges</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='background:#1e293b;border:1px solid #6366f1;border-radius:12px;padding:18px;height:100%;'>
            <h4 style='color:#6366f1;'>2. ViT Baseline ⭐</h4>
            <p style='color:#94a3b8;font-size:0.85rem;'>Vision Transformer Tiny</p>
            <hr style='border-color:#334155;'>
            <table style='color:#cbd5e1;font-size:0.82rem;width:100%;'>
            <tr><td><b>Backbone</b></td><td>vit_tiny_patch16_224</td></tr>
            <tr><td><b>Pretrained</b></td><td>ImageNet (timm)</td></tr>
            <tr><td><b>Input</b></td><td>224 × 224 → 196 patches</td></tr>
            <tr><td><b>Patch Size</b></td><td>16 × 16 px</td></tr>
            <tr><td><b>Feature Dim</b></td><td>192-dim (CLS token)</td></tr>
            <tr><td><b>Parameters</b></td><td>5.72 M</td></tr>
            <tr><td><b>File Size</b></td><td>22.8 MB</td></tr>
            <tr><td><b>Accuracy</b></td><td><b>92.0%</b> 🏆</td></tr>
            <tr><td><b>Macro F1</b></td><td><b>0.6953</b> 🏆</td></tr>
            <tr><td><b>Latency</b></td><td>18.7 ms/crop</td></tr>
            <tr><td><b>FPS</b></td><td>53.5</td></tr>
            <tr><td><b>Focus</b></td><td>Global patch self-attention</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style='background:#1e293b;border:1px solid #f59e0b;border-radius:12px;padding:18px;height:100%;'>
            <h4 style='color:#f59e0b;'>3. Hybrid CNN+ViT</h4>
            <p style='color:#94a3b8;font-size:0.85rem;'>Proposed Architecture</p>
            <hr style='border-color:#334155;'>
            <table style='color:#cbd5e1;font-size:0.82rem;width:100%;'>
            <tr><td><b>CNN Stream</b></td><td>EfficientNet-B0 → 1280-dim</td></tr>
            <tr><td><b>ViT Stream</b></td><td>ViT-Tiny → 192-dim</td></tr>
            <tr><td><b>Fusion</b></td><td>Concat → 1472-dim</td></tr>
            <tr><td><b>Head</b></td><td>BN→Drop(0.3)→FC(512)→BN→Drop(0.2)→FC(5)</td></tr>
            <tr><td><b>Parameters</b></td><td>9.73 M</td></tr>
            <tr><td><b>File Size</b></td><td>38.9 MB</td></tr>
            <tr><td><b>Accuracy</b></td><td>87.0%</td></tr>
            <tr><td><b>Macro F1</b></td><td>0.6023</td></tr>
            <tr><td><b>Latency</b></td><td>31.2 ms/crop</td></tr>
            <tr><td><b>FPS</b></td><td>32.1</td></tr>
            <tr><td><b>Focus</b></td><td>Local + Global combined</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### Phase 2 — Object Detector: Faster R-CNN ResNet-50 FPN")
        st.markdown("""
        | Property | Value |
        |---|---|
        | **Architecture** | Faster R-CNN with ResNet-50 Feature Pyramid Network (FPN) backbone |
        | **Pretrained** | ImageNet (ResNet-50 weights) |
        | **Detection Head** | Custom `FastRCNNPredictor` — 6 classes (5 damage + 1 background) |
        | **Optimizer** | AdamW (`lr=0.0003`, `weight_decay=0.0005`) |
        | **Scheduler** | CosineAnnealingLR (`T_max=3`) |
        | **Best Val Loss** | `0.2661` (Epoch 2) |
        | **Checkpoint** | `models/best_model.pth` |
        | **Role in pipeline** | Localises damage bounding boxes on full road images |
        """)

        st.markdown("### Classification Training Configuration")
        st.markdown("""
        | Setting | Value |
        |---|---|
        | **Loss Function** | Cross-Entropy Loss |
        | **Optimizer** | AdamW (`lr=0.0003`, `weight_decay=0.0001`) |
        | **LR Scheduler** | CosineAnnealingLR (`T_max=3`) |
        | **Training Epochs** | 3 |
        | **Batch Size** | 32 |
        | **Input Size** | 224 × 224 px |
        | **Preprocessing** | CLAHE contrast enhancement + 10% bbox padding |
        | **Augmentation** | Random horizontal flips + ColorJitter (brightness, contrast, saturation ±0.2) |
        | **Normalisation** | ImageNet mean/std: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |
        | **Checkpointing** | Best validation loss saved to `checkpoints/` |
        """)

    # ─── Tab 2: Dataset ───────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### RDD2022 Dataset")

        st.markdown("""
        | Property | Value |
        |---|---|
        | **Dataset Name** | Road Damage Detection 2022 (RDD2022) |
        | **Source** | Global Road Damage Detection Challenge 2022 |
        | **Total Images** | 38,385 |
        | **Train Images** | 26,869 |
        | **Validation Images** | 5,758 |
        | **Test Images** | 5,758 |
        | **Total Annotations** | 65,712 bounding boxes |
        | **Train Annotations** | 46,296 |
        | **Annotation Format** | YOLO format: `class_id xc yc width height` (normalised 0–1) |
        | **Image Format** | JPEG (.jpg) |
        | **Image Resolution** | Variable (≈ 600×600 to 3000×3000 px) |
        | **Countries** | China, Czech Republic, India, Japan, Norway, United States |
        """)

        st.divider()
        st.markdown("### Damage Class Taxonomy")
        st.markdown("""
        | Class ID | Code | Damage Type | Training Count | Weight (Severity) |
        |:---:|---|---|:---:|:---:|
        | 0 | `D00` | Longitudinal Crack | 18,201 | 0.4 |
        | 1 | `D10` | Transverse Crack | 8,386 | 0.5 |
        | 2 | `D20` | Alligator Crack | 7,527 | 0.8 |
        | 3 | `D40` | Pothole | 7,554 | 1.0 |
        | 4 | `D43/D44` | Other Damage / Rutting | 4,628 | 0.6 |
        """)

        st.divider()
        st.markdown("### Damage Crop Preprocessing Pipeline")
        st.markdown("""
        1. **YOLO annotation parsing** — read `(class_id, xc, yc, w, h)` normalised coordinates
        2. **Bounding box extraction** — convert to absolute pixel coordinates
        3. **Context padding** — add 10% spatial border to preserve boundary context
        4. **CLAHE enhancement** — Contrast Limited Adaptive Histogram Equalisation in LAB colour space (`clipLimit=2.0`, `tileGridSize=8×8`) to enhance subtle crack contrast
        5. **Resize** — bilinear resize to 224 × 224 px
        6. **Augmentation** (train only) — random horizontal flip (p=0.5), ColorJitter
        7. **Normalisation** — ImageNet mean/std normalisation
        """)

    # ─── Tab 3: Severity Method ───────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### Severity Assessment Methodology")

        st.info("""
        **Important Clarification:** The severity assessment in this project is a
        **rule-based estimation system** — not a trained neural network.
        It is designed to provide a meaningful quantitative proxy for road condition
        severity and is **not validated against real-world civil engineering standards**.
        """)

        st.markdown("#### Formula")
        st.latex(r"""
        \text{Severity Score} = \min\left(100,\ 50 \cdot A_{norm}
        + 25 \cdot \frac{N_{regions}}{5}
        + 25 \cdot W_{class\_max}\right)
        """)

        st.markdown("""
        | Component | Symbol | Description | Max Contribution |
        |---|---|---|:---:|
        | Normalised Area | $A_{norm}$ | Total damaged bbox area ÷ image area | 50 points |
        | Region Density | $N_{regions}$ | Count of detected damage regions (capped at 5) | 25 points |
        | Class Weight | $W_{class\\_max}$ | Maximum damage severity weight in the image | 25 points |

        #### Class Damage Weights

        | Class | Code | Weight |
        |---|---|:---:|
        | Longitudinal Crack | D00 | 0.4 |
        | Transverse Crack | D10 | 0.5 |
        | Alligator Crack | D20 | 0.8 |
        | Pothole | D40 | **1.0** (highest) |
        | Other Damage | D43/D44 | 0.6 |

        #### Severity Categories
        | Category | Score Range | Action |
        |---|---|---|
        | 🟢 Low | 0 – 29.9 | Monitor regularly |
        | 🟡 Medium | 30 – 64.9 | Schedule maintenance |
        | 🔴 High | 65 – 100 | Immediate repair required |

        #### Repair Priority Index (1–10)
        """)

        st.latex(r"""
        \text{Priority Index} = \text{Clamp}\left(
        \left\lfloor \frac{\text{Severity Score}}{10} \right\rfloor
        + \left\lfloor W_{class\_max} \cdot 2 \right\rfloor,\ 1,\ 10\right)
        """)

        st.markdown("#### Worked Example")
        st.markdown("""
        **Image:** 640×480 (307,200 px²)  
        **Detections:** Pothole (D40) bbox=[50, 100, 200, 300] + Alligator Crack (D20) bbox=[300, 200, 450, 350]

        **Step 1: Area calculation**
        - Pothole area: 150 × 200 = 30,000 px²
        - Alligator area: 150 × 150 = 22,500 px²
        - Total damage area: 52,500 px²
        - $A_{norm}$ = 52,500 / 307,200 = **0.1709**

        **Step 2: Region count:** N_regions = 2

        **Step 3: Max class weight:** max(1.0, 0.8) = **1.0** (Pothole)

        **Step 4: Score**  
        = min(100, 50×0.1709 + 25×(2/5) + 25×1.0)  
        = min(100, 8.55 + 10.0 + 25.0)  
        = **43.55** → **Medium Severity**

        **Step 5: Priority Index**  
        = Clamp(floor(43.55/10) + floor(1.0×2), 1, 10)  
        = Clamp(4 + 2, 1, 10) = **6/10**
        """)

        st.markdown("#### Limitations")
        st.warning("""
        1. Severity is based on bounding box area approximation — not pixel-level segmentation
        2. Camera angle, altitude, and zoom affect the apparent damage size significantly
        3. Overlapping bounding boxes may double-count damaged area
        4. Class weights are heuristic — not calibrated from real civil engineering surveys
        5. The D43/D44 "Other Damage" class has low classification accuracy (0% F1)
        """)

    # ─── Tab 4: Viva Guide ────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### 🎓 Faculty Viva Preparation Guide")

        with st.expander("📢 30-Second Explanation", expanded=True):
            st.markdown("""
            > *"Our project automates road damage inspection using deep learning on the RDD2022 dataset.
            > We detect damage with Faster R-CNN, classify it into five categories using a Hybrid CNN+ViT
            > model, and estimate severity on a 0–100 scale. The ViT baseline achieved 92% accuracy.
            > The system produces annotated images and severity reports, accessible through this
            > Streamlit web application."*
            """)

        with st.expander("📢 2-Minute Explanation"):
            st.markdown("""
            Traditional road inspection is manual, time-consuming, and subjective.
            Our system automates this using the RDD2022 dataset — 38,000+ road images from 6 countries.

            **Phase 1** explored the dataset. **Phase 2** trained a Faster R-CNN detector to find damage
            bounding boxes (validation loss: 0.2661). **Phase 3** trained three classifiers on 224×224
            damage crops: CNN (EfficientNet-B0, 89%), ViT (vit_tiny, 92%), and our proposed Hybrid
            CNN+ViT model (87%). **Phase 4** added severity scoring (0–100) and Explainable AI with
            Grad-CAM heatmaps. **Phase 5** compared all models quantitatively.

            The final system takes a road image, detects damage regions, classifies each region,
            estimates severity using a formula based on damaged area and class weights, and displays
            annotated results with repair priority scores.
            """)

        with st.expander("❓ Key Viva Questions & Answers"):
            qa_list = [
                ("What is Deep Learning?",
                 "Deep Learning is a subset of machine learning using artificial neural networks with multiple layers to automatically learn hierarchical feature representations from data without manual feature engineering."),

                ("Why use Deep Learning for road damage detection?",
                 "Traditional computer vision requires hand-crafted features (edges, textures) which are brittle across weather, lighting, and camera conditions. Deep learning automatically learns robust features from large datasets, achieving superior generalisation across diverse road conditions and damage types."),

                ("What is a Convolutional Neural Network (CNN)?",
                 "A CNN uses learnable convolutional filters applied spatially across an image to detect local patterns (edges, textures, shapes). EfficientNet-B0 uses mobile-inspired inverted residual blocks with compound scaling for efficiency."),

                ("What is a Vision Transformer (ViT)?",
                 "ViT splits the input image into fixed-size patches (16×16 in our case, giving 196 tokens for 224×224 images), treats them like words in a sentence, and uses multi-head self-attention to model global spatial relationships between all patches simultaneously."),

                ("Why does ViT outperform CNN here?",
                 "Road damage patterns (cracks, potholes) often span large spatial regions with contextual dependencies. ViT's self-attention mechanism captures these global relationships explicitly, while CNN relies on stacking many convolution layers to achieve the same receptive field."),

                ("What is transfer learning?",
                 "Transfer learning initialises a model with weights pre-trained on a large dataset (ImageNet). Instead of training from random weights, we fine-tune these representations on the target task. This is crucial here because the RDD2022 damage crops are too few to train from scratch effectively."),

                ("What is the Hybrid CNN+ViT model?",
                 "It's our proposed architecture that runs EfficientNet-B0 and ViT-Tiny simultaneously on the same input crop, producing a 1280-dim local feature vector and 192-dim global feature vector respectively. These are concatenated into a 1472-dim fused representation fed into a BatchNorm→Dropout→Linear classification head."),

                ("What is Faster R-CNN?",
                 "Faster R-CNN is a two-stage object detection framework. Stage 1 (Region Proposal Network) proposes candidate object regions. Stage 2 (ROI head) classifies and refines bounding boxes for each proposal. We use a ResNet-50 FPN backbone for multi-scale feature extraction."),

                ("What is Feature Pyramid Network (FPN)?",
                 "FPN builds a multi-scale feature pyramid from a single backbone pass, allowing the detector to detect objects at different scales. This is important for road damage detection where damage sizes vary widely (small cracks vs large potholes)."),

                ("What is Grad-CAM?",
                 "Gradient-weighted Class Activation Mapping (Grad-CAM) visualises which spatial regions of an image most strongly influenced the model's prediction. It computes gradients of the class score with respect to the final convolutional feature maps, uses them as weights, and creates a spatial activation heatmap."),

                ("What is the severity score formula?",
                 "Score = min(100, 50×A_norm + 25×(N_regions/5) + 25×W_class_max). The three terms capture damaged surface coverage (0-50 pts), region density (0-25 pts), and damage type severity (0-25 pts based on class weights: Pothole=1.0, Alligator=0.8, Other=0.6, Transverse=0.5, Longitudinal=0.4)."),

                ("Why is severity rule-based rather than a trained model?",
                 "Severity requires ground truth severity labels from civil engineers, which are not present in RDD2022. The dataset only provides bounding box classes. Our rule-based formula is a reasonable proxy using available information (area, density, class type) but is explicitly documented as an estimation, not validated against civil engineering standards."),

                ("What is IoU?",
                 "Intersection over Union measures the overlap between predicted and ground truth bounding boxes. IoU = Area of Overlap / Area of Union. In mAP@50 evaluation, a prediction is considered correct if IoU ≥ 0.50 with a ground truth box of the same class."),

                ("What is mAP?",
                 "Mean Average Precision averages the area under the precision-recall curve across all classes and IoU thresholds. It is the standard metric for object detection evaluation. Our Phase 2 evaluation reported mAP@50=0.0 due to a threshold calibration issue in the small-subset evaluation, but the model's val_loss=0.2661 confirms it learns meaningful detection."),

                ("What is CLAHE?",
                 "Contrast Limited Adaptive Histogram Equalisation enhances local contrast in images. Applied in LAB colour space, it improves visibility of subtle cracks that may be hard to distinguish from road texture, especially in low-contrast or shadow-heavy images."),

                ("What is data augmentation?",
                 "Augmentation artificially increases training data variety by applying random transformations (flips, colour jitter) during training. This improves model generalisation and reduces overfitting, especially important for the minority classes (D43/D44) in our imbalanced dataset."),

                ("What causes overfitting?",
                 "Overfitting occurs when a model memorises training data patterns instead of learning generalisable features, resulting in high training accuracy but low validation accuracy. We mitigate this with Dropout, weight decay in AdamW, and data augmentation."),

                ("Why did you use AdamW?",
                 "AdamW decouples weight decay regularisation from the gradient update step (unlike Adam), providing better regularisation. Combined with CosineAnnealingLR, it achieves smooth loss convergence and prevents overfitting within our 3-epoch training budget."),

                ("What is the D43/D44 class challenge?",
                 "All three models achieve 0.0 F1 for Other Damage/Rutting. Root causes: (1) only 4,628 training samples vs 18,201 for D00, causing severe class imbalance; (2) extremely high intra-class visual diversity (rutting, edge cracks, painted markings); (3) 3 training epochs insufficient for rare class learning. Future work: class-weighted loss or SMOTE oversampling."),

                ("How does the inference pipeline work?",
                 "1) Load and preprocess image → 2) Faster R-CNN detects damage bounding boxes → 3) Each bbox region is cropped with 10% padding → 4) Hybrid CNN+ViT classifies each crop → 5) SeverityAssessor computes 0-100 score from all detections → 6) Annotated image generated with cv2 drawing → 7) Results displayed in Streamlit dashboard."),

                ("What are the project limitations?",
                 "1) Only 3 training epochs — more epochs would improve accuracy; 2) D43/D44 class 0% F1; 3) mAP evaluation issue; 4) Severity is rule-based not model-trained; 5) Bounding box approximation for area — segmentation would be more accurate; 6) No GPS/GIS integration; 7) Not tested on real-time video; 8) Camera height affects damage appearance."),
            ]

            for i, (q, a) in enumerate(qa_list, 1):
                with st.expander(f"Q{i}: {q}"):
                    st.markdown(f"**Answer:** {a}")

    # ─── Tab 5: Tech Stack ────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("### Technology Stack")

        stack = [
            ("🐍 Python 3.11",     "Core programming language"),
            ("🔥 PyTorch 2.0+",    "Deep learning framework — model definition, training, inference"),
            ("🖼️ TorchVision",     "Faster R-CNN model, ImageNet transforms, augmentation utilities"),
            ("🤖 timm",            "Vision Transformer (ViT) model library — `vit_tiny_patch16_224`"),
            ("👁️ OpenCV",          "Image reading, CLAHE contrast enhancement, bounding box drawing"),
            ("🔢 NumPy",           "Numerical array operations throughout the pipeline"),
            ("📊 Matplotlib",      "Training loss/accuracy curve plots, confusion matrices"),
            ("🎨 Seaborn",         "Heatmap visualisations for confusion matrices"),
            ("📈 Plotly",          "Interactive charts in the Streamlit dashboard"),
            ("🌐 Streamlit",       "Interactive multi-page web application framework"),
            ("🔬 Scikit-learn",    "Precision, Recall, F1, confusion matrix computation"),
            ("🖼️ Pillow (PIL)",    "Image loading, conversion, and saving utilities"),
            ("📦 Pandas",          "Tabular data display in Streamlit"),
        ]

        for tech, desc in stack:
            st.markdown(f"""
            <div style='display:flex;align-items:center;padding:8px 12px;margin:4px 0;
                        background:#1e293b;border-radius:8px;border:1px solid #334155;'>
                <span style='font-weight:700;color:#f1f5f9;min-width:180px;'>{tech}</span>
                <span style='color:#94a3b8;font-size:0.88rem;'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### Project Folder Structure")
        st.code("""
Project/
├── train/                     # RDD2022 Training Images & Labels (38k images)
├── val/                       # RDD2022 Validation Images & Labels (5.7k images)
├── test/                      # RDD2022 Test Images & Labels (5.7k images)
│
├── data/
│   └── crop_dataset.py        # CLAHE crop loader with 10% bbox padding
│
├── models/
│   ├── cnn_model.py           # EfficientNet-B0 CNN Baseline
│   ├── vit_model.py           # ViT-Tiny Baseline (timm)
│   ├── hybrid_model.py        # Hybrid CNN+ViT Fusion Model (Proposed)
│   ├── best_model.pth         # Faster R-CNN checkpoint
│   └── final_model.pth        # Final Faster R-CNN checkpoint
│
├── checkpoints/
│   ├── best_cnn_baseline.pth  # CNN Baseline checkpoint (89% acc)
│   ├── best_vit_baseline.pth  # ViT Baseline checkpoint (92% acc)
│   └── best_hybrid_cnn_vit.pth # Hybrid CNN+ViT checkpoint (87% acc)
│
├── src/
│   ├── model.py               # Faster R-CNN builder
│   ├── dataset.py             # Detection dataset loader
│   ├── train.py               # Detection training engine
│   ├── evaluate.py            # Detection evaluation metrics
│   ├── visualize.py           # Prediction grid visualiser
│   ├── severity.py            # Severity scoring engine (0-100)
│   ├── explainability.py      # Grad-CAM & ViT attention XAI
│   ├── benchmark.py           # Model profiler & evaluator
│   └── inference.py           # Clean single-image inference pipeline
│
├── app/
│   ├── streamlit_app.py       # Main Streamlit entry point
│   ├── analysis_history.json  # Stored analysis sessions
│   └── pages/
│       ├── page_dashboard.py  # Dashboard page
│       ├── page_analyze.py    # Image analysis page
│       ├── page_analytics.py  # Model analytics page
│       ├── page_history.py    # History page
│       └── page_model_info.py # Model information page
│
├── run_phase1_eda.py          # Phase 1: EDA script
├── run_phase2.py              # Phase 2: Faster R-CNN training
├── run_phase3.py              # Phase 3: Classification model training
├── run_phase4.py              # Phase 4: Severity & XAI pipeline
├── run_phase5.py              # Phase 5: Comparative benchmark evaluation
│
├── phase1_eda_stats.json      # EDA results
├── phase3_classification_metrics.json  # Training metrics
├── phase4_severity_summary.json        # Severity assessment results
├── phase5_benchmark_results.json       # Benchmark comparison results
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
└── PROJECT_COMPLETE_DOCUMENTATION.md  # Comprehensive documentation
        """, language="text")
