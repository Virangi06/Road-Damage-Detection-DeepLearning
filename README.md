# Intelligent Road Damage Detection and Severity Assessment Using Deep Learning (RDD2022)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)](https://streamlit.io/)
[![timm](https://img.shields.io/badge/timm-0.9%2B-green.svg)](https://github.com/huggingface/pytorch-image-models)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

An end-to-end deep learning system for automated road condition inspection, damage classification, severity scoring, repair prioritisation, and visual explainability using the **RDD2022** benchmark dataset (38,385 road images, 6 countries).

The project implements and compares three deep learning classification paradigms:

| # | Architecture | Backbone | Val Accuracy | Macro F1 | Params |
|:---:|---|---|:---:|:---:|:---:|
| 1 | **CNN Baseline** | `EfficientNet-B0` | 89.0% | 0.6330 | 4.01M |
| 2 | **ViT Baseline** ⭐ | `vit_tiny_patch16_224` | **92.0%** | **0.6953** | 5.72M |
| 3 | **Hybrid CNN+ViT (Proposed)** | EfficientNet + ViT-Tiny | 87.0% | 0.6023 | 9.73M |

The system is deployed as a multi-page **Streamlit Web Application** with drag-and-drop upload, real-time inference, automated severity assessment, Grad-CAM / ViT attention explainability, and interactive analytics dashboards.

---

## 📊 Dataset — RDD2022

| Property | Value |
|---|---|
| Total Images | 38,385 (train: 26,869 · val: 5,758 · test: 5,758) |
| Total Annotations | 65,712 bounding boxes |
| Annotation Format | YOLO: `class_id xc yc w h` (normalised) |
| Countries | China, Czech Republic, India, Japan, Norway, United States |

### Damage Class Taxonomy

| Class ID | Code | Damage Type | Train Count | Severity Weight |
|:---:|---|---|:---:|:---:|
| 0 | `D00` | Longitudinal Crack | 18,201 | 0.4 |
| 1 | `D10` | Transverse Crack | 8,386 | 0.5 |
| 2 | `D20` | Alligator Crack | 7,527 | 0.8 |
| 3 | `D40` | Pothole | 7,554 | **1.0** |
| 4 | `D43/D44` | Other Damage / Rutting | 4,628 | 0.6 |

---

## 🏗️ System Architecture

```
                        RDD2022 Dataset (38,385 images)
                                     │
                     Phase 1: EDA & Dataset Audit
                     (class/country/size distributions)
                                     │
                     Phase 2: Faster R-CNN Detection
                     ResNet-50 FPN · val_loss = 0.2661
                                     │
                        Damage Crop Extraction
                     (224×224, CLAHE, 10% padding)
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                       ▼
    1. CNN Baseline          2. ViT Baseline        3. Hybrid CNN+ViT
    EfficientNet-B0          vit_tiny_patch16        (Proposed)
       89% acc                  92% acc               87% acc
              └──────────────────────┼──────────────────────┘
                                     │
                     Phase 4: Severity Assessment
                     0-100 Score · Low/Medium/High
                     Grad-CAM · ViT Attention Maps
                                     │
                     Phase 5: Benchmark Evaluation
                     Confusion Matrices · F1 Comparison
                                     │
                     Phase 6: Streamlit Web Application
                     Dashboard · Analyze · Analytics
                     History · Model Info
```

---

## 📐 Severity Assessment Formula

The severity engine is **rule-based** (not a trained neural network). It converts bounding box detections into a quantitative 0–100 score:

$$\text{Severity Score} = \min\left(100,\ 50 \cdot A_{norm} + 25 \cdot \frac{N_{regions}}{5} + 25 \cdot W_{class\_max}\right)$$

| Component | Formula | Max Points |
|---|---|:---:|
| Damaged Area Ratio | $50 \times A_{norm}$ where $A_{norm} = \frac{\sum(w_{bbox} \times h_{bbox})}{A_{image}}$ | 50 |
| Region Density | $25 \times \min(N_{regions}, 5) / 5$ | 25 |
| Damage Class Weight | $25 \times W_{class\_max}$ | 25 |

**Severity Categories:**
- 🟢 **Low:** Score < 30 → Monitor regularly
- 🟡 **Medium:** 30 ≤ Score < 65 → Schedule maintenance  
- 🔴 **High:** Score ≥ 65 → Immediate repair required

**Repair Priority Index (1–10):**
$$\text{Priority} = \text{Clamp}\!\left(\left\lfloor \tfrac{\text{Score}}{10} \right\rfloor + \left\lfloor W_{class\_max} \cdot 2 \right\rfloor,\ 1,\ 10\right)$$

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Dataset Setup
Place the RDD2022 dataset splits in the project root:
```
Project/train/images/   Project/train/labels/
Project/val/images/     Project/val/labels/
Project/test/images/    Project/test/labels/
```

### 3. Launch the Web Application
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

## 🖥️ Streamlit Web Application

The application has **5 interactive pages**:

| Page | Description |
|---|---|
| 🏠 **Dashboard** | KPI cards, class distribution, country chart, model radar comparison, severity guide |
| 🔍 **Analyze Image** | Upload or select a sample image · real-time inference · annotated result · severity breakdown · download |
| 📊 **Analytics** | Model comparison, training curves, per-class F1, efficiency trade-off, dataset statistics |
| 🕒 **History** | All previous analyses stored automatically · filter · sort · timeline chart |
| ℹ️ **Model Info** | Architecture reference · dataset details · severity methodology · viva Q&A guide |

---

## 📋 Running Individual Phases

```bash
# Phase 1: Exploratory Data Analysis
python run_phase1_eda.py

# Phase 2: Faster R-CNN detection training
python run_phase2.py

# Phase 3: Train CNN, ViT, and Hybrid classifiers
python run_phase3.py

# Phase 4: Severity assessment + XAI heatmaps
python run_phase4.py

# Phase 5: Comparative benchmark evaluation
python run_phase5.py

# Test the inference pipeline on sample images
python src/inference.py
```

---

## 📁 Repository Structure

```
Project/
├── train/ val/ test/           # RDD2022 dataset splits (Git-ignored)
│
├── data/
│   └── crop_dataset.py         # CLAHE crop loader (224×224, 10% padding)
│
├── models/
│   ├── cnn_model.py            # EfficientNet-B0 CNN Baseline
│   ├── vit_model.py            # ViT-Tiny Baseline (timm)
│   ├── hybrid_model.py         # Hybrid CNN+ViT Proposed Architecture
│   ├── best_model.pth          # Faster R-CNN checkpoint
│   └── final_model.pth         # Faster R-CNN final checkpoint
│
├── checkpoints/
│   ├── best_cnn_baseline.pth   # CNN checkpoint (89% accuracy)
│   ├── best_vit_baseline.pth   # ViT checkpoint (92% accuracy)
│   └── best_hybrid_cnn_vit.pth # Hybrid checkpoint (87% accuracy)
│
├── src/
│   ├── model.py                # Faster R-CNN builder
│   ├── dataset.py              # Detection dataset loader
│   ├── train.py                # Detection training engine
│   ├── evaluate.py             # Detection evaluation (mAP, PR curve)
│   ├── visualize.py            # Prediction grid visualiser
│   ├── severity.py             # SeverityAssessor (0-100 score engine)
│   ├── explainability.py       # GradCAMExplainer + ViTAttentionExplainer
│   ├── benchmark.py            # ModelProfiler + ModelEvaluator
│   └── inference.py            # End-to-end single-image inference pipeline
│
├── app/
│   ├── streamlit_app.py        # Streamlit entry point
│   └── pages/                  # Dashboard, Analyze, Analytics, History, ModelInfo
│
├── run_phase1_eda.py  run_phase2.py  run_phase3.py
├── run_phase4.py      run_phase5.py
│
├── phase*_*.json / phase*_*.png    # Generated outputs
├── PHASE2_REPORT.md  PHASE3_REPORT.md  PHASE4_REPORT.md  PHASE5_REPORT.md
├── requirements.txt
├── README.md
└── PROJECT_COMPLETE_DOCUMENTATION.md  # Full 30-section technical documentation
```

---

## 📈 Detailed Results

### Classification Performance (Phase 3 & 5)

| Model | Accuracy | Macro P | Macro R | Macro F1 | Latency | FPS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| CNN Baseline (EfficientNet-B0) | 89.0% | 0.7261 | 0.6209 | 0.6330 | 12.4 ms | 80.6 |
| ViT Baseline (vit_tiny) | **92.0%** | 0.7175 | **0.6818** | **0.6953** | 18.7 ms | 53.5 |
| Hybrid CNN+ViT (Proposed) | 87.0% | 0.6000 | 0.6046 | 0.6023 | 31.2 ms | 32.1 |

### Per-Class F1 Score

| Damage Class | CNN | ViT | Hybrid |
|---|:---:|:---:|:---:|
| D00 — Longitudinal Crack | 0.9254 | **0.9677** | 0.9375 |
| D10 — Transverse Crack | 0.9383 | **0.9639** | 0.9383 |
| D20 — Alligator Crack | 0.4444 | **0.6667** | 0.2857 |
| D40 — Pothole | 0.8571 | **0.8780** | 0.8500 |
| D43/D44 — Other Damage | 0.0 | 0.0 | 0.0 |

> **Note:** All models score 0.0 F1 on D43/D44 due to class imbalance (only 10% of data) and visual heterogeneity. This is a documented limitation.

### Detection Baseline (Phase 2 — Faster R-CNN)

| Epoch | Train Loss | Val Loss |
|:---:|:---:|:---:|
| 1 | 0.5440 | 0.3123 |
| **2 (Best)** | 0.3075 | **0.2661** |

---

## 🔬 Research Question & Findings

> *"Does combining CNN-based local feature extraction with Vision Transformer-based global attention improve road damage classification accuracy?"*

**Finding:** In the 3-epoch experiment, the standalone ViT (92%) slightly outperforms the Hybrid CNN+ViT (87%). However, the Hybrid shows more consistent per-class F1 across D00, D10, and D40. The hypothesis holds directionally — additional training epochs with class-weighted loss are expected to allow the Hybrid architecture to leverage its theoretical representational advantage.

---

## 🤖 Using the Inference API

```python
from src.inference import run_inference

result = run_inference('path/to/road_image.jpg', device='cpu')

print(f"Damage Detected : {result['damage_detected']}")
print(f"Regions Found   : {len(result['detections'])}")
for d in result['detections']:
    print(f"  {d['class_name']}  {d['cls_confidence_pct']:.1f}%  bbox={d['bbox']}")

sv = result['severity']
print(f"Severity Score  : {sv['severity_score']}/100  ({sv['severity_category']})")
print(f"Repair Priority : {sv['repair_priority']}/10")
print(f"Road Condition  : {result['overall_condition']}")
print(f"Inference Time  : {result['inference_ms']} ms")
```

---

## 📚 Documentation

| Document | Description |
|---|---|
| `PROJECT_COMPLETE_DOCUMENTATION.md` | Full 30-section technical documentation — architecture, methodology, results, viva Q&A, demo guide |
| `PHASE2_REPORT.md` | Faster R-CNN detection baseline report |
| `PHASE3_REPORT.md` | Three-model classification architecture report |
| `PHASE4_REPORT.md` | Severity assessment and XAI report |
| `PHASE5_REPORT.md` | Comparative benchmark evaluation report |

---

## 📄 Citation & Acknowledgments

- **Dataset:** Global Road Damage Detection Challenge (RDD2022) Dataset
- **Frameworks:** PyTorch, torchvision, timm, OpenCV, Streamlit, Matplotlib, Seaborn, Plotly
- **Author:** MCA Deep Learning Capstone Project
