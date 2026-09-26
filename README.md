# Intelligent Road Damage Detection and Severity Assessment Using Deep Learning (RDD2022)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

**Intelligent Road Damage Detection and Severity Assessment Using Deep Learning** is an end-to-end computer vision and deep learning system designed for automated road condition inspection, damage classification, severity scoring, repair prioritization, and visual explainability using the **RDD2022** dataset.

The project implements and compares three distinct deep learning classification paradigms:
1. **CNN Baseline:** Fine-tuned `EfficientNet-B0` / `ResNet-50` backbone for local feature extraction.
2. **Vision Transformer Baseline:** Fine-tuned `ViT` (`timm`) for global contextual self-attention.
3. **Hybrid CNN + Vision Transformer (Main Proposed Model):** Feature fusion model concatenating CNN local representations and ViT global representations, followed by BatchNorm/Dropout regularization and a multi-class linear head.

The system is deployed via an interactive, multi-page **Streamlit Web Application** featuring upload analysis, bounding-box localization, automated severity metrics (Low/Medium/High), repair priority scores (1–10), and Explainable AI (Grad-CAM & ViT Attention maps).

---

## 📊 Dataset & Class Taxonomy

The system is built around the **RDD2022** road damage dataset (containing 38,000+ images across `train`, `val`, and `test` splits).

### Bounding Box Class Mapping:
- **Class 0 (`D00`):** Longitudinal Crack
- **Class 1 (`D10`):** Transverse Crack
- **Class 2 (`D20`):** Alligator Crack
- **Class 3 (`D40`):** Pothole
- **Class 4 (`D43/D44`):** Other Damage / Rutting

---

## 🏗️ System Architecture & Workflow

```
                        RDD2022 Raw Dataset
                                 │
                                 ▼
                     Phase 1: Dataset EDA & Audit
                     (Spatial Heatmaps, COCO Stats)
                                 │
                                 ▼
                Phase 2: Bounding Box Damage Crops &
                  Faster R-CNN Detection Baseline
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
          1. CNN Baseline  2. ViT Baseline  3. Hybrid CNN+ViT
           (EfficientNet)    (Transformer)    (Feature Fusion)
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                     Phase 4: Severity & XAI
                (0-100 Score, Grad-CAM, Attn Maps)
                                 │
                                 ▼
                  Phase 5: Comparative Evaluation
                  (Accuracy, F1, Confusion Matrix)
                                 │
                                 ▼
                Phase 6: Streamlit Web Dashboard
                 (Interactive 5-Page Application)
```

---

## 📐 Severity & Repair Priority Formulas

### **1. Project-Defined Severity Score (0 – 100):**
The severity score combines normalized damaged area ratio $A_{norm}$, damage region count $N_{regions}$, and maximum class damage weight $W_{class\_max}$:

$$A_{norm} = \frac{\sum (w \cdot h)_{bbox}}{\text{Image Area}}$$

$$\text{Severity Score} = \min\left(100, \, 50 \cdot A_{norm} + 25 \cdot \frac{N_{regions}}{5} + 25 \cdot W_{class\_max}\right)$$

- **Low Severity:** Score < 30
- **Medium Severity:** 30 $\le$ Score < 65
- **High Severity:** Score $\ge$ 65

### **2. Repair Priority Index (1 – 10):**
An experimental decision-support metric assisting civil maintenance planning:

$$\text{Priority Score} = \text{Clamp}\left(\left\lfloor \frac{\text{Severity Score}}{10} \right\rfloor + W_{class\_max}, \; 1, \; 10\right)$$

---

## 📁 Repository Structure

```
Project/
│
├── train/                        # RDD2022 Train Images & Labels (Git Ignored)
├── val/                          # RDD2022 Validation Images & Labels (Git Ignored)
├── test/                         # RDD2022 Test Images & Labels (Git Ignored)
│
├── src/                          # Baseline Object Detection Pipeline
│   ├── dataset.py                # PyTorch Dataset Loader with Augmentations
│   ├── model.py                  # Faster R-CNN ResNet-50 FPN Model Definition
│   ├── train.py                  # Loss Tracking & Checkpoint Engine
│   ├── evaluate.py               # mAP@50, Precision, Recall & PR Curve Metrics
│   └── visualize.py              # Qualitative Prediction Grid Visualizer
│
├── models/                       # Checkpoints Directory (Git Ignored)
│   ├── best_model.pth            # Optimal Faster R-CNN Model Checkpoint
│   └── final_model.pth           # Final Model Checkpoint
│
├── run_phase1_eda.py             # Phase 1 Exploratory Data Analysis Script
├── run_phase2.py                 # Phase 2 Baseline Pipeline Runner
│
├── requirements.txt              # Python Dependencies
├── .gitignore                    # Version Control Exclusions
└── README.md                     # Project Documentation
```

---

## 🚀 Quick Start Guide

### **1. Prerequisites & Environment Setup**
Clone the repository and set up a Python 3.11 virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### **2. Run Phase 1: Exploratory Data Analysis (EDA)**
Generate dataset statistics, COCO size distribution, spatial heatmaps, and ground truth sample visualizations:

```bash
python run_phase1_eda.py
```

---

### **3. Run Phase 2: Object Detection Pipeline & Baseline Training**
Execute the Faster R-CNN baseline training, validation loss tracking, quantitative evaluation, and qualitative prediction grid generation:

```bash
python run_phase2.py
```

---

### **4. Launch Interactive Streamlit Web Application**
Expose the complete system via the interactive 5-page dashboard:

```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 Model Comparison Research Question

This project addresses the core research question:

> *"Does combining CNN-based local feature extraction with Vision Transformer-based global feature extraction improve road damage classification accuracy and robustness compared to standalone CNN and ViT models?"*

| Model Architecture | Local Features | Global Self-Attention | Primary Feature Focus |
|---|:---:|:---:|---|
| **CNN Baseline (EfficientNet)** | High | Low | Textures, Edges, Crack Patterns |
| **ViT Baseline (Vision Transformer)** | Low | High | Global Road Layout, Context |
| **Hybrid CNN + ViT (Proposed)** | High | High | Combined Local Fine Details + Global Context |

---

## 📄 Citation & Acknowledgments

- **Dataset:** Global Road Damage Detection Challenge (RDD2022) Dataset.
- **Frameworks:** PyTorch, torchvision, `timm`, OpenCV, Streamlit, Matplotlib, Seaborn.
- **Author:** MCA Deep Learning Capstone Project.
