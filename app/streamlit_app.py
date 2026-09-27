"""
app/streamlit_app.py
====================
Main entry point for the Road Damage Detection & Severity Assessment
Interactive Streamlit Web Application.

Run with:
    streamlit run app/streamlit_app.py

Pages:
    1. 🏠 Dashboard       — project stats & overview
    2. 🔍 Analyze Image   — upload & analyse road images
    3. 📊 Analytics       — model performance & damage distribution
    4. 🕒 History         — previous analysis sessions
    5. ℹ️  Model Info      — architecture details & viva guide
"""

import os
import sys

# Add project root to path so src/ and models/ are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Road Damage AI — RDD2022",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Intelligent Road Damage Detection & Severity Assessment using Deep Learning (RDD2022) — MCA Capstone Project"
    }
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar navigation */
    .css-1d391kg { background-color: #1a1a2e; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.9rem !important;
        font-weight: 700;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Info cards */
    .info-card {
        background: #1e293b;
        border-left: 4px solid #0ea5e9;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .severity-low    { color: #22c55e; font-weight: 700; }
    .severity-medium { color: #f59e0b; font-weight: 700; }
    .severity-high   { color: #ef4444; font-weight: 700; }

    /* Upload area */
    .uploadedFile { border-radius: 10px; }

    /* Footer */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.78rem;
        padding: 20px 0;
        border-top: 1px solid #1e293b;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ── Import pages ──────────────────────────────────────────────────────────────
from app.pages import (
    page_dashboard,
    page_analyze,
    page_analytics,
    page_history,
    page_model_info,
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/road.png",
    width=72,
)
st.sidebar.markdown("## 🛣️ Road Damage AI")
st.sidebar.markdown("*RDD2022 Deep Learning System*")
st.sidebar.markdown("---")

PAGE_MAP = {
    "🏠  Dashboard":      page_dashboard.render,
    "🔍  Analyze Image":  page_analyze.render,
    "📊  Analytics":      page_analytics.render,
    "🕒  History":        page_history.render,
    "ℹ️   Model Info":     page_model_info.render,
}

selected_page = st.sidebar.radio(
    "Navigate",
    list(PAGE_MAP.keys()),
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.75rem; color:#64748b;'>
<b>Tech Stack</b><br>
PyTorch · EfficientNet-B0<br>
ViT-Tiny · Hybrid CNN+ViT<br>
Faster R-CNN · Streamlit<br><br>
<b>Dataset:</b> RDD2022<br>
38,385 road images · 5 classes
</div>
""", unsafe_allow_html=True)

# ── Render selected page ──────────────────────────────────────────────────────
PAGE_MAP[selected_page]()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    Intelligent Road Damage Detection & Severity Assessment — MCA Deep Learning Capstone Project
    · RDD2022 Dataset · PyTorch · Streamlit
</div>
""", unsafe_allow_html=True)
