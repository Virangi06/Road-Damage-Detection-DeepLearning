"""
app/streamlit_app.py
====================
Main entry point — Road Damage AI (Pastel Professional Theme)
Streamlit 1.64.0 compatible.

Run:
    python -m streamlit run app/streamlit_app.py
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# ── Page config — MUST be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="Road Damage AI — RDD2022",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "🚧 Road Damage AI\n"
            "Intelligent Road Damage Detection & Severity Assessment\n"
            "MCA Deep Learning Capstone · RDD2022 Dataset · PyTorch"
        )
    },
)

# ── Inject global CSS (pastel theme) ─────────────────────────────────────────
from app.styles.theme import inject_css, COLORS
inject_css()

# ── Import page modules ───────────────────────────────────────────────────────
from app.pages import (
    page_dashboard,
    page_analyze,
    page_analytics,
    page_history,
    page_model_info,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown(
        f"""
        <div style='text-align:center;padding:18px 0 12px;'>
            <div style='font-size:2.4rem;'>🚧</div>
            <div style='color:{COLORS["text"]};font-size:1.25rem;font-weight:800;
                        letter-spacing:-0.02em;margin-top:4px;'>Road Damage AI</div>
            <div style='color:{COLORS["text_muted"]};font-size:0.78rem;font-weight:500;
                        margin-top:2px;'>RDD2022 Deep Learning System</div>
        </div>
        <hr style='border-color:{COLORS["border"]};margin:0 0 12px;'/>
        """,
        unsafe_allow_html=True,
    )

    # Navigation
    PAGE_MAP = {
        "🏠  Dashboard":     page_dashboard.render,
        "🔍  Analyze Image": page_analyze.render,
        "📊  Analytics":     page_analytics.render,
        "🕒  History":       page_history.render,
        "ℹ️   Model Info":    page_model_info.render,
    }

    selected = st.radio(
        "Navigation",
        list(PAGE_MAP.keys()),
        label_visibility="collapsed",
    )

    # Sidebar footer info
    st.markdown(
        f"""
        <hr style='border-color:{COLORS["border"]};margin:16px 0 12px;'/>
        <div style='font-size:0.74rem;color:{COLORS["text_muted"]};line-height:1.7;'>
            <b style='color:{COLORS["text"]}'>Stack</b><br>
            PyTorch · EfficientNet-B0<br>
            ViT-Tiny · Hybrid CNN+ViT<br>
            Faster R-CNN · Streamlit<br><br>
            <b style='color:{COLORS["text"]}'>Dataset</b><br>
            RDD2022 · 38,385 images<br>
            5 damage classes
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render selected page ──────────────────────────────────────────────────────
PAGE_MAP[selected]()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='rd-footer'>"
    "🚧 Road Damage AI &nbsp;·&nbsp; "
    "Intelligent Road Damage Detection & Severity Assessment &nbsp;·&nbsp; "
    "MCA Deep Learning Capstone &nbsp;·&nbsp; RDD2022 &nbsp;·&nbsp; PyTorch"
    "</div>",
    unsafe_allow_html=True,
)
