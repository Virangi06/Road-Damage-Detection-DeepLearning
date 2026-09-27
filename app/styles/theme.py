"""
app/styles/theme.py
====================
Centralized color palette, Plotly layout helpers, and CSS injection
for the Road Damage AI — Pastel Professional Theme.

Import this in every page:
    from app.styles.theme import COLORS, PLOTLY_COLORS, pastel_layout, inject_css
"""

# ── Pastel Color Palette ──────────────────────────────────────────────────────
# All hex values — safe for CSS and HTML
COLORS = {
    # Brand
    "primary":    "#A8DADC",   # soft teal
    "secondary":  "#BDE0FE",   # sky blue
    "accent":     "#CDB4DB",   # lavender
    # Semantic
    "success":    "#B7E4C7",   # mint green
    "warning":    "#FFE5B4",   # peach
    "danger":     "#FFB5A7",   # salmon
    # Surfaces
    "bg":         "#F8F9FC",   # near-white background
    "surface":    "#FFFFFF",   # card surface
    "border":     "#E2E8F0",   # subtle border
    "border_accent": "#A8DADC",
    # Text
    "text":       "#334155",   # dark slate
    "text_muted": "#64748B",   # muted slate
    "text_light": "#94A3B8",   # light slate
    # Severity
    "sev_low":    "#52B788",   # green
    "sev_medium": "#F4A261",   # orange
    "sev_high":   "#E63946",   # red
    # Chart accent solids (for Plotly — NOT rgba strings)
    "chart_1":    "#6EC6CA",   # teal (pastel primary darker)
    "chart_2":    "#7EB8F7",   # blue
    "chart_3":    "#C09BD8",   # lavender
    "chart_4":    "#F4A261",   # orange
    "chart_5":    "#74C69D",   # green
    "chart_6":    "#E9C46A",   # yellow
}

# ── Plotly-safe RGBA fill colors (with alpha, properly formatted) ─────────────
# These are derived from the chart colors with 0.18 alpha for fills
PLOTLY_FILL = {
    "chart_1": "rgba(110,198,202,0.18)",
    "chart_2": "rgba(126,184,247,0.18)",
    "chart_3": "rgba(192,155,216,0.18)",
    "chart_4": "rgba(244,162,97,0.18)",
    "chart_5": "rgba(116,198,157,0.18)",
    "chart_6": "rgba(233,196,106,0.18)",
}

# Ordered lists for convenience
CHART_COLORS  = [COLORS[f"chart_{i}"] for i in range(1, 7)]
CHART_FILLS   = [PLOTLY_FILL[f"chart_{i}"] for i in range(1, 7)]


# ── Plotly layout base — pastel / light theme ─────────────────────────────────
def pastel_layout(**overrides) -> dict:
    """
    Returns a dict of Plotly layout kwargs for the pastel theme.
    Pass keyword overrides to customise per-chart.

    Usage:
        fig.update_layout(**pastel_layout(title="My Chart", height=300))
    """
    base = dict(
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], size=12, family="Inter, sans-serif"),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_muted"]),
            title_font=dict(color=COLORS["text"]),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_muted"]),
            title_font=dict(color=COLORS["text"]),
        ),
        legend=dict(
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(color=COLORS["text"], size=11),
        ),
        margin=dict(l=20, r=20, t=45, b=20),
        title_font=dict(color=COLORS["text"], size=14, family="Inter, sans-serif"),
    )
    base.update(overrides)
    return base


# ── Master CSS injection ──────────────────────────────────────────────────────
_CSS = f"""
<style>
/* ── Google font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* ── Main background ── */
.stApp {{
    background-color: {COLORS['bg']};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FFFFFF 0%, #EEF6F7 100%);
    border-right: 1px solid {COLORS['border']};
}}
[data-testid="stSidebar"] .stRadio > label {{
    color: {COLORS['text']} !important;
    font-weight: 600;
    font-size: 0.92rem;
}}

/* ── Metric cards ── */
div[data-testid="metric-container"] {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-top: 3px solid {COLORS['primary']};
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(168,218,220,0.15);
}}
div[data-testid="metric-container"] label {{
    color: {COLORS['text_muted']} !important;
    font-size: 0.80rem !important;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
    color: {COLORS['text']} !important;
    font-size: 1.75rem !important;
    font-weight: 700;
}}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {{
    color: {COLORS['sev_low']} !important;
    font-size: 0.78rem !important;
}}

/* ── Dividers ── */
hr {{
    border-color: {COLORS['border']} !important;
    margin: 1.2rem 0 !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
    color: {COLORS['text']};
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.92rem;
    padding: 0.5rem 1.4rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 6px rgba(168,218,220,0.3);
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(168,218,220,0.4);
    background: linear-gradient(135deg, #93D0D3 0%, #A8D8FD 100%);
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%);
    color: {COLORS['text']};
    font-size: 1rem;
    padding: 0.6rem 1.6rem;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {COLORS['surface']};
    border-radius: 10px;
    border: 1px solid {COLORS['border']};
    padding: 4px;
    gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    color: {COLORS['text_muted']};
    font-weight: 500;
    padding: 6px 16px;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%) !important;
    color: {COLORS['text']} !important;
    font-weight: 600;
}}

/* ── Dataframe ── */
.stDataFrame {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid {COLORS['border']};
}}

/* ── Selectbox / radio / slider ── */
.stSelectbox > div > div {{
    background: {COLORS['surface']};
    border-color: {COLORS['border']};
    border-radius: 8px;
    color: {COLORS['text']};
}}
.stSlider > div > div > div > div {{
    background: {COLORS['primary']};
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: {COLORS['surface']};
    border: 2px dashed {COLORS['primary']};
    border-radius: 12px;
    padding: 8px;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {COLORS['accent']};
    background: #F0FBFC;
}}

/* ── Alerts ── */
.stSuccess {{
    background: {COLORS['success']};
    border-left-color: {COLORS['sev_low']};
    color: {COLORS['text']};
}}
.stWarning {{
    background: {COLORS['warning']};
    border-left-color: {COLORS['sev_medium']};
    color: {COLORS['text']};
}}
.stError {{
    background: {COLORS['danger']};
    border-left-color: {COLORS['sev_high']};
    color: {COLORS['text']};
}}

/* ── Spinner ── */
.stSpinner > div {{
    border-top-color: {COLORS['primary']} !important;
}}

/* ── Footer ── */
.rd-footer {{
    text-align: center;
    color: {COLORS['text_light']};
    font-size: 0.76rem;
    padding: 18px 0 8px;
    border-top: 1px solid {COLORS['border']};
    margin-top: 32px;
}}

/* ── Reusable card classes ── */
.rd-card {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 18px 20px;
    margin: 6px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}
.rd-card-accent {{
    border-left: 4px solid {COLORS['primary']};
}}
.rd-page-title {{
    color: {COLORS['text']};
    font-size: 1.9rem;
    font-weight: 800;
    margin-bottom: 2px;
    line-height: 1.2;
}}
.rd-page-subtitle {{
    color: {COLORS['text_muted']};
    font-size: 1rem;
    margin-top: 0;
    margin-bottom: 0;
}}
.rd-section-title {{
    color: {COLORS['text']};
    font-size: 1.1rem;
    font-weight: 700;
    margin: 18px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid {COLORS['primary']};
    display: inline-block;
}}
.rd-badge-low {{
    background: {COLORS['success']};
    color: #1B5E3B;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 700;
}}
.rd-badge-medium {{
    background: {COLORS['warning']};
    color: #7A4700;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 700;
}}
.rd-badge-high {{
    background: {COLORS['danger']};
    color: #7A1523;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 700;
}}
.rd-step-card {{
    text-align: center;
    padding: 16px 10px;
    background: {COLORS['surface']};
    border-radius: 12px;
    border: 1px solid {COLORS['border']};
    border-top: 3px solid {COLORS['accent']};
    height: 130px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}}
.rd-empty-state {{
    border: 2px dashed {COLORS['border']};
    border-radius: 14px;
    padding: 48px 24px;
    text-align: center;
    background: {COLORS['surface']};
    margin-top: 16px;
}}
</style>
"""


def inject_css():
    """Call once per page to inject the master stylesheet."""
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Convenience HTML builders ─────────────────────────────────────────────────

def page_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a styled page title + subtitle."""
    import streamlit as st
    icon_html = f"<span style='margin-right:8px;'>{icon}</span>" if icon else ""
    st.markdown(
        f"<h1 class='rd-page-title'>{icon_html}{title}</h1>"
        f"<p class='rd-page-subtitle'>{subtitle}</p>",
        unsafe_allow_html=True
    )


def section_title(text: str):
    """Render a styled section heading."""
    import streamlit as st
    st.markdown(f"<div class='rd-section-title'>{text}</div>", unsafe_allow_html=True)


def severity_badge(category: str) -> str:
    """Return HTML badge for a severity category."""
    cls_map = {"Low": "rd-badge-low", "Medium": "rd-badge-medium", "High": "rd-badge-high"}
    icon_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    css_cls = cls_map.get(category, "rd-badge-low")
    icon    = icon_map.get(category, "⚪")
    return f"<span class='{css_cls}'>{icon} {category}</span>"


def card(content_html: str, accent: bool = False) -> str:
    """Return an rd-card HTML block."""
    cls = "rd-card rd-card-accent" if accent else "rd-card"
    return f"<div class='{cls}'>{content_html}</div>"


def empty_state(icon: str, title: str, subtitle: str = "") -> str:
    """Return HTML for an empty-state placeholder."""
    sub = f"<p style='color:#94A3B8;font-size:0.88rem;margin-top:6px;'>{subtitle}</p>" if subtitle else ""
    return f"""
    <div class='rd-empty-state'>
        <div style='font-size:2.8rem;'>{icon}</div>
        <p style='color:#64748B;font-weight:600;font-size:1rem;margin:12px 0 4px;'>{title}</p>
        {sub}
    </div>"""
