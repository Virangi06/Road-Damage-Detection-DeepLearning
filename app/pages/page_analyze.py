"""
app/pages/page_analyze.py
==========================
Image Analysis page — drag-and-drop upload, real-time inference,
bounding-box visualisation, severity score, and downloadable results.
"""

import os
import sys
import io
import json
import datetime
import base64
import numpy as np
import streamlit as st
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Severity colour helpers ───────────────────────────────────────────────────

_SEV_COLOR = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
_SEV_ICON  = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
_COND_ICON = {
    "Good":     "✅",
    "Moderate": "⚠️",
    "Fair":     "🔶",
    "Poor":     "🚨",
}


def _sev_html(category: str) -> str:
    color = _SEV_COLOR.get(category, "#94a3b8")
    icon  = _SEV_ICON.get(category,  "⚪")
    return (f"<span style='color:{color};font-size:1.4rem;font-weight:800;'>"
            f"{icon} {category.upper()} SEVERITY</span>")


def _overall_icon(label: str) -> str:
    for key, icon in _COND_ICON.items():
        if key.lower() in label.lower():
            return icon
    return "ℹ️"


# ── History helper ────────────────────────────────────────────────────────────

HISTORY_FILE = os.path.join(PROJECT_ROOT, 'app', 'analysis_history.json')


def _append_history(record: dict):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                history = json.load(f)
            except Exception:
                history = []
    history.insert(0, record)
    history = history[:50]   # keep last 50
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    st.markdown("""
    <h2 style='background:linear-gradient(90deg,#0ea5e9,#6366f1);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               font-size:1.9rem;font-weight:800;'>
        🔍 Road Image Analysis
    </h2>
    <p style='color:#94a3b8;'>
        Upload a road photograph and the AI system will detect damage,
        classify its type, and estimate severity in seconds.
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Upload area ───────────────────────────────────────────────────────────
    st.markdown("##### 📤 Upload Road Image")
    uploaded = st.file_uploader(
        "Drag & drop or click to browse — JPG, JPEG, PNG supported",
        type=["jpg", "jpeg", "png"],
        help="Upload a road photograph. The model works best on drone or camera-level road surface images.",
        label_visibility="collapsed",
    )

    # ── Sample images quick-select ────────────────────────────────────────────
    st.markdown("**Or use a sample image from the dataset:**")
    import glob
    sample_paths = (
        sorted(glob.glob(os.path.join(PROJECT_ROOT, 'val',  'images', '*.jpg')))[:6] +
        sorted(glob.glob(os.path.join(PROJECT_ROOT, 'test', 'images', '*.jpg')))[:6]
    )
    if sample_paths:
        sample_names = [os.path.basename(p) for p in sample_paths[:6]]
        selected_sample = st.selectbox(
            "Sample images",
            ["— Select sample image —"] + sample_names,
            label_visibility="collapsed",
        )
        if selected_sample != "— Select sample image —":
            sel_path = next((p for p in sample_paths if os.path.basename(p) == selected_sample), None)
            if sel_path:
                uploaded = sel_path   # treat path as "uploaded"
    else:
        st.caption("No sample images found — upload your own above.")

    if uploaded is None:
        # Landing state
        st.markdown("""
        <div style='border:2px dashed #334155;border-radius:14px;padding:40px;
                    text-align:center;background:#0f172a;margin-top:16px;'>
            <div style='font-size:3rem;'>🛣️</div>
            <div style='color:#64748b;font-size:1rem;margin-top:12px;'>
                Upload a road image to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Load image ────────────────────────────────────────────────────────────
    try:
        if isinstance(uploaded, str):
            pil_img  = Image.open(uploaded).convert('RGB')
            img_name = os.path.basename(uploaded)
        else:
            pil_img  = Image.open(uploaded).convert('RGB')
            img_name = uploaded.name
    except Exception as e:
        st.error(f"Could not open image: {e}")
        return

    col_upload, col_analyze = st.columns([1.2, 0.8], gap="large")

    with col_upload:
        st.markdown("**📸 Uploaded Image**")
        st.image(pil_img, caption=img_name, use_column_width=True)

    with col_analyze:
        st.markdown("**🎛️ Analysis Settings**")
        device_choice = st.radio("Compute device", ["CPU", "CUDA (GPU)"],
                                 horizontal=True,
                                 help="GPU significantly speeds up inference. Falls back to CPU if unavailable.")
        device = 'cuda' if device_choice == "CUDA (GPU)" else 'cpu'

        model_choice = st.selectbox(
            "Classification model",
            ["Hybrid CNN+ViT (Proposed)", "ViT Baseline (92% acc)", "CNN Baseline (89% acc)"],
            help="The Hybrid model is the primary proposed architecture. ViT achieved the highest standalone accuracy.",
        )

        conf_threshold = st.slider(
            "Detection confidence threshold",
            min_value=0.05, max_value=0.60, value=0.15, step=0.05,
            help="Lower values detect more regions (may include false positives)."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Analyze Image", type="primary", use_container_width=True)

    if not analyze_btn:
        return

    # ── Run inference ─────────────────────────────────────────────────────────
    with st.spinner("🔄 Running deep learning inference..."):
        try:
            from src.inference import run_inference
            result = run_inference(
                image_input=uploaded if isinstance(uploaded, str) else pil_img,
                device=device,
            )
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.info("Make sure model checkpoints exist in `checkpoints/` and `models/` directories.")
            return

    if not result["success"]:
        st.error(f"Inference error: {result['error']}")
        return

    # ── Results display ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("## 📋 Analysis Results")

    # Annotated image + metrics side by side
    r_left, r_right = st.columns([1.3, 0.7], gap="large")

    with r_left:
        st.markdown("**🖼️ Annotated Road Image**")
        annotated = result.get("annotated_image")
        if annotated is not None:
            st.image(annotated, caption="Detected damage regions with bounding boxes",
                     use_column_width=True)
        else:
            st.image(pil_img, caption="No annotated image available", use_column_width=True)

        # Toggle original/annotated
        if st.checkbox("🔄 Show original image", value=False):
            st.image(pil_img, caption="Original (unprocessed)", use_column_width=True)

    with r_right:
        sv = result.get("severity", {})
        detections = result.get("detections", [])

        # Severity banner
        sev_cat   = sv.get("severity_category", "Low")
        sev_score = sv.get("severity_score", 0.0)
        priority  = sv.get("repair_priority", 1)
        n_regions = sv.get("region_count", 0)
        area_ratio = sv.get("normalized_area_ratio", 0.0)
        overall   = result.get("overall_condition", "Unknown")

        st.markdown(_sev_html(sev_cat), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # KPI metrics
        m1, m2 = st.columns(2)
        m1.metric("Severity Score",   f"{sev_score:.1f} / 100")
        m2.metric("Repair Priority",  f"{priority} / 10")
        m3, m4 = st.columns(2)
        m3.metric("Damage Regions",   str(n_regions))
        m4.metric("Affected Area",    f"{area_ratio*100:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**{_overall_icon(overall)} Overall Condition**")
        st.markdown(f"<div style='background:#1e293b;border-radius:8px;padding:12px;"
                    f"color:#f1f5f9;font-size:1rem;font-weight:600;'>{overall}</div>",
                    unsafe_allow_html=True)

        st.markdown(f"<div style='margin-top:10px;color:#64748b;font-size:0.8rem;'>"
                    f"⏱️ Inference time: {result.get('inference_ms',0):.1f} ms</div>",
                    unsafe_allow_html=True)

    # ── Per-detection table ───────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🏷️ Detected Damage Regions")

    if not detections:
        st.success("✅ No damage detected — road surface appears clear.")
    else:
        import pandas as pd
        rows = []
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            rows.append({
                "#":              i + 1,
                "Damage Type":    d.get("class_name", "Unknown"),
                "Confidence":     f"{d.get('cls_confidence_pct', 0):.1f}%",
                "Det. Score":     f"{d.get('det_confidence', 0)*100:.1f}%",
                "BBox (x1,y1)":   f"({x1}, {y1})",
                "BBox (x2,y2)":   f"({x2}, {y2})",
                "Box Width":      f"{x2-x1} px",
                "Box Height":     f"{y2-y1} px",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

    # ── Severity breakdown ────────────────────────────────────────────────────
    st.markdown("#### 📐 Severity Score Breakdown")

    sv_details = [
        ("Area Component",    f"50 × A_norm = 50 × {area_ratio:.4f} = {50*area_ratio:.2f}",
         f"{50*area_ratio:.2f} / 50"),
        ("Region Component",  f"25 × (N / 5) = 25 × ({n_regions}/5) = {25*min(n_regions,5)/5:.2f}",
         f"{25*min(n_regions,5)/5:.2f} / 25"),
        ("Class Weight",      f"25 × W_max = 25 × {sv.get('max_class_weight',0):.2f} = {25*sv.get('max_class_weight',0):.2f}",
         f"{25*sv.get('max_class_weight',0):.2f} / 25"),
        ("**Total Score**",   f"min(100, sum) = **{sev_score:.2f}**",
         f"**{sev_score:.2f} / 100**"),
    ]
    for label, formula, value in sv_details:
        c1_, c2_, c3_ = st.columns([1.2, 2.5, 1])
        c1_.markdown(label)
        c2_.markdown(f"`{formula}`")
        c3_.markdown(value)

    # ── Download result ───────────────────────────────────────────────────────
    st.divider()
    dl1, dl2 = st.columns(2)

    # Download annotated image
    if result.get("annotated_image") is not None:
        ann_pil = Image.fromarray(result["annotated_image"])
        buf = io.BytesIO()
        ann_pil.save(buf, format="JPEG", quality=92)
        dl1.download_button(
            "⬇️ Download Annotated Image",
            data=buf.getvalue(),
            file_name=f"road_analysis_{img_name}",
            mime="image/jpeg",
            use_container_width=True,
        )

    # Download JSON report
    json_report = {
        "image": img_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "detections": result.get("detections", []),
        "severity": {k: v for k, v in sv.items() if k != "details"},
        "overall_condition": overall,
        "inference_ms": result.get("inference_ms", 0),
    }
    dl2.download_button(
        "⬇️ Download JSON Report",
        data=json.dumps(json_report, indent=2),
        file_name=f"road_report_{img_name.split('.')[0]}.json",
        mime="application/json",
        use_container_width=True,
    )

    # ── Save to history ───────────────────────────────────────────────────────
    history_record = {
        "timestamp":  datetime.datetime.now().isoformat(),
        "image_name": img_name,
        "damage_detected": result["damage_detected"],
        "regions":    n_regions,
        "severity_score":    sev_score,
        "severity_category": sev_cat,
        "repair_priority":   priority,
        "overall_condition": overall,
        "inference_ms":      result.get("inference_ms", 0),
        "damage_types": [d.get("class_name") for d in detections],
    }
    _append_history(history_record)
    st.success("✅ Analysis complete! Result saved to history.")
