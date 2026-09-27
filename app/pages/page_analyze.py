"""
app/pages/page_analyze.py
==========================
Image Analysis page — upload, real-time inference, severity, download.
Streamlit 1.64.0 compatible (use_container_width, no use_column_width).
Python 3.11 compatible — no backslashes inside f-string expressions.
"""

import os
import sys
import io
import json
import datetime
import numpy as np
import streamlit as st
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.styles.theme import (
    COLORS, inject_css, page_header, section_title,
    severity_badge, empty_state,
)
from app.components.prediction_card import (
    severity_banner, condition_card, detection_row_html,
)

# ── Colour aliases — safe inside f-strings (no backslash) ────────────────────
C = COLORS   # use C['key'] in f-strings instead of COLORS["key"]

HISTORY_FILE = os.path.join(PROJECT_ROOT, "app", "analysis_history.json")


# ── History persistence ───────────────────────────────────────────────────────
def _append_history(record: dict):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.insert(0, record)
    history = history[:50]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def _load_inference_fn():
    try:
        from src.inference import run_inference
        return run_inference
    except Exception as exc:
        return exc


# ── Main render ───────────────────────────────────────────────────────────────
def render():
    inject_css()

    page_header(
        "Road Image Analysis",
        "Upload a road photograph — the AI detects damage, classifies it, and estimates severity.",
        "🔍",
    )
    st.divider()

    # ── Check models loaded ───────────────────────────────────────────────────
    inference_fn = _load_inference_fn()
    if isinstance(inference_fn, Exception):
        st.error(
            "Could not load the AI models. "
            "Ensure checkpoints/ and models/ contain trained weights."
        )
        with st.expander("Technical details"):
            st.code(str(inference_fn))
        return

    # ── Upload + settings layout ──────────────────────────────────────────────
    up_col, set_col = st.columns([1.3, 0.7], gap="large")

    with up_col:
        section_title("📤 Upload Road Image")
        uploaded = st.file_uploader(
            "upload",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="JPG, JPEG or PNG · Max 200 MB",
        )

        import glob
        sample_paths = (
            sorted(glob.glob(os.path.join(PROJECT_ROOT, "val",  "images", "*.jpg")))[:8] +
            sorted(glob.glob(os.path.join(PROJECT_ROOT, "test", "images", "*.jpg")))[:4]
        )
        sample_names = [os.path.basename(p) for p in sample_paths]

        if sample_names:
            muted = C['text_muted']
            st.markdown(
                f"<div style='color:{muted};font-size:0.83rem;"
                "font-weight:600;margin-top:10px;'>Or choose a dataset sample:</div>",
                unsafe_allow_html=True,
            )
            sel = st.selectbox(
                "sample_select",
                ["— Select a sample image —"] + sample_names,
                label_visibility="collapsed",
            )
            if sel != "— Select a sample image —":
                sel_path = next(
                    (p for p in sample_paths if os.path.basename(p) == sel), None
                )
                if sel_path:
                    uploaded = sel_path
        else:
            st.caption("No sample images found in val/images/ — upload your own.")

    with set_col:
        section_title("🎛️ Settings")
        device_choice = st.radio(
            "Compute device",
            ["CPU", "CUDA (GPU)"],
            horizontal=True,
            help="Falls back to CPU if CUDA is unavailable.",
        )
        device = "cuda" if device_choice == "CUDA (GPU)" else "cpu"

        st.selectbox(
            "Classification model",
            ["Hybrid CNN+ViT (Proposed)", "ViT Baseline (92%)", "CNN Baseline (89%)"],
            help="The Hybrid CNN+ViT checkpoint is used for inference.",
        )
        st.slider(
            "Detection confidence threshold",
            min_value=0.05, max_value=0.60, value=0.15, step=0.05,
            help="Lower = more detections (more false positives possible).",
        )

    # ── No image state ────────────────────────────────────────────────────────
    if uploaded is None:
        st.markdown(empty_state(
            "🛣️",
            "No road image selected yet",
            "Upload a JPG/PNG above or pick a sample image from the dataset.",
        ), unsafe_allow_html=True)
        return

    # ── Load and validate image ───────────────────────────────────────────────
    try:
        if isinstance(uploaded, str):
            pil_img  = Image.open(uploaded).convert("RGB")
            img_name = os.path.basename(uploaded)
        else:
            raw = uploaded.read()
            if len(raw) == 0:
                st.warning("The uploaded file appears to be empty. Please try again.")
                return
            pil_img  = Image.open(io.BytesIO(raw)).convert("RGB")
            img_name = uploaded.name
    except Exception as exc:
        st.error("Could not open this image. Please verify it is a valid JPG or PNG file.")
        with st.expander("Technical details"):
            st.code(str(exc))
        return

    # ── Image preview ─────────────────────────────────────────────────────────
    st.divider()
    prev_l, prev_r = st.columns([1.3, 0.7], gap="large")
    with prev_l:
        text_muted = C['text_muted']
        st.markdown(
            f"<div style='color:{text_muted};font-size:0.8rem;"
            "font-weight:600;margin-bottom:4px;'>PREVIEW</div>",
            unsafe_allow_html=True,
        )
        # ── FIXED: use_container_width (not use_column_width) ─────────────────
        st.image(pil_img, caption=img_name, use_container_width=True)
        w, h = pil_img.size
        text_light = C['text_light']
        st.markdown(
            f"<div style='color:{text_light};font-size:0.76rem;margin-top:3px;'>"
            f"{w} × {h} px</div>",
            unsafe_allow_html=True,
        )

    with prev_r:
        surf    = C['surface']
        border  = C['border']
        text_col = C['text']
        st.markdown(
            f"<div class='rd-card' style='margin-top:24px;'>"
            f"<div style='color:{text_muted};font-size:0.75rem;font-weight:600;"
            "text-transform:uppercase;'>File Info</div>"
            f"<div style='margin-top:10px;'>"
            f"<div style='color:{text_col};font-size:0.88rem;word-break:break-all;'>"
            f"<b>Name:</b> {img_name}</div>"
            f"<div style='color:{text_col};font-size:0.88rem;margin-top:4px;'>"
            f"<b>Size:</b> {w} × {h} px</div>"
            f"<div style='color:{text_col};font-size:0.88rem;margin-top:4px;'>"
            f"<b>Device:</b> {device.upper()}</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button(
            "🔍 Analyze Road Damage",
            type="primary",
            use_container_width=True,
        )

    if not analyze_btn:
        return

    # ── Run inference with staged progress ───────────────────────────────────
    st.divider()
    progress_box = st.empty()
    primary_col = C['primary']
    text_col2   = C['text']

    def _status(msg: str, done: bool = False):
        icon = "✅" if done else "⏳"
        progress_box.markdown(
            f"<div class='rd-card' style='border-left:3px solid {primary_col};'>"
            f"<span style='color:{text_col2};font-size:0.9rem;'>{icon} {msg}</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    _status("Preparing image…")
    try:
        _status("Running deep learning model…")
        result = inference_fn(
            image_input=uploaded if isinstance(uploaded, str) else pil_img,
            device=device,
        )
        _status("Analysis complete!", done=True)
    except Exception as exc:
        progress_box.empty()
        st.error("Analysis failed. Please try again with a different image.")
        with st.expander("Technical details"):
            st.code(str(exc))
        st.info("Ensure models/best_model.pth and checkpoints/best_hybrid_cnn_vit.pth exist.")
        return

    if not result.get("success"):
        progress_box.empty()
        err = result.get("error", "Unknown error")
        st.error(f"Inference error: {err}")
        return

    progress_box.empty()

    # ── Results layout ────────────────────────────────────────────────────────
    section_title("📋 Analysis Results")

    res_img_col, res_info_col = st.columns([1.3, 0.7], gap="large")

    annotated   = result.get("annotated_image")
    sv          = result.get("severity", {})
    detections  = result.get("detections", [])
    sev_cat     = sv.get("severity_category", "Low")
    sev_score   = sv.get("severity_score", 0.0)
    priority    = sv.get("repair_priority", 1)
    n_regions   = sv.get("region_count", 0)
    area_ratio  = sv.get("normalized_area_ratio", 0.0)
    overall     = result.get("overall_condition", "Unknown")

    with res_img_col:
        tab_ann, tab_orig = st.tabs(["🖼️ Detection Result", "📷 Original Image"])
        with tab_ann:
            if annotated is not None:
                # ── FIXED: use_container_width ────────────────────────────────
                st.image(
                    annotated,
                    caption="Detected damage regions with bounding boxes",
                    use_container_width=True,
                )
            else:
                st.image(pil_img, caption="No annotations available",
                         use_container_width=True)
        with tab_orig:
            # ── FIXED: use_container_width ────────────────────────────────────
            st.image(pil_img, caption="Original (unprocessed)",
                     use_container_width=True)

    with res_info_col:
        st.markdown(severity_banner(sev_cat, sev_score, priority),
                    unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Severity Score",  f"{sev_score:.1f}/100")
        m2.metric("Repair Priority", f"{priority}/10")
        m3, m4 = st.columns(2)
        m3.metric("Damage Regions",  str(n_regions))
        m4.metric("Affected Area",   f"{area_ratio*100:.2f}%")

        st.markdown(condition_card(overall), unsafe_allow_html=True)

        text_light2 = C['text_light']
        ms_val = result.get('inference_ms', 0)
        st.markdown(
            f"<div style='color:{text_light2};font-size:0.76rem;margin-top:6px;'>"
            f"⏱️ {ms_val:.0f} ms inference</div>",
            unsafe_allow_html=True,
        )

    # ── Detections list ───────────────────────────────────────────────────────
    st.divider()
    section_title("🏷️ Detected Damage Regions")

    if not detections:
        st.success("No damage detected — road surface appears clear.")
    else:
        for i, d in enumerate(detections, 1):
            st.markdown(
                detection_row_html(
                    i,
                    d.get("class_name", "?"),
                    d.get("cls_confidence", 0.0),
                    d.get("bbox", [0, 0, 0, 0]),
                ),
                unsafe_allow_html=True,
            )
        with st.expander("📄 Full detection data"):
            import pandas as pd
            rows = []
            for i, d in enumerate(detections, 1):
                x1, y1, x2, y2 = [int(v) for v in d.get("bbox", [0, 0, 0, 0])]
                rows.append({
                    "#":           i,
                    "Damage Type": d.get("class_name", "?"),
                    "Confidence":  f"{d.get('cls_confidence_pct', 0):.1f}%",
                    "Det. Score":  f"{d.get('det_confidence', 0)*100:.1f}%",
                    "BBox":        f"({x1},{y1})→({x2},{y2})",
                    "W×H (px)":    f"{x2-x1}×{y2-y1}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Severity breakdown ────────────────────────────────────────────────────
    with st.expander("📐 Severity Score Calculation"):
        area_pts   = 50 * area_ratio
        region_pts = 25 * min(n_regions, 5) / 5
        weight_pts = 25 * sv.get("max_class_weight", 0.0)
        st.markdown(
            f"| Component | Formula | Points |\n"
            "|---|---|---|\n"
            f"| Area | 50 × {area_ratio:.4f} | **{area_pts:.2f} / 50** |\n"
            f"| Regions | 25 × ({n_regions}/5) | **{region_pts:.2f} / 25** |\n"
            f"| Class Weight | 25 × {sv.get('max_class_weight',0):.2f} | **{weight_pts:.2f} / 25** |\n"
            f"| **Total** | min(100, {area_pts+region_pts+weight_pts:.2f}) | **{sev_score:.2f} / 100** |"
        )
        max_cls = sv.get('max_severity_class', 'N/A')
        st.caption(f"Max severity class: {max_cls}")

    # ── Download buttons ──────────────────────────────────────────────────────
    st.divider()
    dl1, dl2 = st.columns(2)

    if annotated is not None:
        try:
            ann_pil = Image.fromarray(annotated)
            buf = io.BytesIO()
            ann_pil.save(buf, format="JPEG", quality=92)
            dl1.download_button(
                "⬇️ Download Annotated Image",
                data=buf.getvalue(),
                file_name=f"road_analysis_{img_name}",
                mime="image/jpeg",
                use_container_width=True,
            )
        except Exception:
            pass

    json_report = {
        "image": img_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "detections": detections,
        "severity": {k: v for k, v in sv.items() if k != "details"},
        "overall_condition": overall,
        "inference_ms": result.get("inference_ms", 0),
    }
    base_name = img_name.rsplit(".", 1)[0]
    dl2.download_button(
        "⬇️ Download JSON Report",
        data=json.dumps(json_report, indent=2),
        file_name=f"road_report_{base_name}.json",
        mime="application/json",
        use_container_width=True,
    )

    # ── Save to history ───────────────────────────────────────────────────────
    _append_history({
        "timestamp":         datetime.datetime.now().isoformat(),
        "image_name":        img_name,
        "damage_detected":   result.get("damage_detected", False),
        "regions":           n_regions,
        "severity_score":    sev_score,
        "severity_category": sev_cat,
        "repair_priority":   priority,
        "overall_condition": overall,
        "inference_ms":      result.get("inference_ms", 0),
        "damage_types":      [d.get("class_name") for d in detections],
    })

    st.success("Analysis complete! Results saved to history.")
