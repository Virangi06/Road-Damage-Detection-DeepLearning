"""
api.py — Flask REST API for Road Damage AI
==========================================
FIXED VERSION v2:
- Models loaded at startup (not lazy) — avoids first-request timeout
- All numpy types converted to Python native before jsonify
- Detailed error logging to console
- Threaded Flask server with no timeout
- CORS configured for all origins
- /api/predict now surfaces whole-image fallback detections correctly
- /api/calibration endpoint exposes current threshold and score stats

Run:
    python api.py
    -> http://localhost:5000
"""

import os
import sys
import json
import base64
import datetime
import io
import traceback
import logging

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('road-damage-api')

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024   # 200 MB upload limit
app.config['JSON_SORT_KEYS'] = False
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── File paths ────────────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join(PROJECT_ROOT, "app", "analysis_history.json")
BENCH_FILE   = os.path.join(PROJECT_ROOT, "phase5_benchmark_results.json")
EDA_FILE     = os.path.join(PROJECT_ROOT, "phase1_eda_stats.json")
PHASE3_FILE  = os.path.join(PROJECT_ROOT, "phase3_classification_metrics.json")
PHASE2H_FILE = os.path.join(PROJECT_ROOT, "phase2_training_history.json")


# ── JSON helpers ──────────────────────────────────────────────────────────────
def _load_json(path, fallback=None):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not read {path}: {e}")
    return fallback if fallback is not None else {}


def _save_history(history: list):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _to_python(obj):
    """
    Recursively convert numpy / torch types to native Python types
    so Flask's jsonify never throws a TypeError.
    """
    import numpy as np
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    # torch tensors (if any leak through)
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
    except ImportError:
        pass
    return obj


# ── Model loading (at startup) ────────────────────────────────────────────────
_inference_fn    = None
_inference_error = None


def _load_models():
    """Called once at startup — loads Faster R-CNN + Hybrid CNN+ViT into RAM."""
    global _inference_fn, _inference_error
    log.info("Loading ML models (this may take 20-60 seconds on first run)…")
    try:
        from src.inference import run_inference, load_models
        # Pre-warm the model cache inside inference.py
        load_models(device='cpu')
        _inference_fn = run_inference
        log.info("ML models loaded successfully.")
    except Exception as exc:
        _inference_error = traceback.format_exc()
        log.error(f"Model loading failed:\n{_inference_error}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status":       "ok",
        "models_loaded": _inference_fn is not None,
        "model_error":  _inference_error,
        "timestamp":    datetime.datetime.now().isoformat(),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    log.info(f"POST /api/predict — files: {list(request.files.keys())}")

    # ── Validate file ─────────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"success": False,
                        "error": "No image file found in request. Send file with key 'image'."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in {"jpg", "jpeg", "png"}:
        return jsonify({"success": False,
                        "error": f"Unsupported file type '.{ext}'. Upload JPG or PNG."}), 400

    # ── Model availability ────────────────────────────────────────────────────
    if _inference_fn is None:
        msg = (
            "ML models are not loaded. "
            + ("Error: " + _inference_error[:300] if _inference_error else
               "Restart the API server to retry model loading.")
        )
        log.error(msg)
        return jsonify({"success": False, "error": msg}), 503

    # ── Load PIL image ────────────────────────────────────────────────────────
    try:
        raw     = file.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        log.info(f"Image loaded: {file.filename}  {pil_img.size}")
    except Exception as exc:
        log.error(f"Image open failed: {exc}")
        return jsonify({"success": False,
                        "error": f"Could not open image: {exc}"}), 400

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        log.info("Running inference…")
        result = _inference_fn(image_input=pil_img, device="cpu")
        log.info(f"Inference done — success={result.get('success')}, "
                 f"regions={len(result.get('detections', []))}, "
                 f"ms={result.get('inference_ms', 0):.0f}")
    except Exception as exc:
        tb = traceback.format_exc()
        log.error(f"Inference exception:\n{tb}")
        return jsonify({"success": False,
                        "error": f"Inference failed: {exc}",
                        "traceback": tb}), 500

    if not result.get("success"):
        err = result.get("error", "Unknown error from inference pipeline")
        log.error(f"Inference returned success=False: {err}")
        return jsonify({"success": False, "error": err}), 500

    # ── Build response ────────────────────────────────────────────────────────
    sv         = result.get("severity", {})
    detections = result.get("detections", [])

    # Annotated image → base64
    ann_b64 = result.get("annotated_b64")
    if not ann_b64 and result.get("annotated_image") is not None:
        try:
            import numpy as np
            ann_arr = result["annotated_image"]
            pil_ann = Image.fromarray(ann_arr.astype('uint8'))
            buf = io.BytesIO()
            pil_ann.save(buf, format="JPEG", quality=92)
            ann_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            log.warning(f"Could not encode annotated image: {e}")

    # Original image → base64
    try:
        orig_buf = io.BytesIO()
        pil_img.save(orig_buf, format="JPEG", quality=92)
        orig_b64 = base64.b64encode(orig_buf.getvalue()).decode("utf-8")
    except Exception as e:
        log.warning(f"Could not encode original image: {e}")
        orig_b64 = ""

    response_data = {
        "success":         True,
        "image_name":      file.filename,
        "image_size":      result.get("image_size", {}),
        "damage_detected": result.get("damage_detected", False),
        "detections":      detections,
        "severity":        {k: v for k, v in sv.items() if k != "details"},
        "overall_condition": result.get("overall_condition", ""),
        "inference_ms":    result.get("inference_ms", 0),
        "images": {
            "original": f"data:image/jpeg;base64,{orig_b64}" if orig_b64 else None,
            "annotated": f"data:image/jpeg;base64,{ann_b64}" if ann_b64 else None,
        },
    }

    # Convert all numpy types before jsonify
    safe_response = _to_python(response_data)

    # ── Append to history (non-fatal) ────────────────────────────────────────
    try:
        history = _load_json(HISTORY_FILE, [])
        history.insert(0, {
            "timestamp":         datetime.datetime.now().isoformat(),
            "image_name":        file.filename,
            "damage_detected":   bool(safe_response["damage_detected"]),
            "regions":           int(sv.get("region_count", 0)),
            "severity_score":    float(sv.get("severity_score", 0.0)),
            "severity_category": str(sv.get("severity_category", "Low")),
            "repair_priority":   int(sv.get("repair_priority", 1)),
            "overall_condition": str(safe_response["overall_condition"]),
            "inference_ms":      float(safe_response["inference_ms"]),
            "damage_types":      [str(d.get("class_name", "?")) for d in detections],
        })
        _save_history(history[:50])
        log.info("History saved.")
    except Exception as e:
        log.warning(f"History save failed (non-fatal): {e}")

    log.info("Returning prediction response.")
    return jsonify(safe_response)


@app.route("/api/history")
def get_history():
    data = _load_json(HISTORY_FILE, [])
    return jsonify({"success": True, "history": data, "count": len(data)})


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    try:
        _save_history([])
        return jsonify({"success": True, "message": "History cleared"})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/stats")
def stats():
    eda   = _load_json(EDA_FILE,   {})
    bench = _load_json(BENCH_FILE, {})
    hist  = _load_json(HISTORY_FILE, [])

    train = eda.get("train", {})
    val   = eda.get("val",   {})
    test  = eda.get("test",  {})

    total_images = (train.get("num_images", 26869) +
                    val.get("num_images",   5758)  +
                    test.get("num_images",  5758))
    total_boxes  = (train.get("total_boxes", 46296) +
                    val.get("total_boxes",   9741)  +
                    test.get("total_boxes",  9675))

    best_acc = 0.0
    best_f1  = 0.0
    for m in bench.values():
        acc = m.get("metrics", {}).get("accuracy", 0)
        f1  = m.get("metrics", {}).get("macro_f1", 0)
        if acc > best_acc: best_acc = acc
        if f1  > best_f1:  best_f1  = f1

    total_analyzed = len(hist)
    damaged_count  = sum(1 for h in hist if h.get("damage_detected"))
    high_sev       = sum(1 for h in hist if h.get("severity_category") == "High")
    avg_score      = (sum(h.get("severity_score", 0) for h in hist) /
                      max(total_analyzed, 1))

    class_counts = train.get("class_counts", {
        "D00 (Longitudinal Crack)": 18201,
        "D10 (Transverse Crack)":   8386,
        "D20 (Alligator Crack)":    7527,
        "D40 (Pothole)":            7554,
        "D43/D44 (Other Damage)":   4628,
    })

    return jsonify({
        "success": True,
        "dataset": {
            "total_images":      total_images,
            "total_annotations": total_boxes,
            "train_images":      train.get("num_images", 26869),
            "val_images":        val.get("num_images",   5758),
            "test_images":       test.get("num_images",  5758),
            "num_classes":       5,
            "class_counts":      class_counts,
            "country_counts":    train.get("country_counts", {}),
            "box_sizes":         train.get("box_sizes", {}),
        },
        "models": {
            "best_accuracy": best_acc,
            "best_macro_f1": round(best_f1, 4),
            "fastest_fps":   80.6,
            "best_model":    "ViT Baseline (vit_tiny)",
        },
        "history": {
            "total_analyzed":    total_analyzed,
            "damage_detected":   damaged_count,
            "high_severity":     high_sev,
            "avg_severity_score": round(avg_score, 2),
        },
    })


@app.route("/api/model-info")
def model_info():
    bench = _load_json(BENCH_FILE, {})

    training_history = {
        "CNN Baseline": {
            "epochs":    [1, 2, 3],
            "train_loss":[1.1091, 0.644, 0.4348],
            "val_loss":  [0.6137, 0.4139, 0.3785],
            "train_acc": [67.0, 78.33, 86.0],
            "val_acc":   [86.0, 88.0, 89.0],
        },
        "ViT Baseline": {
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

    return jsonify({
        "success": True,
        "benchmark": bench,
        "training_history": training_history,
        "detection_history": {
            "epochs":    [1, 2],
            "train_loss":[0.544, 0.3075],
            "val_loss":  [0.3123, 0.2661],
        },
        "class_weights": {
            "D00 (Longitudinal Crack)": 0.4,
            "D10 (Transverse Crack)":   0.5,
            "D20 (Alligator Crack)":    0.8,
            "D40 (Pothole)":            1.0,
            "D43/D44 (Other Damage)":   0.6,
        },
    })


@app.route("/api/eda-stats")
def eda_stats():
    data = _load_json(EDA_FILE, {})
    return jsonify({"success": True, "eda": data})


@app.route("/api/calibration")
def calibration():
    """Return current confidence threshold and any calibration results."""
    from src.inference import CONFIDENCE_THRESHOLD, FALLBACK_THRESHOLDS
    cal_data = _load_json(
        os.path.join(PROJECT_ROOT, "calibration_results.json"), {}
    )
    return jsonify({
        "success":              True,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "fallback_thresholds":  FALLBACK_THRESHOLDS,
        "calibration_data":     cal_data,
    })


# ── Startup + server ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Road Damage AI -- Flask API Server (v2 fixed)")
    print("  http://localhost:5000")
    print("=" * 60)
    print()

    # Load models BEFORE starting the HTTP server so the first
    # request doesn't time out waiting for PyTorch to warm up.
    _load_models()

    print()
    print("API ready. Starting HTTP server on port 5000...")
    print("Keep this window open while using the React frontend.")
    print()

    # threaded=True so concurrent requests don't block each other
    # use_reloader=False because the reloader breaks with heavy ML models
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,          # debug=True causes double model load
        threaded=True,
        use_reloader=False,
    )
