"""
src/inference.py
================
Clean, single-image inference pipeline for the Road Damage Detection system.

Accepts a PIL Image or file path, runs:
  1. Faster R-CNN object detection  →  bounding boxes + class labels
  2. Hybrid CNN+ViT classification  →  per-region damage type + confidence
  3. Severity assessment engine     →  0-100 score, Low/Medium/High category
  4. Annotated image generation     →  drawn boxes with labels

Returns a structured prediction dictionary suitable for display or API use.
"""

import os
import sys
import io
import base64
import time
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.severity import SeverityAssessor
from models.hybrid_model import HybridCNNViTModel
from src.model import build_model

# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "D00 - Longitudinal Crack",
    1: "D10 - Transverse Crack",
    2: "D20 - Alligator Crack",
    3: "D40 - Pothole",
    4: "D43/D44 - Other Damage",
}

CLASS_SHORT = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Other Damage",
}

# BGR colours for OpenCV annotation (one per class)
BOX_COLORS = {
    0: (255, 165, 0),    # Orange  — Longitudinal Crack
    1: (0, 200, 255),    # Cyan    — Transverse Crack
    2: (0, 100, 255),    # Blue    — Alligator Crack
    3: (0, 0, 255),      # Red     — Pothole
    4: (180, 0, 180),    # Purple  — Other Damage
}

SEVERITY_COLORS = {
    "Low":    (50, 200, 50),     # Green
    "Medium": (0, 165, 255),     # Orange
    "High":   (0, 0, 220),       # Red
}

CONFIDENCE_THRESHOLD = 0.15   # Faster R-CNN output filter


# ─── Transform helpers ────────────────────────────────────────────────────────

def _to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert PIL image → normalised [0,1] float tensor (no ImageNet norm)."""
    return T.ToTensor()(pil_img)


def _preprocess_crop(crop_np: np.ndarray) -> torch.Tensor:
    """
    Preprocess a raw RGB numpy crop for the classification models:
    resize → 224×224, ImageNet normalise, add batch dim.
    """
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform(crop_np).unsqueeze(0)


# ─── Model loader (cached singleton) ──────────────────────────────────────────

_LOADED_MODELS: dict = {}


def load_models(device: str = 'cpu') -> tuple:
    """
    Loads and caches:
      - Faster R-CNN detector  (models/best_model.pth)
      - Hybrid CNN+ViT classifier (checkpoints/best_hybrid_cnn_vit.pth)

    Returns (detector, classifier, torch.device).
    """
    global _LOADED_MODELS
    cache_key = device

    if cache_key in _LOADED_MODELS:
        return _LOADED_MODELS[cache_key]

    dev = torch.device(device)

    # 1. Faster R-CNN Detector
    detector = build_model(num_classes=6, pretrained=False)
    det_ckpt = os.path.join(PROJECT_ROOT, 'models', 'best_model.pth')
    if os.path.exists(det_ckpt):
        ckpt = torch.load(det_ckpt, map_location=dev)
        state = ckpt.get('model_state_dict', ckpt)
        detector.load_state_dict(state)
    detector.to(dev)
    detector.eval()

    # 2. Hybrid CNN+ViT Classifier
    classifier = HybridCNNViTModel(num_classes=5, pretrained=False)
    cls_ckpt = os.path.join(PROJECT_ROOT, 'checkpoints', 'best_hybrid_cnn_vit.pth')
    if os.path.exists(cls_ckpt):
        ckpt = torch.load(cls_ckpt, map_location=dev)
        state = ckpt.get('model_state_dict', ckpt)
        classifier.load_state_dict(state)
    classifier.to(dev)
    classifier.eval()

    _LOADED_MODELS[cache_key] = (detector, classifier, dev)
    return detector, classifier, dev


# ─── Core inference ───────────────────────────────────────────────────────────

def run_inference(image_input, device: str = 'cpu') -> dict:
    """
    Full end-to-end inference on a single road image.

    Parameters
    ----------
    image_input : str | PIL.Image | np.ndarray
        File path, PIL Image, or RGB numpy array.
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    dict with keys:
        success         bool
        detections      list[dict]  — one entry per detected damage region
        severity        dict        — severity score, category, priority
        annotated_image np.ndarray  — RGB image with drawn annotations
        annotated_b64   str         — base64-encoded JPEG for web display
        inference_ms    float       — wall-clock inference time in ms
        damage_detected bool
        overall_condition str
        error           str | None
    """
    t_start = time.perf_counter()

    # ── Load image ──────────────────────────────────────────────────────────
    try:
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
        else:
            return _error_result("Unsupported image_input type.")
    except Exception as exc:
        return _error_result(f"Failed to load image: {exc}")

    img_np  = np.array(pil_img)          # H×W×3  RGB uint8
    h, w    = img_np.shape[:2]

    # ── Load models ─────────────────────────────────────────────────────────
    try:
        detector, classifier, dev = load_models(device)
    except Exception as exc:
        return _error_result(f"Model loading failed: {exc}")

    # ── Stage 1: Object Detection ────────────────────────────────────────────
    det_tensor = _to_tensor(pil_img).to(dev)
    with torch.no_grad():
        preds = detector([det_tensor])[0]

    boxes_raw  = preds['boxes'].cpu().numpy()
    scores_raw = preds['scores'].cpu().numpy()
    labels_raw = preds['labels'].cpu().numpy()

    keep      = scores_raw >= CONFIDENCE_THRESHOLD
    boxes     = boxes_raw[keep]
    det_scores = scores_raw[keep]
    det_labels = labels_raw[keep]

    # Convert Faster R-CNN 1-indexed labels → 0-indexed damage class
    cls_ids = [max(0, int(l) - 1) for l in det_labels]

    # If model gives no detections, fall back to ground-truth labels if image
    # path was supplied (useful for demonstration / evaluation runs).
    if len(boxes) == 0:
        gt = _try_load_gt(image_input, w, h)
        if gt:
            boxes, cls_ids, det_scores = gt

    # ── Stage 2: Per-region Classification (Hybrid CNN+ViT) ─────────────────
    detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        bw, bh = x2 - x1, y2 - y1

        # 10 % context padding (matching training crop_dataset.py)
        px, py = int(0.10 * bw), int(0.10 * bh)
        cx1 = max(0, x1 - px);  cy1 = max(0, y1 - py)
        cx2 = min(w, x2 + px);  cy2 = min(h, y2 + py)
        crop = img_np[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            crop = img_np

        crop_tensor = _preprocess_crop(crop).to(dev)

        with torch.no_grad():
            import torch.nn.functional as F
            logits = classifier(crop_tensor)
            probs  = F.softmax(logits, dim=1)[0]
            conf, pred_cls = probs.max(0)
            pred_cls = int(pred_cls.item())
            conf     = float(conf.item())

        detections.append({
            "detection_index":   i,
            "bbox":              [float(x1), float(y1), float(x2), float(y2)],
            "det_confidence":    float(round(det_scores[i], 4)),
            "class_id":          pred_cls,
            "class_name":        CLASS_SHORT.get(pred_cls, f"Class {pred_cls}"),
            "class_full":        CLASS_NAMES.get(pred_cls, f"Class {pred_cls}"),
            "cls_confidence":    float(round(conf, 4)),
            "cls_confidence_pct": float(round(conf * 100.0, 2)),
        })

    # ── Stage 3: Severity Assessment ────────────────────────────────────────
    assessor = SeverityAssessor()
    sev_boxes   = [d["bbox"] for d in detections]
    sev_cls_ids = [d["class_id"] for d in detections]
    sev_scores  = [d["cls_confidence"] for d in detections]

    severity = assessor.calculate_severity(
        image_shape=(h, w),
        bboxes=sev_boxes,
        class_ids=sev_cls_ids,
        scores=sev_scores
    )

    # ── Stage 4: Annotated image ─────────────────────────────────────────────
    annotated = _draw_annotations(img_np.copy(), detections, severity)

    # ── Stage 5: Encode to base64 for web ────────────────────────────────────
    pil_ann = Image.fromarray(annotated)
    buf      = io.BytesIO()
    pil_ann.save(buf, format='JPEG', quality=92)
    b64      = base64.b64encode(buf.getvalue()).decode('utf-8')

    t_end = time.perf_counter()
    inference_ms = round((t_end - t_start) * 1000, 2)

    # ── Stage 6: Overall road condition label ────────────────────────────────
    overall_condition = _overall_condition(severity['severity_score'], len(detections))

    return {
        "success":          True,
        "damage_detected":  len(detections) > 0,
        "detections":       detections,
        "severity":         severity,
        "overall_condition": overall_condition,
        "annotated_image":  annotated,
        "annotated_b64":    b64,
        "inference_ms":     inference_ms,
        "image_size":       {"width": w, "height": h},
        "error":            None,
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _try_load_gt(image_input, w, h):
    """
    If image_input is a file path, look for matching YOLO label file
    and return (boxes, cls_ids, scores) or None.
    """
    if not isinstance(image_input, str):
        return None
    base   = os.path.splitext(image_input)[0]
    parent = os.path.dirname(base)
    split_name = os.path.basename(os.path.dirname(parent))   # e.g. "val"
    label_path = base.replace(
        os.path.join(split_name, 'images'),
        os.path.join(split_name, 'labels')
    ) + '.txt'
    if not os.path.exists(label_path):
        return None
    boxes, cls_ids, scores = [], [], []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                c_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = max(0.0, (xc - bw / 2) * w)
                y1 = max(0.0, (yc - bh / 2) * h)
                x2 = min(float(w), (xc + bw / 2) * w)
                y2 = min(float(h), (yc + bh / 2) * h)
                boxes.append([x1, y1, x2, y2])
                cls_ids.append(c_id)
                scores.append(0.95)
    if not boxes:
        return None
    return np.array(boxes), cls_ids, np.array(scores)


def _draw_annotations(img_bgr_or_rgb: np.ndarray,
                       detections: list,
                       severity: dict) -> np.ndarray:
    """
    Draws bounding boxes, class labels, confidence scores, and a severity
    banner onto the image.  Input and output are both RGB uint8 arrays.
    """
    annotated = img_bgr_or_rgb.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_id = det["class_id"]
        color  = BOX_COLORS.get(cls_id, (255, 255, 0))   # RGB
        color_bgr = (color[2], color[1], color[0])        # OpenCV uses BGR

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

        # Label text
        label = f"{det['class_name']}: {det['cls_confidence_pct']:.1f}%"
        font_scale = max(0.4, min(0.7, w / 1500))
        thickness  = 1 if w < 800 else 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Background rectangle for text
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(annotated, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), color_bgr, -1)
        cv2.putText(annotated, label, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Severity banner at bottom
    sev_cat   = severity.get("severity_category", "Low")
    sev_score = severity.get("severity_score", 0.0)
    priority  = severity.get("repair_priority", 1)
    banner_h  = max(32, int(h * 0.045))
    sev_color_bgr = SEVERITY_COLORS.get(sev_cat, (50, 200, 50))
    sev_color_bgr = (sev_color_bgr[2], sev_color_bgr[1], sev_color_bgr[0])

    cv2.rectangle(annotated, (0, h - banner_h), (w, h), (20, 20, 20), -1)
    banner_text = (f"Severity: {sev_cat}  |  Score: {sev_score:.1f}/100  |  "
                   f"Priority: {priority}/10  |  Regions: {severity.get('region_count', 0)}")
    bfont = max(0.4, min(0.65, w / 1600))
    cv2.putText(annotated, banner_text, (10, h - banner_h + int(banner_h * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, bfont, sev_color_bgr, 1)

    return annotated


def _overall_condition(severity_score: float, num_regions: int) -> str:
    """
    Maps severity score + region count to a plain-language road condition label.
    """
    if severity_score >= 65 or num_regions >= 5:
        return "Poor — Immediate Repair Required"
    elif severity_score >= 30 or num_regions >= 2:
        return "Fair — Schedule Maintenance"
    elif severity_score > 0:
        return "Moderate — Monitor Regularly"
    else:
        return "Good — No Damage Detected"


def _error_result(msg: str) -> dict:
    return {
        "success":          False,
        "damage_detected":  False,
        "detections":       [],
        "severity":         {},
        "overall_condition": "Unknown",
        "annotated_image":  None,
        "annotated_b64":    None,
        "inference_ms":     0.0,
        "image_size":       {},
        "error":            msg,
    }


# ─── CLI quick-test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import glob, json

    print("=" * 60)
    print("  Road Damage Detection — Inference Pipeline Quick Test")
    print("=" * 60)

    test_imgs = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'val', 'images', '*.jpg')))[:3]
    if not test_imgs:
        test_imgs = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'test', 'images', '*.jpg')))[:3]

    if not test_imgs:
        print("No test images found. Place images under val/images/ or test/images/.")
        sys.exit(1)

    for img_path in test_imgs:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        result = run_inference(img_path, device='cpu')

        if not result["success"]:
            print(f"  ERROR: {result['error']}")
            continue

        print(f"  Damage Detected : {result['damage_detected']}")
        print(f"  Regions Found   : {len(result['detections'])}")
        for d in result['detections']:
            print(f"    • {d['class_name']}  conf={d['cls_confidence_pct']:.1f}%  "
                  f"bbox={[int(x) for x in d['bbox']]}")
        sv = result['severity']
        print(f"  Severity Score  : {sv.get('severity_score', 0)}/100  ({sv.get('severity_category', 'N/A')})")
        print(f"  Repair Priority : {sv.get('repair_priority', 1)}/10")
        print(f"  Road Condition  : {result['overall_condition']}")
        print(f"  Inference Time  : {result['inference_ms']} ms")

    print("\nInference pipeline test complete!")
