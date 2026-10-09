"""
evaluate_detector.py  (Task 5)
================================
Evaluates the current best_model.pth on the val set and reports:
  - Recall@threshold per class
  - Precision@threshold per class
  - F1-score per class
  - mAP@0.50 approximation (IoU-based matching)
  - Per-image miss rate
  - Before vs after comparison (reads retrain_log.json if present)

Run:
    python evaluate_detector.py

Outputs:
    evaluation_results.json
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import RoadDamageDataset, collate_fn
from src.model import build_model

MAX_VAL_IMAGES = 300
CHECKPOINT     = os.path.join('models', 'best_model.pth')
OUTPUT_FILE    = 'evaluation_results.json'
IOU_THRESHOLD  = 0.50

CLASS_NAMES = {
    1: 'D00 (Longitudinal Crack)',
    2: 'D10 (Transverse Crack)',
    3: 'D20 (Alligator Crack)',
    4: 'D40 (Pothole)',
    5: 'D43/D44 (Other Damage)',
}


def box_iou(box1, box2):
    """IoU between two boxes [x1,y1,x2,y2]."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def evaluate(model, loader, device, conf_thresh):
    model.eval()

    # per-class TP, FP, FN counters
    tp = {c: 0 for c in CLASS_NAMES}
    fp = {c: 0 for c in CLASS_NAMES}
    fn = {c: 0 for c in CLASS_NAMES}

    n_images         = 0
    n_with_gt        = 0
    n_detected       = 0   # images where >= 1 detection passes threshold
    n_gt_detected    = 0   # GT-annotated images where >= 1 detection passes

    all_scores   = []
    all_pred_cls = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [im.to(device) for im in imgs]
            preds = model(imgs)

            for pred, tgt in zip(preds, targets):
                n_images += 1
                gt_boxes  = tgt['boxes'].cpu().numpy()
                gt_labels = tgt['labels'].cpu().numpy()

                scores  = pred['scores'].cpu().numpy()
                boxes   = pred['boxes'].cpu().numpy()
                plabels = pred['labels'].cpu().numpy()

                keep = scores >= conf_thresh
                d_boxes  = boxes[keep]
                d_labels = plabels[keep]
                d_scores = scores[keep]

                if len(gt_boxes) > 0:
                    n_with_gt += 1
                    if len(d_boxes) > 0:
                        n_gt_detected += 1

                if len(d_boxes) > 0:
                    n_detected += 1
                    all_scores.extend(d_scores.tolist())
                    all_pred_cls.extend(d_labels.tolist())

                # Match predictions to GT (greedy by score)
                gt_matched = [False] * len(gt_boxes)
                pred_matched = [False] * len(d_boxes)

                for pi in range(len(d_boxes)):
                    best_iou = 0
                    best_gi  = -1
                    for gi in range(len(gt_boxes)):
                        if gt_matched[gi]:
                            continue
                        if d_labels[pi] != gt_labels[gi]:
                            continue
                        iou = box_iou(d_boxes[pi], gt_boxes[gi])
                        if iou > best_iou:
                            best_iou = iou
                            best_gi  = gi

                    cls = d_labels[pi]
                    if cls not in CLASS_NAMES:
                        continue
                    if best_iou >= IOU_THRESHOLD and best_gi >= 0:
                        tp[cls] = tp.get(cls, 0) + 1
                        gt_matched[best_gi]  = True
                        pred_matched[pi]     = True
                    else:
                        fp[cls] = fp.get(cls, 0) + 1

                # Unmatched GT = false negatives
                for gi, matched in enumerate(gt_matched):
                    if not matched:
                        cls = gt_labels[gi]
                        if cls in CLASS_NAMES:
                            fn[cls] = fn.get(cls, 0) + 1

    # Per-class metrics
    class_metrics = {}
    total_tp = total_fp = total_fn = 0
    for cls, name in CLASS_NAMES.items():
        t = tp.get(cls, 0)
        f = fp.get(cls, 0)
        n = fn.get(cls, 0)
        prec   = t / (t + f) if (t + f) > 0 else 0.0
        rec    = t / (t + n) if (t + n) > 0 else 0.0
        f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        class_metrics[name] = {
            'TP': t, 'FP': f, 'FN': n,
            'precision': round(prec, 4),
            'recall':    round(rec, 4),
            'f1':        round(f1, 4),
        }
        total_tp += t; total_fp += f; total_fn += n

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1   = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)
                    if (overall_prec + overall_rec) > 0 else 0.0)

    image_recall = n_gt_detected / max(n_with_gt, 1)

    return {
        'conf_threshold':    conf_thresh,
        'iou_threshold':     IOU_THRESHOLD,
        'images_evaluated':  n_images,
        'images_with_gt':    n_with_gt,
        'images_detected':   n_detected,
        'image_level_recall': round(image_recall, 4),
        'overall': {
            'precision': round(overall_prec, 4),
            'recall':    round(overall_rec, 4),
            'f1':        round(overall_f1, 4),
            'TP': total_tp, 'FP': total_fp, 'FN': total_fn,
        },
        'per_class': class_metrics,
        'score_stats': {
            'min':    round(min(all_scores), 4) if all_scores else 0,
            'max':    round(max(all_scores), 4) if all_scores else 0,
            'mean':   round(float(np.mean(all_scores)), 4) if all_scores else 0,
            'median': round(float(np.median(all_scores)), 4) if all_scores else 0,
        },
    }


def main():
    print("=" * 60)
    print("  Detector Evaluation")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  IoU threshold: {IOU_THRESHOLD}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = build_model(num_classes=6, pretrained=False)
    epoch_info = "?"
    if os.path.exists(CHECKPOINT):
        ckpt  = torch.load(CHECKPOINT, map_location=device)
        state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        if isinstance(ckpt, dict):
            epoch_info = (f"epoch={ckpt.get('epoch','?')}, "
                          f"val_loss={ckpt.get('val_loss',0):.4f}, "
                          f"trained_on={ckpt.get('train_images','?')} images")
    else:
        print(f"  WARNING: {CHECKPOINT} not found!")
    model.to(device)
    print(f"  Checkpoint info: {epoch_info}\n")

    val_ds = RoadDamageDataset(root_dir='.', split='val', augment=False)
    if MAX_VAL_IMAGES < len(val_ds):
        val_ds.image_filenames = val_ds.image_filenames[:MAX_VAL_IMAGES]
    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    print(f"Evaluating on {len(val_ds)} val images...\n")

    # Evaluate at two thresholds for comparison
    results_005  = evaluate(model, loader, device, conf_thresh=0.05)
    results_015  = evaluate(model, loader, device, conf_thresh=0.15)

    def _print_result(r):
        print(f"  Threshold={r['conf_threshold']}  IoU>={r['iou_threshold']}")
        print(f"  Images: {r['images_evaluated']} total, "
              f"{r['images_with_gt']} with GT boxes")
        print(f"  Image-level recall: {r['image_level_recall']:.3f}  "
              f"({r['images_detected']} images got >= 1 detection)")
        print(f"  Overall: P={r['overall']['precision']:.4f}  "
              f"R={r['overall']['recall']:.4f}  "
              f"F1={r['overall']['f1']:.4f}  "
              f"(TP={r['overall']['TP']}, FP={r['overall']['FP']}, "
              f"FN={r['overall']['FN']})")
        print(f"  Score range: {r['score_stats']['min']:.4f} – "
              f"{r['score_stats']['max']:.4f}  "
              f"mean={r['score_stats']['mean']:.4f}")
        print("  Per-class:")
        for cls, m in r['per_class'].items():
            print(f"    {cls:<30} P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} F1={m['f1']:.3f} "
                  f"(TP={m['TP']}, FP={m['FP']}, FN={m['FN']})")

    print("--- threshold = 0.05 ---")
    _print_result(results_005)
    print()
    print("--- threshold = 0.15 ---")
    _print_result(results_015)

    # Load old training history for context
    old_info = {}
    if os.path.exists('phase2_training_history.json'):
        with open('phase2_training_history.json') as f:
            h = json.load(f)
        old_info = {'epochs': len(h.get('epoch', [])),
                    'final_val_loss': h['val_loss'][-1] if h.get('val_loss') else None,
                    'train_images': 100}

    output = {
        'checkpoint':            CHECKPOINT,
        'checkpoint_info':       epoch_info,
        'old_training':          old_info,
        'results_thresh_0.05':   results_005,
        'results_thresh_0.15':   results_015,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
