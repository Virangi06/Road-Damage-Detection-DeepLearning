"""
calibrate_threshold.py  (Task 3)
=================================
After retraining, run this script to find the optimal CONFIDENCE_THRESHOLD
for the new model based on val-set score statistics.

It:
  1. Loads the new best_model.pth checkpoint
  2. Runs inference on up to 200 val images
  3. Collects ALL prediction scores (before any threshold filter)
  4. Computes recall@N (fraction of GT-annotated images with >=1 detection)
     at a sweep of thresholds
  5. Prints a recommendation and patches src/inference.py automatically

Run:
    python calibrate_threshold.py
"""

import os, sys, json, re, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import RoadDamageDataset, collate_fn
from src.model import build_model

# ── Config ────────────────────────────────────────────────────────────────────
MAX_VAL_IMAGES = 200
CHECKPOINT     = os.path.join('models', 'best_model.pth')
INFERENCE_FILE = os.path.join('src', 'inference.py')
RESULT_FILE    = 'calibration_results.json'

SWEEP = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def main():
    print("=" * 60)
    print("  Confidence Threshold Calibration")
    print(f"  Checkpoint: {CHECKPOINT}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = build_model(num_classes=6, pretrained=False)
    if os.path.exists(CHECKPOINT):
        ckpt  = torch.load(CHECKPOINT, map_location=device)
        state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        if isinstance(ckpt, dict):
            print(f"  Loaded checkpoint: epoch={ckpt.get('epoch','?')}, "
                  f"val_loss={ckpt.get('val_loss',0):.4f}, "
                  f"trained_on={ckpt.get('train_images','?')} images")
    else:
        print(f"  WARNING: {CHECKPOINT} not found, using random weights")
    model.to(device)
    model.eval()

    # Val dataset
    val_ds = RoadDamageDataset(root_dir='.', split='val', augment=False)
    if MAX_VAL_IMAGES < len(val_ds):
        val_ds.image_filenames = val_ds.image_filenames[:MAX_VAL_IMAGES]
    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    print(f"\nEvaluating on {len(val_ds)} val images...")

    # Collect scores and whether GT boxes exist
    all_scores     = []   # every score from every prediction
    has_gt         = []   # whether each image has >= 1 GT box
    top1_scores    = []   # highest score per image

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(loader):
            imgs = [im.to(device) for im in imgs]
            preds = model(imgs)
            scores = preds[0]['scores'].cpu().numpy()

            gt_has = len(targets[0]['boxes']) > 0
            has_gt.append(gt_has)
            all_scores.extend(scores.tolist())
            top1_scores.append(float(scores.max()) if len(scores) > 0 else 0.0)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(val_ds)}]  scores so far: "
                      f"min={min(all_scores):.4f} max={max(all_scores):.4f} "
                      f"mean={np.mean(all_scores):.4f}")

    total    = len(has_gt)
    with_gt  = sum(has_gt)
    print(f"\nImages with GT boxes: {with_gt}/{total}")
    if all_scores:
        print(f"Score statistics across all {len(all_scores)} predictions:")
        print(f"  min={min(all_scores):.4f}  max={max(all_scores):.4f}  "
              f"mean={np.mean(all_scores):.4f}  median={np.median(all_scores):.4f}")
        print(f"  p90={np.percentile(all_scores, 90):.4f}  "
              f"p95={np.percentile(all_scores, 95):.4f}  "
              f"p99={np.percentile(all_scores, 99):.4f}")

    # Threshold sweep — recall on GT-annotated images
    print("\nThreshold sweep (recall = fraction of GT images with >= 1 detection):")
    print(f"  {'Threshold':>10}  {'Recall':>8}  {'Det/img':>8}  {'Recommendation'}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*20}")

    sweep_results = []
    best_thresh   = 0.05
    best_score    = -1

    for thresh in SWEEP:
        # recall: of GT-annotated images, how many get >= 1 detection?
        detected_gt = 0
        total_dets  = 0
        for i, top in enumerate(top1_scores):
            if top >= thresh:
                total_dets += 1
                if has_gt[i]:
                    detected_gt += 1
        recall   = detected_gt / max(with_gt, 1)
        avg_dets = total_dets / max(total, 1)

        # Simple heuristic: maximise recall while keeping avg_dets < 25
        # (too many detections = noise; too few = misses)
        score = recall - 0.02 * max(0, avg_dets - 15)

        note = ""
        if thresh == SWEEP[0]:
            note = "too many FPs likely"
        elif recall >= 0.80 and avg_dets < 20:
            note = "GOOD balance"
        elif recall >= 0.70:
            note = "acceptable"
        elif recall < 0.30:
            note = "too strict"

        if score > best_score:
            best_score  = score
            best_thresh = thresh

        print(f"  {thresh:>10.2f}  {recall:>8.3f}  {avg_dets:>8.2f}  {note}")
        sweep_results.append({
            'threshold': thresh, 'recall': round(recall, 4),
            'avg_detections': round(avg_dets, 2),
        })

    print(f"\nRecommended threshold: {best_thresh}")

    # Save calibration results
    calibration = {
        'checkpoint':              CHECKPOINT,
        'val_images':              total,
        'images_with_gt':          with_gt,
        'score_min':               round(min(all_scores), 4) if all_scores else 0,
        'score_max':               round(max(all_scores), 4) if all_scores else 0,
        'score_mean':              round(float(np.mean(all_scores)), 4) if all_scores else 0,
        'score_median':            round(float(np.median(all_scores)), 4) if all_scores else 0,
        'score_p95':               round(float(np.percentile(all_scores, 95)), 4) if all_scores else 0,
        'recommended_threshold':   best_thresh,
        'sweep':                   sweep_results,
    }
    with open(RESULT_FILE, 'w') as f:
        json.dump(calibration, f, indent=2)
    print(f"Results saved to: {RESULT_FILE}")

    # Patch src/inference.py with the recommended threshold
    _patch_inference(best_thresh)

    print("\nCalibration complete.")
    return best_thresh


def _patch_inference(new_thresh: float):
    """Update CONFIDENCE_THRESHOLD in src/inference.py."""
    if not os.path.exists(INFERENCE_FILE):
        print(f"  Cannot patch {INFERENCE_FILE} — file not found")
        return

    with open(INFERENCE_FILE, 'r', encoding='utf-8') as f:
        src = f.read()

    old_pattern = re.compile(r'^CONFIDENCE_THRESHOLD\s*=\s*[\d.]+', re.MULTILINE)
    if not old_pattern.search(src):
        print(f"  Pattern not found in {INFERENCE_FILE}, skipping patch")
        return

    shutil.copy2(INFERENCE_FILE, INFERENCE_FILE + '.bak')
    new_src = old_pattern.sub(f'CONFIDENCE_THRESHOLD = {new_thresh}', src)
    with open(INFERENCE_FILE, 'w', encoding='utf-8') as f:
        f.write(new_src)
    print(f"\nPatched {INFERENCE_FILE}:")
    print(f"  CONFIDENCE_THRESHOLD = {new_thresh}")


if __name__ == '__main__':
    main()
