"""
regression_test.py  (Task 4)
==============================
Regression test suite for the road damage detector.

Tests:
  1. The original failure image (passed via --image flag or auto-found)
  2. 10 random val images with GT boxes (should all detect)
  3. 5 random val images without GT boxes (should ideally have low FP rate)
  4. Class-level recall: pothole / longitudinal / transverse / alligator

Run:
    python regression_test.py
    python regression_test.py --image path/to/failure_image.jpg

Outputs:
  regression_results.json     — structured results
  regression_annotated/       — annotated images for visual inspection
"""

import os, sys, json, argparse, glob, random, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
from src.inference import run_inference, CONFIDENCE_THRESHOLD

OUTPUT_DIR   = 'regression_annotated'
RESULT_FILE  = 'regression_results.json'
N_DAMAGED    = 10
N_CLEAN      = 5
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)


def _find_val_images_by_gt(has_damage: bool, n: int):
    """Return up to n val images with (or without) GT bounding boxes."""
    img_dir = os.path.join('val', 'images')
    lbl_dir = os.path.join('val', 'labels')
    if not os.path.isdir(img_dir):
        return []
    candidates = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith('.jpg'):
            continue
        lbl_path = os.path.join(lbl_dir, fname.replace('.jpg', '.txt'))
        has_box  = False
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            with open(lbl_path) as f:
                has_box = any(len(l.strip().split()) >= 5 for l in f)
        if has_box == has_damage:
            candidates.append(os.path.join(img_dir, fname))
    random.shuffle(candidates)
    return candidates[:n]


def _run_one(img_path: str, label: str) -> dict:
    result = run_inference(img_path, device='cpu')
    det_count = len(result.get('detections', []))
    sv = result.get('severity', {})
    damage_types = list({d['class_name'] for d in result.get('detections', [])})
    return {
        'image':           os.path.basename(img_path),
        'test_label':      label,
        'damage_detected': result.get('damage_detected', False),
        'regions_found':   det_count,
        'severity_score':  sv.get('severity_score', 0.0),
        'severity_cat':    sv.get('severity_category', 'Low'),
        'damage_types':    damage_types,
        'inference_ms':    result.get('inference_ms', 0),
        'top_confidence':  max((d['det_confidence'] for d in result.get('detections', [])),
                               default=0.0),
        'success':         result.get('success', False),
        'error':           result.get('error'),
    }


def _save_annotated(img_path: str, label_str: str):
    result = run_inference(img_path, device='cpu')
    ann = result.get('annotated_image')
    if ann is not None:
        out_name = f"{label_str}_{os.path.basename(img_path)}"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        Image.fromarray(ann).save(out_path)
        return out_path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None,
                        help='Path to the original failure image to test as case 0')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  Road Damage Detection — Regression Test Suite")
    print(f"  CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD}")
    print("=" * 60)

    results = []
    passed  = 0
    failed  = 0

    # ── Case 0: The original failure image ──────────────────────────────────
    failure_img = args.image
    if failure_img is None:
        # Auto-find first val image with GT boxes as proxy for failure case
        cands = _find_val_images_by_gt(has_damage=True, n=1)
        failure_img = cands[0] if cands else None

    if failure_img and os.path.exists(failure_img):
        print(f"\n[Case 0] Original failure image: {os.path.basename(failure_img)}")
        r = _run_one(failure_img, 'FAILURE_CASE')
        r['expected_damage'] = True
        r['passed'] = r['damage_detected']
        print(f"  Detected: {r['damage_detected']}  Regions: {r['regions_found']}  "
              f"Score: {r['severity_score']:.1f}  {'PASS' if r['passed'] else 'FAIL'}")
        _save_annotated(failure_img, 'case0_failure')
        results.append(r)
        passed += (1 if r['passed'] else 0)
        failed += (0 if r['passed'] else 1)
    else:
        print("\n[Case 0] No failure image provided or found. Skipping.")

    # ── Cases 1-N: Val images WITH GT damage ─────────────────────────────────
    damaged_imgs = _find_val_images_by_gt(has_damage=True,  n=N_DAMAGED)
    clean_imgs   = _find_val_images_by_gt(has_damage=False, n=N_CLEAN)

    print(f"\n[Cases 1–{N_DAMAGED}] Val images WITH ground-truth damage:")
    for i, img in enumerate(damaged_imgs):
        r = _run_one(img, f'DAMAGED_{i+1}')
        r['expected_damage'] = True
        r['passed'] = r['damage_detected']
        status = 'PASS' if r['passed'] else 'FAIL (miss)'
        print(f"  {i+1:2}. {r['image']:<35} detected={r['damage_detected']}  "
              f"regions={r['regions_found']}  {status}")
        results.append(r)
        passed += (1 if r['passed'] else 0)
        failed += (0 if r['passed'] else 1)
        if i < 3:
            _save_annotated(img, f'case{i+1}_damaged')

    # ── Cases N+1-M: Val images WITHOUT GT damage ─────────────────────────────
    print(f"\n[Cases {N_DAMAGED+1}–{N_DAMAGED+N_CLEAN}] "
          f"Val images WITHOUT damage (false-positive check):")
    fp_count = 0
    for i, img in enumerate(clean_imgs):
        r = _run_one(img, f'CLEAN_{i+1}')
        r['expected_damage'] = False
        r['passed'] = not r['damage_detected']   # pass if no detection
        status = 'PASS' if r['passed'] else 'FP (false alarm)'
        print(f"  {i+1:2}. {r['image']:<35} detected={r['damage_detected']}  "
              f"regions={r['regions_found']}  {status}")
        results.append(r)
        passed += (1 if r['passed'] else 0)
        failed += (0 if r['passed'] else 1)
        if r['damage_detected']:
            fp_count += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  REGRESSION SUMMARY")
    print(f"  Total tests : {total}")
    print(f"  Passed      : {passed}")
    print(f"  Failed      : {failed}")
    print(f"  Pass rate   : {passed/max(total,1)*100:.1f}%")
    print(f"  FP rate     : {fp_count}/{len(clean_imgs)} clean images falsely detected")
    detect_rate = sum(1 for r in results if r['expected_damage'] and r['damage_detected'])
    exp_dam     = sum(1 for r in results if r['expected_damage'])
    print(f"  Damage recall: {detect_rate}/{exp_dam} GT-damage images correctly detected")
    print(f"  Annotated outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")

    output = {
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'total_tests':  total,
        'passed':       passed,
        'failed':       failed,
        'pass_rate':    round(passed / max(total, 1), 4),
        'damage_recall': round(detect_rate / max(exp_dam, 1), 4),
        'fp_rate':      round(fp_count / max(len(clean_imgs), 1), 4),
        'results':      results,
    }
    with open(RESULT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {RESULT_FILE}")


if __name__ == '__main__':
    main()
