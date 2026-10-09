"""
run_post_retrain.py
====================
Runs the complete post-retraining pipeline sequentially:
  Step 1: Calibrate confidence threshold from retrained model
  Step 2: Evaluate retrained model (precision, recall, F1)
  Step 3: Run regression tests
  Step 4: Print final summary comparing old vs new

Run AFTER retrain_detector.py completes:
    python run_post_retrain.py
"""

import os, sys, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT = os.path.join('models', 'best_model.pth')


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def step(n, title):
    print(f"\n{'='*60}")
    print(f"  Step {n}: {title}")
    print(f"{'='*60}\n")


def main():
    print("=" * 60)
    print("  Post-Retraining Pipeline")
    print("=" * 60)

    # Verify checkpoint exists and was updated
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: {CHECKPOINT} not found. Run retrain_detector.py first.")
        sys.exit(1)

    import torch
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    if isinstance(ckpt, dict):
        epoch    = ckpt.get('epoch', '?')
        val_loss = ckpt.get('val_loss', '?')
        n_train  = ckpt.get('train_images', '?')
        print(f"Checkpoint: epoch={epoch}, val_loss={val_loss:.4f}, "
              f"trained_on={n_train} images")
    else:
        print("Checkpoint loaded (plain state dict)")

    # ── Step 1: Calibrate ────────────────────────────────────────────────────
    step(1, "Confidence Threshold Calibration")
    import importlib.util
    spec = importlib.util.spec_from_file_location("calibrate", "calibrate_threshold.py")
    cal_mod = importlib.util.load_module(spec)  # will call main() below

    from calibrate_threshold import main as cal_main
    best_thresh = cal_main()
    print(f"\nCalibrated threshold: {best_thresh}")

    # ── Step 2: Evaluate ─────────────────────────────────────────────────────
    step(2, "Model Evaluation (Precision / Recall / F1)")
    from evaluate_detector import main as eval_main
    eval_main()

    # ── Step 3: Regression tests ─────────────────────────────────────────────
    step(3, "Regression Tests")
    from regression_test import main as reg_main
    reg_main()

    # ── Step 4: Final summary ─────────────────────────────────────────────────
    step(4, "Final Summary")

    cal_data  = _load_json('calibration_results.json')
    eval_data = _load_json('evaluation_results.json')
    reg_data  = _load_json('regression_results.json')
    log_data  = _load_json('retrain_log.json')

    print("RETRAINING:")
    if log_data:
        for ep, tl, vl in zip(log_data.get('epochs', []),
                               log_data.get('train_loss', []),
                               log_data.get('val_loss', [])):
            print(f"  Epoch {ep}: train_loss={tl:.4f}  val_loss={vl:.4f}")
        print(f"  Best epoch: {log_data.get('best_epoch')}")

    print("\nCALIBRATION:")
    print(f"  Recommended threshold: {cal_data.get('recommended_threshold')}")
    print(f"  Score range: {cal_data.get('score_min'):.4f} – "
          f"{cal_data.get('score_max'):.4f}")

    print("\nEVALUATION (thresh=0.05):")
    r005 = eval_data.get('results_thresh_0.05', {})
    ov   = r005.get('overall', {})
    print(f"  Precision: {ov.get('precision', 0):.4f}")
    print(f"  Recall:    {ov.get('recall', 0):.4f}")
    print(f"  F1:        {ov.get('f1', 0):.4f}")
    print(f"  Image-level recall: {r005.get('image_level_recall', 0):.3f}")

    print("\nREGRESSION:")
    print(f"  Total tests:    {reg_data.get('total_tests', 0)}")
    print(f"  Passed:         {reg_data.get('passed', 0)}")
    print(f"  Damage recall:  {reg_data.get('damage_recall', 0):.3f}")
    print(f"  FP rate:        {reg_data.get('fp_rate', 0):.3f}")

    print("\nCOMPARISON (original 100-img model vs retrained):")
    print(f"  Original mAP@50: 0.0 (all classes)")
    r_rec = r005.get('image_level_recall', 0)
    print(f"  Retrained image-level recall: {r_rec:.3f}")
    print(f"  Original detections on val: 0 (threshold=0.15, scores 0.05-0.14)")
    score_max = cal_data.get('score_max', 0)
    print(f"  Retrained max score: {score_max:.4f}")

    print("\n" + "=" * 60)
    print("  POST-RETRAIN PIPELINE COMPLETE")
    print("  All results saved to JSON files in project root.")
    print("=" * 60)


if __name__ == '__main__':
    main()
