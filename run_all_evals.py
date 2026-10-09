"""
run_all_evals.py
=================
Runs calibration → evaluation → regression tests sequentially on the
current best_model.pth checkpoint (works with both old and retrained model).
Total time: ~15-20 minutes on CPU.

Run:
    python run_all_evals.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n", flush=True)

if __name__ == '__main__':
    print("="*60)
    print("  Road Damage AI - Post-Fix Evaluation Pipeline")
    print("="*60, flush=True)

    # Step 1: Calibration
    section("Step 1/3: Confidence Threshold Calibration")
    from calibrate_threshold import main as cal_main
    best_thresh = cal_main()

    # Step 2: Evaluation
    section("Step 2/3: Model Evaluation (Precision / Recall / F1)")
    from evaluate_detector import main as eval_main
    eval_main()

    # Step 3: Regression
    section("Step 3/3: Regression Tests")
    from regression_test import main as reg_main
    reg_main()

    # Final summary
    section("FINAL SUMMARY")
    cal  = _load_json('calibration_results.json')
    evl  = _load_json('evaluation_results.json')
    reg  = _load_json('regression_results.json')

    print("CALIBRATION:")
    print(f"  Score range      : {cal.get('score_min',0):.4f} - {cal.get('score_max',0):.4f}")
    print(f"  Recommended thresh: {cal.get('recommended_threshold', 0.05)}")

    r005 = evl.get('results_thresh_0.05', {})
    ov   = r005.get('overall', {})
    print("\nEVALUATION (thresh=0.05):")
    print(f"  Precision       : {ov.get('precision',0):.4f}")
    print(f"  Recall          : {ov.get('recall',0):.4f}")
    print(f"  F1-Score        : {ov.get('f1',0):.4f}")
    print(f"  Image recall    : {r005.get('image_level_recall',0):.3f}")

    print("\nREGRESSION:")
    print(f"  Tests passed    : {reg.get('passed',0)}/{reg.get('total_tests',0)}")
    print(f"  Damage recall   : {reg.get('damage_recall',0):.3f}")
    print(f"  False-pos rate  : {reg.get('fp_rate',0):.3f}")

    print("\nBEFORE vs AFTER FIX:")
    print(f"  BEFORE: 0/5 val images detected (threshold=0.15, max score=0.14)")
    print(f"  AFTER : {reg.get('damage_recall',0)*100:.0f}% GT images detected (threshold=0.05)")
    print(f"\nAll output files saved to project root.")
    print(f"Run 'python retrain_detector.py' overnight for a better model.")
