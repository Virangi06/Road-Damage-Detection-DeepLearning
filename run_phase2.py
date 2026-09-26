import os
import sys
import time
import json
import torch

from src.dataset import RoadDamageDataset, CLASS_NAMES
from src.train import run_training
from src.evaluate import run_evaluation
from src.visualize import visualize_predictions

def main():
    print("==================================================================", flush=True)
    print("      STARTING PHASE 2: MODEL TRAINING & EVALUATION PIPELINE       ", flush=True)
    print("==================================================================", flush=True)
    
    start_time = time.time()
    
    # 1. Dataset verification
    print("\n[Step 1/4] Verifying Dataset & Map Schema...", flush=True)
    train_ds = RoadDamageDataset(root_dir='.', split='train', augment=True)
    val_ds = RoadDamageDataset(root_dir='.', split='val', augment=False)
    test_ds = RoadDamageDataset(root_dir='.', split='test', augment=False)
    print(f"Dataset Counts -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}", flush=True)
    print("Class Mapping Schema:")
    for k, v in CLASS_NAMES.items():
        print(f"  Class {k}: {v}")
        
    # 2. Model Training & Validation Loss Tracking
    print("\n[Step 2/4] Executing Deep Learning Model Training (Faster R-CNN ResNet-50 FPN)...", flush=True)
    train_history = run_training(
        epochs=3,
        batch_size=4,
        lr=0.0003,
        sample_train_limit=100,
        sample_val_limit=30,
        pretrained=True,
        augment=True
    )
    
    # 3. Quantitative Evaluation (mAP, Precision, Recall, F1, Confusion Matrix, PR Curves)
    print("\n[Step 3/4] Running Quantitative Performance Evaluation...", flush=True)
    eval_results = run_evaluation(
        model_path='models/best_model.pth',
        root_dir='.',
        split='val',
        sample_limit=30
    )



    
    # 4. Qualitative Visualization of Predictions
    print("\n[Step 4/4] Generating Qualitative Bounding Box Predictions Grid...", flush=True)
    visualize_predictions(
        model_path='models/best_model.pth',
        root_dir='.',
        split='val',
        num_samples=6,
        score_thresh=0.25
    )
    
    elapsed_total = time.time() - start_time
    print("\n==================================================================", flush=True)
    print(f"   PHASE 2 COMPLETED SUCCESSFULLY IN {elapsed_total/60:.2f} MINUTES!    ", flush=True)
    print("==================================================================", flush=True)
    print("Phase 2 Deliverables Generated:")
    print("  1. Models Checkpoint: models/best_model.pth & models/final_model.pth")
    print("  2. Loss Convergence Curve: phase2_training_loss_curve.png")
    print("  3. Training History Data: phase2_training_history.json")
    print("  4. Precision-Recall Curves: phase2_pr_curve.png")
    print("  5. Detection Confusion Matrix: phase2_confusion_matrix.png")
    print("  6. Qualitative Predictions: phase2_predictions_visualization.png")
    print("  7. Performance Metrics JSON: phase2_evaluation_metrics.json")
    print("==================================================================", flush=True)

if __name__ == '__main__':
    main()
