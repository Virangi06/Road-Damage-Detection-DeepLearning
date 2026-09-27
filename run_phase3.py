import os
import sys
import time
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data.crop_dataset import get_crop_dataloader, CLASS_MAPPING
from models.cnn_model import CNNBaselineModel
from models.vit_model import ViTBaselineModel
from models.hybrid_model import HybridCNNViTModel
from training.train_cnn import train_cnn
from training.train_vit import train_vit
from training.train_hybrid import train_hybrid

os.makedirs('results', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

def evaluate_classifier(model, val_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    
    per_class_metrics = {}
    for cls_id, cls_name in CLASS_MAPPING.items():
        if cls_id < len(per_class_p):
            per_class_metrics[cls_name] = {
                'precision': round(float(per_class_p[cls_id]), 4),
                'recall': round(float(per_class_r[cls_id]), 4),
                'f1': round(float(per_class_f1[cls_id]), 4)
            }
            
    return {
        'accuracy': round(float(acc) * 100.0, 2),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_score': round(float(f1), 4),
        'per_class_metrics': per_class_metrics
    }

def main():
    print("==================================================================", flush=True)
    print("      STARTING PHASE 3: DL CLASSIFICATION ARCHITECTURES            ", flush=True)
    print("==================================================================", flush=True)
    
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Train CNN Baseline
    print("\n[Step 1/4] Training CNN Baseline (EfficientNet-B0)...", flush=True)
    cnn_history = train_cnn(epochs=3, batch_size=16, lr=0.0003, max_train_samples=300, max_val_samples=100)
    
    # 2. Train ViT Baseline
    print("\n[Step 2/4] Training Vision Transformer Baseline (ViT)...", flush=True)
    vit_history = train_vit(epochs=3, batch_size=16, lr=0.0003, max_train_samples=300, max_val_samples=100)
    
    # 3. Train Hybrid CNN + ViT Proposed Model
    print("\n[Step 3/4] Training Main Proposed Model (Hybrid CNN + ViT)...", flush=True)
    hybrid_history = train_hybrid(epochs=3, batch_size=16, lr=0.0003, max_train_samples=300, max_val_samples=100)
    
    # 4. Quantitative Evaluation of All 3 Models
    print("\n[Step 4/4] Quantitative Evaluation of CNN vs ViT vs Hybrid...", flush=True)
    val_loader = get_crop_dataloader(split='val', batch_size=16, augment=False, max_samples=100)
    
    # Load Best Weights for Evaluation
    cnn_model = CNNBaselineModel(num_classes=5, pretrained=False).to(device)
    cnn_ckpt = torch.load('checkpoints/best_cnn_baseline.pth', map_location=device)
    cnn_model.load_state_dict(cnn_ckpt['model_state_dict'])
    cnn_metrics = evaluate_classifier(cnn_model, val_loader, device)
    
    vit_model = ViTBaselineModel(num_classes=5, pretrained=False).to(device)
    vit_ckpt = torch.load('checkpoints/best_vit_baseline.pth', map_location=device)
    vit_model.load_state_dict(vit_ckpt['model_state_dict'])
    vit_metrics = evaluate_classifier(vit_model, val_loader, device)
    
    hybrid_model = HybridCNNViTModel(num_classes=5, pretrained=False).to(device)
    hybrid_ckpt = torch.load('checkpoints/best_hybrid_cnn_vit.pth', map_location=device)
    hybrid_model.load_state_dict(hybrid_ckpt['model_state_dict'])
    hybrid_metrics = evaluate_classifier(hybrid_model, val_loader, device)
    
    phase3_results = {
        'CNN_Baseline': {
            'history': cnn_history,
            'metrics': cnn_metrics
        },
        'ViT_Baseline': {
            'history': vit_history,
            'metrics': vit_metrics
        },
        'Hybrid_CNN_ViT': {
            'history': hybrid_history,
            'metrics': hybrid_metrics
        }
    }
    
    with open('phase3_classification_metrics.json', 'w') as f:
        json.dump(phase3_results, f, indent=4)
        
    print("\n================== PHASE 3 EVALUATION SUMMARY ==================")
    print(f"{'Model Architecture':<25} | {'Accuracy (%)':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("------------------------------------------------------------------")
    print(f"{'CNN Baseline (EfficientNet)':<25} | {cnn_metrics['accuracy']:<12.2f} | {cnn_metrics['precision']:<10.4f} | {cnn_metrics['recall']:<10.4f} | {cnn_metrics['f1_score']:<10.4f}")
    print(f"{'ViT Baseline (Transformer)':<25} | {vit_metrics['accuracy']:<12.2f} | {vit_metrics['precision']:<10.4f} | {vit_metrics['recall']:<10.4f} | {vit_metrics['f1_score']:<10.4f}")
    print(f"{'Hybrid CNN + ViT (Proposed)':<25} | {hybrid_metrics['accuracy']:<12.2f} | {hybrid_metrics['precision']:<10.4f} | {hybrid_metrics['recall']:<10.4f} | {hybrid_metrics['f1_score']:<10.4f}")
    print("==================================================================\n")
    
    # Plot 1: Validation Loss Convergence Comparison
    epochs_range = list(range(1, len(cnn_history['val_loss']) + 1))
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, cnn_history['val_loss'], label='CNN Baseline (EfficientNet-B0)', color='#e63946', linewidth=2.5, marker='o')
    plt.plot(epochs_range, vit_history['val_loss'], label='ViT Baseline (Transformer)', color='#f4a261', linewidth=2.5, marker='s')
    plt.plot(epochs_range, hybrid_history['val_loss'], label='Hybrid CNN + ViT (Proposed)', color='#2a9d8f', linewidth=2.5, marker='^')
    plt.title('Phase 3 Validation Loss Convergence (CNN vs ViT vs Hybrid)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Validation Loss', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('phase3_training_loss_curves.png', dpi=300)
    plt.close()
    
    # Plot 2: Validation Accuracy Comparison
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, cnn_history['val_acc'], label='CNN Baseline (EfficientNet-B0)', color='#e63946', linewidth=2.5, marker='o')
    plt.plot(epochs_range, vit_history['val_acc'], label='ViT Baseline (Transformer)', color='#f4a261', linewidth=2.5, marker='s')
    plt.plot(epochs_range, hybrid_history['val_acc'], label='Hybrid CNN + ViT (Proposed)', color='#2a9d8f', linewidth=2.5, marker='^')
    plt.title('Phase 3 Validation Accuracy (%) Comparison (CNN vs ViT vs Hybrid)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('phase3_training_accuracy_curves.png', dpi=300)
    plt.close()
    
    elapsed_total = time.time() - start_time
    print(f"--- PHASE 3 COMPLETED SUCCESSFULLY IN {elapsed_total/60:.2f} MINUTES! ---", flush=True)

if __name__ == '__main__':
    main()
