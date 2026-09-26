import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from torch.utils.data import DataLoader

from src.dataset import RoadDamageDataset, collate_fn, CLASS_NAMES
from src.model import build_model

def calculate_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between boxA and boxB.
    box format: [xmin, ymin, xmax, ymax]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def compute_ap(recalls, precisions):
    """
    Computes Average Precision (AP) using standard 11-point or COCO area integration.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)

def evaluate_model(model, data_loader, device, iou_thresh=0.5, score_thresh=0.3):
    """
    Evaluates model on data_loader and returns mAP, precision, recall, F1, and PR curves.
    """
    model.eval()
    
    # Store predictions and ground truths per class
    # class_id range: 1..5
    gt_boxes_per_class = defaultdict(list)
    pred_boxes_per_class = defaultdict(list)
    
    img_idx = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for target, output in zip(targets, outputs):
                # Target boxes and labels
                t_boxes = target['boxes'].cpu().numpy()
                t_labels = target['labels'].cpu().numpy()
                
                for b, l in zip(t_boxes, t_labels):
                    gt_boxes_per_class[int(l)].append({
                        'img_id': img_idx,
                        'box': b,
                        'matched': False
                    })
                    
                # Predicted boxes, labels, and scores
                p_boxes = output['boxes'].cpu().numpy()
                p_labels = output['labels'].cpu().numpy()
                p_scores = output['scores'].cpu().numpy()
                
                for b, l, s in zip(p_boxes, p_labels, p_scores):
                    if s >= score_thresh:
                        pred_boxes_per_class[int(l)].append({
                            'img_id': img_idx,
                            'box': b,
                            'score': float(s)
                        })
                        
                img_idx += 1

    # Confusion matrix structure: 6x6 (Background + 5 classes)
    conf_matrix = np.zeros((6, 6), dtype=int)
    
    class_metrics = {}
    pr_curves_data = {}
    
    all_aps = []
    
    for cls_id in range(1, 6):
        cls_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        gts = gt_boxes_per_class[cls_id]
        preds = pred_boxes_per_class[cls_id]
        
        # Sort predictions by score descending
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        
        n_gt = len(gts)
        n_pred = len(preds)
        
        if n_gt == 0 and n_pred == 0:
            class_metrics[cls_name] = {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'ap50': 1.0, 'support': 0}
            all_aps.append(1.0)
            continue
        elif n_gt == 0 or n_pred == 0:
            class_metrics[cls_name] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'ap50': 0.0, 'support': n_gt}
            all_aps.append(0.0)
            continue
            
        tp = np.zeros(n_pred)
        fp = np.zeros(n_pred)
        
        # Create image to gt map for fast lookup
        gt_img_map = defaultdict(list)
        for g in gts:
            gt_img_map[g['img_id']].append(g)
            
        for i, p in enumerate(preds):
            img_gts = gt_img_map[p['img_id']]
            best_iou = 0.0
            best_gt = None
            
            for g in img_gts:
                iou = calculate_iou(p['box'], g['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = g
                    
            if best_iou >= iou_thresh:
                if not best_gt['matched']:
                    tp[i] = 1.0
                    best_gt['matched'] = True
                    conf_matrix[cls_id, cls_id] += 1
                else:
                    fp[i] = 1.0
                    conf_matrix[0, cls_id] += 1  # False positive (background predicted as cls)
            else:
                fp[i] = 1.0
                conf_matrix[0, cls_id] += 1
                
        # Calculate unmatched GT as False Negatives
        for g in gts:
            if not g['matched']:
                conf_matrix[cls_id, 0] += 1
                
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        recalls = cum_tp / float(n_gt)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
        
        ap = compute_ap(recalls, precisions)
        all_aps.append(ap)
        
        final_p = float(precisions[-1]) if len(precisions) > 0 else 0.0
        final_r = float(recalls[-1]) if len(recalls) > 0 else 0.0
        f1 = (2 * final_p * final_r) / (final_p + final_r + 1e-6)
        
        class_metrics[cls_name] = {
            'precision': round(final_p, 4),
            'recall': round(final_r, 4),
            'f1': round(f1, 4),
            'ap50': round(ap, 4),
            'support': n_gt
        }
        
        pr_curves_data[cls_name] = {
            'recall': recalls.tolist(),
            'precision': precisions.tolist()
        }

    mAP50 = float(np.mean(all_aps))
    
    results = {
        'mAP50': round(mAP50, 4),
        'class_metrics': class_metrics,
        'pr_curves': pr_curves_data,
        'confusion_matrix': conf_matrix.tolist()
    }
    
    return results

def run_evaluation(model_path='models/best_model.pth', root_dir='.', split='val', sample_limit=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- STARTING PHASE 2 QUANTITATIVE EVALUATION ON SPLIT: {split} ---", flush=True)
    
    dataset = RoadDamageDataset(root_dir=root_dir, split=split, augment=False)
    if sample_limit and sample_limit < len(dataset):
        dataset.image_filenames = dataset.image_filenames[:sample_limit]
        
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    model = build_model(num_classes=6, pretrained=False)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained checkpoint from {model_path}", flush=True)
    else:
        print(f"Warning: {model_path} not found. Running evaluation on uninitialized model for baseline testing.", flush=True)
        
    model.to(device)
    
    results = evaluate_model(model, data_loader, device, iou_thresh=0.5, score_thresh=0.25)
    
    print("\n=================== PHASE 2 EVALUATION METRICS ===================")
    print(f"Overall mAP@0.50: {results['mAP50']:.4f}")
    print("------------------------------------------------------------------")
    print(f"{'Class Name':<30} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'AP@0.50':<9}")
    print("------------------------------------------------------------------")
    for cls_name, metrics in results['class_metrics'].items():
        print(f"{cls_name:<30} | {metrics['precision']:<9.4f} | {metrics['recall']:<9.4f} | {metrics['f1']:<9.4f} | {metrics['ap50']:<9.4f}")
    print("==================================================================\n")
    
    # Save metrics JSON
    with open('phase2_evaluation_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved evaluation metrics to phase2_evaluation_metrics.json", flush=True)

    # Plot 1: Precision-Recall Curves
    plt.figure(figsize=(9, 6))
    colors = ['#e63946', '#f4a261', '#2a9d8f', '#457b9d', '#9d4edd']
    for idx, (cls_name, pr_data) in enumerate(results['pr_curves'].items()):
        if len(pr_data['recall']) > 0:
            plt.plot(pr_data['recall'], pr_data['precision'], label=f"{cls_name} (AP={results['class_metrics'][cls_name]['ap50']:.3f})", color=colors[idx % len(colors)], linewidth=2)
            
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall (PR) Curves per Road Damage Class (Phase 2)', fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=10, loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('phase2_pr_curve.png', dpi=300)
    plt.close()
    
    # Plot 2: Confusion Matrix Heatmap
    plt.figure(figsize=(8, 7))
    labels_display = ['Background', 'D00', 'D10', 'D20', 'D40', 'D43/D44']
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_display, yticklabels=labels_display)
    plt.title('Phase 2 Road Damage Detection Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Ground Truth Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('phase2_confusion_matrix.png', dpi=300)
    plt.close()

    print("Generated phase2_pr_curve.png and phase2_confusion_matrix.png.", flush=True)
    return results

if __name__ == '__main__':
    run_evaluation()
