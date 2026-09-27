import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, log_loss
)

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

CLASS_NAMES = [
    'D00 (Longitudinal Crack)',
    'D10 (Transverse Crack)',
    'D20 (Alligator Crack)',
    'D40 (Pothole)',
    'D43/D44 (Other Damage)'
]

class ModelProfiler:
    """
    Profiles Deep Learning Model Parameters, Memory Footprint, and Latency/Throughput.
    """
    def __init__(self, model, checkpoint_path=None, device='cpu'):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def profile_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        file_size_mb = 0.0
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            file_size_mb = os.path.getsize(self.checkpoint_path) / (1024.0 * 1024.0)
            
        return {
            "total_params": total_params,
            "total_params_m": round(total_params / 1e6, 3),
            "trainable_params": trainable_params,
            "trainable_params_m": round(trainable_params / 1e6, 3),
            "file_size_mb": round(file_size_mb, 2)
        }

    def profile_latency(self, input_shape=(1, 3, 224, 224), iterations=100, warmup=10):
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
                
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                _ = self.model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    
        elapsed_time = time.perf_counter() - start_time
        latency_ms_per_crop = (elapsed_time / iterations) * 1000.0
        fps = 1000.0 / latency_ms_per_crop if latency_ms_per_crop > 0 else 0.0
        
        return {
            "latency_ms_per_crop": round(latency_ms_per_crop, 3),
            "fps": round(fps, 2)
        }


class ModelEvaluator:
    """
    Evaluates PyTorch classification models on dataset split and produces structured metrics.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, dataloader):
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)
        
        acc = accuracy_score(y_true, y_pred) * 100.0
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        
        # Per-class metrics
        per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
        
        try:
            loss_val = log_loss(y_true, y_probs, labels=list(range(len(CLASS_NAMES))))
        except Exception:
            loss_val = 0.0
            
        class_breakdown = {}
        for idx, cls_name in enumerate(CLASS_NAMES):
            class_breakdown[cls_name] = {
                "precision": round(float(per_class_p[idx]), 4),
                "recall": round(float(per_class_r[idx]), 4),
                "f1_score": round(float(per_class_f1[idx]), 4),
                "support": int(per_class_supp[idx])
            }
            
        return {
            "accuracy": round(float(acc), 2),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "micro_f1": round(float(micro_f1), 4),
            "log_loss": round(float(loss_val), 4),
            "confusion_matrix": cm.tolist(),
            "class_breakdown": class_breakdown,
            "y_true": y_true,
            "y_pred": y_pred
        }


if __name__ == '__main__':
    print("Testing ModelProfiler & ModelEvaluator...")
    from models.cnn_model import CNNBaselineModel
    model = CNNBaselineModel(pretrained=False)
    profiler = ModelProfiler(model)
    p_info = profiler.profile_parameters()
    l_info = profiler.profile_latency()
    print(f"Profiler Info: {p_info}")
    print(f"Latency Info: {l_info}")
    print("Benchmark module test passed!")
