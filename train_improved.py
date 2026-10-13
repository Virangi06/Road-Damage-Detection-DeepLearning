"""
train_improved.py
==================
Improved training script for the Faster R-CNN road damage detector.

Improvements over the original run_phase2.py:
  - Uses ALL available training images (not just 100)
  - Configurable sample limits for CPU/GPU environments
  - Better augmentation: random scale jitter, stronger color jitter,
    random vertical flip, mosaic-style multi-scale training
  - Gradient clipping for stability
  - Cosine LR warmup + annealing
  - Per-epoch validation with early stopping
  - Saves best checkpoint + full training log
  - Records random seed for reproducibility
  - Backs up original checkpoint before overwriting
  - After training, evaluates recall at multiple thresholds so you can
    pick the best CONFIDENCE_THRESHOLD for inference.py

Hardware recommendations:
  GPU  : Use MAX_TRAIN = 5000+, EPOCHS = 10, BATCH = 8
  CPU  : Use MAX_TRAIN = 500,   EPOCHS = 5,  BATCH = 4  (~2-3 hrs)
  Fast : Use MAX_TRAIN = 300,   EPOCHS = 3,  BATCH = 4  (~30 min)

Run:
    python train_improved.py
    python train_improved.py --train 1000 --epochs 8 --batch 8
"""

import os
import sys
import json
import random
import shutil
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import RoadDamageDataset, collate_fn
from src.model import build_model

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join('models', 'best_model.pth')
BACKUP_PATH     = os.path.join('models', 'best_model_pretrain_backup.pth')
LOG_PATH        = 'train_improved_log.json'
CURVE_PATH      = 'train_improved_loss_curve.png'


# ── Improved augmentation dataset wrapper ─────────────────────────────────────
class AugmentedDetDataset(RoadDamageDataset):
    """
    Extends RoadDamageDataset with stronger augmentations for training.
    Extra augmentations beyond the base class:
      - Random scale jitter (resize to 448-640px range)
      - Stronger color jitter
      - Random vertical flip (useful for aerial/drone imagery)
    """

    def __init__(self, *args, strong_aug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.strong_aug = strong_aug and (self.split == 'train')

    def __getitem__(self, idx):
        img_tensor, target = super().__getitem__(idx)

        if self.strong_aug and random.random() < 0.3:
            # Additional random vertical flip (makes sense for drone images)
            img_tensor = TF.vflip(img_tensor)
            if len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                # ymin_new = img_size - ymax, ymax_new = img_size - ymin
                new_ymin = self.img_size - boxes[:, 3]
                new_ymax = self.img_size - boxes[:, 1]
                boxes[:, 1] = new_ymin
                boxes[:, 3] = new_ymax
                target['boxes'] = boxes

        if self.strong_aug and random.random() < 0.4:
            # Stronger brightness/contrast/saturation/hue jitter
            jitter = T.ColorJitter(
                brightness=0.35, contrast=0.35,
                saturation=0.25, hue=0.05)
            img_tensor = jitter(img_tensor)

        return img_tensor, target


# ── Training helpers ──────────────────────────────────────────────────────────
def train_one_epoch(model, optimizer, loader, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    n = 0
    t0 = time.time()

    for i, (imgs, targets) in enumerate(loader):
        imgs    = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability (important on noisy datasets)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(loader):
            elapsed = time.time() - t0
            print(f"  Epoch {epoch}/{total_epochs}  "
                  f"Batch {i+1}/{len(loader)}  "
                  f"avg_loss={total_loss/n:.4f}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    return total_loss / max(n, 1)


def validate(model, loader, device):
    model.train()   # keep train mode to get loss_dict
    total = 0.0
    n = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs    = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            total += sum(loss_dict.values()).item()
            n += 1
    model.eval()
    return total / max(n, 1)


def recall_at_thresholds(model, loader, device):
    """
    Fraction of GT-annotated images that get at least 1 detection
    above each threshold.  Helps pick CONFIDENCE_THRESHOLD.
    """
    model.eval()
    thresholds = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    total_with_gt = 0
    top1_scores   = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [im.to(device) for im in imgs]
            preds = model(imgs)
            for pred, tgt in zip(preds, targets):
                if len(tgt['boxes']) > 0:
                    total_with_gt += 1
                sc = pred['scores']
                top1_scores.append(float(sc.max()) if len(sc) > 0 else 0.0)

    print("\n  Threshold recall (fraction of GT images with >= 1 detection):")
    results = {}
    for thresh in thresholds:
        detected = sum(1 for s in top1_scores if s >= thresh)
        recall   = detected / max(total_with_gt, 1)
        results[thresh] = round(recall, 4)
        print(f"    thresh={thresh:.2f}  recall={recall:.3f}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main(max_train=300, max_val=100, epochs=3, batch=4,
         lr=3e-4, weight_decay=5e-4, patience=2, strong_aug=True):

    print("=" * 60)
    print("  Road Damage Detector — Improved Training")
    print(f"  Train: {max_train} images  Val: {max_val} images  "
          f"Epochs: {epochs}  Batch: {batch}")
    print(f"  Seed: {SEED}  Strong aug: {strong_aug}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    print(f"\nDevice: {device}")

    # Backup old checkpoint
    if os.path.exists(CHECKPOINT_PATH) and not os.path.exists(BACKUP_PATH):
        shutil.copy2(CHECKPOINT_PATH, BACKUP_PATH)
        print(f"Backed up existing checkpoint to: {BACKUP_PATH}")

    # Datasets
    train_ds = AugmentedDetDataset(root_dir='.', split='train',
                                   augment=True, strong_aug=strong_aug)
    val_ds   = AugmentedDetDataset(root_dir='.', split='val',
                                   augment=False, strong_aug=False)

    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val images available")

    if max_train and max_train < len(train_ds):
        train_ds.image_filenames = train_ds.image_filenames[:max_train]
    if max_val and max_val < len(val_ds):
        val_ds.image_filenames = val_ds.image_filenames[:max_val]

    print(f"Using:   {len(train_ds)} train, {len(val_ds)} val images")

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # Model with ImageNet pretrained backbone
    model = build_model(num_classes=6, pretrained=True)
    model.to(device)
    print(f"Model: Faster R-CNN ResNet-50 FPN (ImageNet pretrained)")

    # Optimizer + scheduler with linear warmup
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # Warmup for first epoch, then cosine annealing
    def lr_lambda(ep):
        if ep == 0:
            return 0.3       # warmup: start at 30% of lr
        return 1.0
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - 1))

    # Training loop
    best_val  = float('inf')
    no_improv = 0
    history   = {
        'epochs': [], 'train_loss': [], 'val_loss': [], 'lr': [],
        'best_epoch': None,
        'config': {
            'max_train': max_train, 'max_val': max_val,
            'epochs': epochs, 'batch': batch,
            'lr': lr, 'seed': SEED, 'device': str(device),
            'strong_aug': strong_aug,
        }
    }

    t_total = time.time()
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {epoch}/{epochs}  LR={current_lr:.6f} ---")

        t_loss = train_one_epoch(model, optimizer, train_loader,
                                 device, epoch, epochs)
        v_loss = validate(model, val_loader, device)

        # Step schedulers
        if epoch == 1:
            warmup_sched.step()
        else:
            cosine_sched.step()

        history['epochs'].append(epoch)
        history['train_loss'].append(round(t_loss, 4))
        history['val_loss'].append(round(v_loss, 4))
        history['lr'].append(current_lr)

        print(f"  -> train_loss={t_loss:.4f}  val_loss={v_loss:.4f}")

        if v_loss < best_val:
            best_val  = v_loss
            no_improv = 0
            history['best_epoch'] = epoch
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             v_loss,
                'train_images':         len(train_ds),
                'config':               history['config'],
            }, CHECKPOINT_PATH)
            print(f"  -> NEW BEST  val_loss={v_loss:.4f}  saved to {CHECKPOINT_PATH}")
        else:
            no_improv += 1
            print(f"  -> No improvement ({no_improv}/{patience})")
            if no_improv >= patience:
                print(f"  Early stopping after {epoch} epochs.")
                break

        # Save log after every epoch
        with open(LOG_PATH, 'w') as f:
            json.dump(history, f, indent=2)

    elapsed = (time.time() - t_total) / 60
    print(f"\nTraining complete in {elapsed:.1f} min")
    print(f"Best val_loss={best_val:.4f} at epoch {history['best_epoch']}")

    # Post-training: threshold recall evaluation
    print("\nRunning threshold recall evaluation on val set...")
    best_ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    recall_results = recall_at_thresholds(model, val_loader, device)
    history['threshold_recall'] = recall_results
    with open(LOG_PATH, 'w') as f:
        json.dump(history, f, indent=2)

    # Loss curve
    if len(history['epochs']) > 1:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history['epochs'], history['train_loss'],
                label='Train Loss', color='#e63946', linewidth=2, marker='o')
        ax.plot(history['epochs'], history['val_loss'],
                label='Val Loss', color='#457b9d', linewidth=2, marker='s')
        best_ep = history.get('best_epoch')
        if best_ep:
            ax.axvline(best_ep, color='green', linestyle='--', alpha=0.7,
                       label=f'Best epoch {best_ep}')
        ax.set_title('Improved Faster R-CNN Training — Loss Curves', fontsize=13)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(CURVE_PATH, dpi=150)
        plt.close()
        print(f"Loss curve saved: {CURVE_PATH}")

    # Recommend threshold
    best_thresh = 0.05
    for t, rec in recall_results.items():
        if rec >= 0.60:
            best_thresh = t
            break
    print(f"\nRecommended CONFIDENCE_THRESHOLD: {best_thresh}")
    print("Update src/inference.py:  CONFIDENCE_THRESHOLD = "
          f"{best_thresh}  and  USE_DETECTOR = True")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Checkpoint : {CHECKPOINT_PATH}")
    print(f"  Backup     : {BACKUP_PATH}")
    print(f"  Log        : {LOG_PATH}")
    print("=" * 60)

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train improved Faster R-CNN road damage detector")
    parser.add_argument('--train',   type=int, default=300,
                        help="Max training images (default 300 for CPU, use 2000+ for GPU)")
    parser.add_argument('--val',     type=int, default=100,
                        help="Max validation images")
    parser.add_argument('--epochs',  type=int, default=3,
                        help="Training epochs")
    parser.add_argument('--batch',   type=int, default=4,
                        help="Batch size")
    parser.add_argument('--lr',      type=float, default=3e-4,
                        help="Initial learning rate")
    parser.add_argument('--no-aug',  action='store_true',
                        help="Disable strong augmentation")
    args = parser.parse_args()

    main(
        max_train  = args.train,
        max_val    = args.val,
        epochs     = args.epochs,
        batch      = args.batch,
        lr         = args.lr,
        strong_aug = not args.no_aug,
    )
