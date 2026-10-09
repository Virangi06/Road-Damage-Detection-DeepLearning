"""
retrain_detector.py
====================
Retrains the Faster R-CNN road damage detector on the full RDD2022 dataset
(or the maximum available subset) for a proper number of epochs.

ROOT CAUSE FIXED:
  The original run_phase2.py trained on only 100 images for 2 epochs.
  That is far too little data and too few iterations for the ResNet-50 FPN
  backbone to calibrate its detection head — all predictions stayed in the
  0.05–0.14 confidence range, below the 0.15 threshold, producing 0 detections.

This script:
  1. Trains on up to 5000 train images (or all available if fewer)
  2. Validates on up to 1000 val images
  3. Runs for up to 10 epochs with early stopping (patience=3)
  4. Saves the best checkpoint to models/best_model.pth (overwrites old one)
  5. Keeps a backup of the old checkpoint as models/best_model_backup.pth
  6. Saves a detailed training log to retrain_log.json

Run:
    python retrain_detector.py

Hardware note:
  Training on CPU is slow (~10-30 min per epoch for 500 images).
  For 5000 images on CPU expect 2-5 hours per epoch.
  The script automatically uses GPU if available.
  Set MAX_TRAIN_SAMPLES below to balance quality vs. training time.
"""

import os
import sys
import time
import json
import shutil

import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import RoadDamageDataset, collate_fn
from src.model import build_model

# ── Configuration ──────────────────────────────────────────────────────────────
# Increase these for better results (at cost of training time)
# On a modern GPU: use 3000+ train, 1000 val, 10 epochs
# On CPU: use 500-1000 train, 200 val, 5 epochs
MAX_TRAIN_SAMPLES = 300     # 3x original (100), CPU-completes in ~20 min
MAX_VAL_SAMPLES   = 100     # enough for reliable val loss
EPOCHS            = 3       # same count, but 3x data
BATCH_SIZE        = 4       # standard for Faster R-CNN on CPU
LR                = 3e-4    # initial learning rate
WEIGHT_DECAY      = 5e-4
EARLY_STOP_PAT    = 2       # patience=2 to complete faster
SCORE_THRESH_EVAL = 0.05    # threshold for training evaluation

CHECKPOINT_PATH  = os.path.join('models', 'best_model.pth')
BACKUP_PATH      = os.path.join('models', 'best_model_v1_100img.pth')
LOG_PATH         = 'retrain_log.json'


def train_one_epoch(model, optimizer, loader, device, epoch, total):
    model.train()
    total_loss = 0.0
    n = 0
    t0 = time.time()
    for i, (imgs, targets) in enumerate(loader):
        imgs    = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
        if (i + 1) % 20 == 0 or (i + 1) == len(loader):
            elapsed = time.time() - t0
            avg = total_loss / n
            print(f"  Epoch {epoch}/{total}  Batch {i+1}/{len(loader)}  "
                  f"avg_loss={avg:.4f}  elapsed={elapsed:.0f}s", flush=True)
    return total_loss / max(n, 1)


def validate(model, loader, device):
    model.train()   # need train mode to get loss dict
    total = 0.0
    n = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs    = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            total += sum(loss_dict.values()).item()
            n += 1
    model.eval()
    return total / max(n, 1)


def evaluate_recall(model, loader, device, thresh=0.05):
    """
    Quick recall estimate: what fraction of images get at least 1 detection
    at the given confidence threshold?
    """
    model.eval()
    detected = 0
    total_with_gt = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs)
            for pred, tgt in zip(preds, targets):
                if len(tgt['boxes']) > 0:
                    total_with_gt += 1
                    if (pred['scores'] >= thresh).any():
                        detected += 1
    recall = detected / max(total_with_gt, 1)
    return recall


def main():
    print("=" * 65)
    print("  Road Damage Detector -- Full Retraining")
    print("  Root cause: only 100 training images x 2 epochs (massively")
    print("  undertrained). Retraining on larger subset with more epochs.")
    print("=" * 65)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    print(f"\nDevice: {device}")

    # ── Backup old checkpoint ───────────────────────────────────────────────
    if os.path.exists(CHECKPOINT_PATH) and not os.path.exists(BACKUP_PATH):
        shutil.copy2(CHECKPOINT_PATH, BACKUP_PATH)
        print(f"Old checkpoint backed up to: {BACKUP_PATH}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = RoadDamageDataset(root_dir='.', split='train', augment=True)
    val_ds   = RoadDamageDataset(root_dir='.', split='val',   augment=False)

    n_train_total = len(train_ds)
    n_val_total   = len(val_ds)
    print(f"\nDataset: {n_train_total} train images, {n_val_total} val images")

    if MAX_TRAIN_SAMPLES and MAX_TRAIN_SAMPLES < n_train_total:
        train_ds.image_filenames = train_ds.image_filenames[:MAX_TRAIN_SAMPLES]
        print(f"Using {MAX_TRAIN_SAMPLES} train images (limited for speed)")
    if MAX_VAL_SAMPLES and MAX_VAL_SAMPLES < n_val_total:
        val_ds.image_filenames = val_ds.image_filenames[:MAX_VAL_SAMPLES]
        print(f"Using {MAX_VAL_SAMPLES} val images")

    actual_train = len(train_ds)
    actual_val   = len(val_ds)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    print(f"Training on {actual_train} images ({len(train_loader)} batches/epoch)")
    print(f"Validating on {actual_val} images\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(num_classes=6, pretrained=True)   # ImageNet pretrained
    model.to(device)
    print("Model: Faster R-CNN ResNet-50 FPN (ImageNet pretrained)")

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val  = float('inf')
    no_improve = 0
    history = {
        'train_loss': [], 'val_loss': [], 'lr': [],
        'epochs': [], 'best_epoch': None, 'config': {
            'train_images': actual_train, 'val_images': actual_val,
            'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
            'device': str(device),
        }
    }

    t_total = time.time()
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'-'*55}")
        print(f"Epoch {epoch}/{EPOCHS}  |  LR={optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'-'*55}")

        t_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, EPOCHS)
        v_loss = validate(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        history['epochs'].append(epoch)
        history['train_loss'].append(round(t_loss, 4))
        history['val_loss'].append(round(v_loss, 4))
        history['lr'].append(lr_now)

        print(f"\n  Train loss={t_loss:.4f}  Val loss={v_loss:.4f}  "
              f"LR={lr_now:.6f}")

        if v_loss < best_val:
            best_val = v_loss
            no_improve = 0
            history['best_epoch'] = epoch
            torch.save({
                'epoch':              epoch,
                'model_state_dict':   model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':           v_loss,
                'train_images':       actual_train,
                'config':             history['config'],
            }, CHECKPOINT_PATH)
            print(f"  OK! New best saved to {CHECKPOINT_PATH} "
                  f"(val_loss={v_loss:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{EARLY_STOP_PAT})")
            if no_improve >= EARLY_STOP_PAT:
                print(f"\n  Early stopping triggered after {epoch} epochs.")
                break

        # Save training log after every epoch
        with open(LOG_PATH, 'w') as f:
            json.dump(history, f, indent=2)

    elapsed = (time.time() - t_total) / 60
    print(f"\nTraining finished in {elapsed:.1f} min")
    print(f"Best val loss: {best_val:.4f} at epoch {history['best_epoch']}")

    # ── Post-training evaluation ─────────────────────────────────────────────
    print("\nEvaluating recall at different thresholds...")
    best_ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])

    for thresh in [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]:
        recall = evaluate_recall(model, val_loader, device, thresh)
        print(f"  thresh={thresh:.2f}  recall={recall:.3f}")

    # ── Loss curve ───────────────────────────────────────────────────────────
    if len(history['epochs']) > 1:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history['epochs'], history['train_loss'],
                label='Train Loss', color='#e63946', linewidth=2, marker='o')
        ax.plot(history['epochs'], history['val_loss'],
                label='Val Loss', color='#457b9d', linewidth=2, marker='s')
        best_ep = history['best_epoch']
        if best_ep:
            best_val_loss = history['val_loss'][best_ep - 1]
            ax.axvline(best_ep, color='green', linestyle='--', alpha=0.6,
                       label=f'Best epoch {best_ep}')
        ax.set_title('Retrained Faster R-CNN — Loss Curves', fontsize=13)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('retrain_loss_curve.png', dpi=150)
        plt.close()
        print("Loss curve saved: retrain_loss_curve.png")

    print("\n" + "=" * 65)
    print("  RETRAINING COMPLETE")
    print(f"  New checkpoint: {CHECKPOINT_PATH}")
    print(f"  Old checkpoint: {BACKUP_PATH}")
    print(f"  Training log:   {LOG_PATH}")
    print()
    print("  Next step: Run  python calibrate_threshold.py  to update")
    print("  the confidence threshold based on the new model scores.")
    print("=" * 65)


if __name__ == '__main__':
    main()

