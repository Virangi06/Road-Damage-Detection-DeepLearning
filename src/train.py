import os
import sys
import time
import json
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import RoadDamageDataset, collate_fn
from src.model import build_model

os.makedirs('models', exist_ok=True)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    loss_components = {'cls': 0.0, 'box': 0.0, 'obj': 0.0, 'rpn_box': 0.0}
    num_batches = 0
    
    start_time = time.time()
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Faster R-CNN loss dictionary
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        loss_components['cls'] += loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
        loss_components['box'] += loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
        loss_components['obj'] += loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
        loss_components['rpn_box'] += loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()
        
        num_batches += 1
        
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(data_loader):
            print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(data_loader)}] - Batch Loss: {losses.item():.4f} (Avg: {total_loss/num_batches:.4f})", flush=True)
            
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, num_batches)
    avg_components = {k: v / max(1, num_batches) for k, v in loss_components.items()}
    
    print(f"Epoch [{epoch}] Finished in {elapsed:.1f}s | Avg Train Loss: {avg_loss:.4f} (Cls: {avg_components['cls']:.4f}, Box: {avg_components['box']:.4f})", flush=True)
    return avg_loss, avg_components

def validate(model, data_loader, device):
    val_loss = 0.0
    num_batches = 0
    
    # Switch to train mode inside no_grad to compute validation loss dictionary
    model.train()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
            num_batches += 1
            
    avg_val_loss = val_loss / max(1, num_batches)
    model.eval()
    return avg_val_loss

def run_training(epochs=3, batch_size=4, lr=0.0003, sample_train_limit=100, sample_val_limit=30, pretrained=True, augment=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"Configured PyTorch CPU threads: {num_threads}", flush=True)
        
    print(f"--- STARTING PHASE 2 MODEL TRAINING ON DEVICE: {device} ---", flush=True)
    
    train_dataset = RoadDamageDataset(root_dir='.', split='train', augment=augment)
    val_dataset = RoadDamageDataset(root_dir='.', split='val', augment=False)

    
    if sample_train_limit and sample_train_limit < len(train_dataset):
        train_dataset.image_filenames = train_dataset.image_filenames[:sample_train_limit]
    if sample_val_limit and sample_val_limit < len(val_dataset):
        val_dataset.image_filenames = val_dataset.image_filenames[:sample_val_limit]
        
    print(f"Dataset active subset sizes -> Train: {len(train_dataset)} | Val: {len(val_dataset)}", flush=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = build_model(num_classes=6, pretrained=pretrained)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': [], 'loss_cls': [], 'loss_box': []}
    
    for epoch in range(1, epochs + 1):
        t_loss, t_comp = train_one_epoch(model, optimizer, train_loader, device, epoch)
        v_loss = validate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()
        
        history['epoch'].append(epoch)
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['lr'].append(current_lr)
        history['loss_cls'].append(t_comp['cls'])
        history['loss_box'].append(t_comp['box'])
        
        print(f"==> Epoch [{epoch}/{epochs}] Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | LR: {current_lr:.6f}", flush=True)
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            checkpoint_path = os.path.join('models', 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': v_loss
            }, checkpoint_path)
            print(f"--> Saved New Best Checkpoint to {checkpoint_path} (Val Loss: {v_loss:.4f})", flush=True)
            
    # Save final model state
    torch.save(model.state_dict(), os.path.join('models', 'final_model.pth'))
    
    # Save history json
    with open('phase2_training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
        
    # Plot Training & Validation Loss Curves
    plt.figure(figsize=(9, 5))
    plt.plot(history['epoch'], history['train_loss'], label='Training Loss', color='#e63946', linewidth=2.5, marker='o')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss', color='#457b9d', linewidth=2.5, marker='s')
    plt.title('Phase 2 Faster R-CNN Training & Validation Loss Convergence', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('phase2_training_loss_curve.png', dpi=300)
    plt.close()
    
    print("--- PHASE 2 MODEL TRAINING COMPLETED SUCCESSFULLY! ---", flush=True)
    return history

if __name__ == '__main__':
    run_training(epochs=5, batch_size=8, lr=0.0003, sample_train_limit=600, sample_val_limit=200)


