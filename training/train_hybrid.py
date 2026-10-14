import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from data.crop_dataset import get_crop_dataloader
from models.hybrid_model import HybridCNNViTModel
from training.common_trainer import set_seed

def train_hybrid(epochs=5, batch_size=32, lr=0.0003, max_train_samples=1500, max_val_samples=300):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== TRAINING HYBRID CNN+ViT (Balanced Multi-Country, Samples={max_train_samples}) ===", flush=True)
    
    train_loader = get_crop_dataloader(split='train', batch_size=batch_size, augment=True, max_samples=max_train_samples)
    val_loader = get_crop_dataloader(split='val', batch_size=batch_size, augment=False, max_samples=max_val_samples)
    
    model = HybridCNNViTModel(num_classes=5, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # STAGE 1: Train fusion head with frozen backbones for 2 epochs to prevent catastrophic forgetting
    print("\n--- STAGE 1: Feature Fusion Head Alignment (Frozen Backbones) ---", flush=True)
    for p in model.cnn_stream.parameters():
        p.requires_grad = False
    for p in model.vit_stream.parameters():
        p.requires_grad = False
        
    optimizer_stage1 = torch.optim.AdamW(model.fusion_head.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for ep in range(1, 3):
        model.train()
        t_loss, correct, total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer_stage1.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer_stage1.step()
            t_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == lbls).sum().item()
            total += imgs.size(0)
            
        # Val
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                v_loss += loss.item() * imgs.size(0)
                v_corr += (outputs.argmax(1) == lbls).sum().item()
                v_tot += imgs.size(0)
                
        print(f"[Stage 1] Epoch {ep}/2 - Train Loss: {t_loss/total:.4f}, Train Acc: {correct/total*100:.2f}% | Val Loss: {v_loss/v_tot:.4f}, Val Acc: {v_corr/v_tot*100:.2f}%", flush=True)

    # STAGE 2: End-to-End Fine-Tuning with Differential Learning Rates
    print("\n--- STAGE 2: Full Network Fine-Tuning (Differential LR) ---", flush=True)
    for p in model.cnn_stream.parameters():
        p.requires_grad = True
    for p in model.vit_stream.parameters():
        p.requires_grad = True
        
    params = [
        {'params': model.cnn_stream.parameters(), 'lr': 3e-5},
        {'params': model.vit_stream.parameters(), 'lr': 3e-5},
        {'params': model.fusion_head.parameters(), 'lr': 3e-4}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 2)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = os.path.join('checkpoints', 'best_hybrid_cnn_vit.pth')
    
    history = {
        'model_name': 'hybrid_cnn_vit',
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    for ep in range(1, epochs - 1):
        model.train()
        t_loss, correct, total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == lbls).sum().item()
            total += imgs.size(0)
            
        scheduler.step()
        
        # Val
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                v_loss += loss.item() * imgs.size(0)
                v_corr += (outputs.argmax(1) == lbls).sum().item()
                v_tot += imgs.size(0)
                
        t_acc = correct / total * 100.0
        v_acc = v_corr / v_tot * 100.0
        avg_t_loss = t_loss / total
        avg_v_loss = v_loss / v_tot
        
        history['epoch'].append(ep + 2)
        history['train_loss'].append(round(avg_t_loss, 4))
        history['val_loss'].append(round(avg_v_loss, 4))
        history['train_acc'].append(round(t_acc, 2))
        history['val_acc'].append(round(v_acc, 2))
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"[Stage 2] Epoch {ep}/{epochs-2} - Train Loss: {avg_t_loss:.4f}, Train Acc: {t_acc:.2f}% | Val Loss: {avg_v_loss:.4f}, Val Acc: {v_acc:.2f}%", flush=True)
        
        if avg_v_loss < best_val_loss or v_acc > best_val_acc:
            if avg_v_loss < best_val_loss:
                best_val_loss = avg_v_loss
            if v_acc > best_val_acc:
                best_val_acc = v_acc
            torch.save({
                'epoch': ep + 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_v_loss,
                'val_acc': v_acc
            }, ckpt_path)
            print(f"--> Saved Best Hybrid CNN+ViT Checkpoint to {ckpt_path} (Val Loss: {avg_v_loss:.4f}, Val Acc: {v_acc:.2f}%)", flush=True)
            
    print(f"\nTraining Complete! Best Val Acc: {best_val_acc:.2f}%, Best Val Loss: {best_val_loss:.4f}", flush=True)
    return history

if __name__ == '__main__':
    train_hybrid()
