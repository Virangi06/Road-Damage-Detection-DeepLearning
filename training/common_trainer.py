import os
import time
import json
import torch
import torch.nn as nn
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class ModelTrainer:
    """
    Reusable PyTorch Trainer Engine supporting early stopping, CosineAnnealing LR scheduling,
    model checkpointing, reproducible random seeds, and loss/accuracy convergence logging.
    """
    def __init__(self, model, model_name='model', device=None, checkpoint_dir='checkpoints', lr=0.0003, weight_decay=0.0001):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
        epoch_loss = total_loss / max(1, total)
        epoch_acc = (correct / max(1, total)) * 100.0
        return epoch_loss, epoch_acc

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data).item()
                total += labels.size(0)
                
        val_loss = total_loss / max(1, total)
        val_acc = (correct / max(1, total)) * 100.0
        return val_loss, val_acc

    def fit(self, train_loader, val_loader, epochs=5, patience=5):
        print(f"--- STARTING TRAINING FOR {self.model_name.upper()} ON DEVICE: {self.device} ---", flush=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        history = {
            'model_name': self.model_name,
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
        
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            t_loss, t_acc = self.train_epoch(train_loader)
            v_loss, v_acc = self.evaluate(val_loader)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            scheduler.step()
            
            history['epoch'].append(epoch)
            history['train_loss'].append(round(t_loss, 4))
            history['val_loss'].append(round(v_loss, 4))
            history['train_acc'].append(round(t_acc, 2))
            history['val_acc'].append(round(v_acc, 2))
            history['lr'].append(current_lr)
            
            print(f"[{self.model_name}] Epoch [{epoch}/{epochs}] - Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.2f}% | LR: {current_lr:.6f}", flush=True)
            
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                ckpt_path = os.path.join(self.checkpoint_dir, f"best_{self.model_name}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': v_loss,
                    'val_acc': v_acc
                }, ckpt_path)
                print(f"--> Saved Best {self.model_name} Checkpoint to {ckpt_path} (Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.2f}%)", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered for {self.model_name} after {epoch} epochs.", flush=True)
                    break
                    
        elapsed = time.time() - start_time
        print(f"Finished {self.model_name} training in {elapsed:.1f}s!", flush=True)
        return history
