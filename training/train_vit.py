import os
from data.crop_dataset import get_crop_dataloader
from models.vit_model import ViTBaselineModel
from training.common_trainer import ModelTrainer, set_seed

def train_vit(epochs=3, batch_size=16, lr=0.0003, max_train_samples=300, max_val_samples=100):
    set_seed(42)
    train_loader = get_crop_dataloader(split='train', batch_size=batch_size, augment=True, max_samples=max_train_samples)
    val_loader = get_crop_dataloader(split='val', batch_size=batch_size, augment=False, max_samples=max_val_samples)
    
    model = ViTBaselineModel(num_classes=5, pretrained=True)
    trainer = ModelTrainer(model, model_name='vit_baseline', lr=lr)
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    return history

if __name__ == '__main__':
    train_vit()
