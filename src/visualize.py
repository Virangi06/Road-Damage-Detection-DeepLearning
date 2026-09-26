import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from src.dataset import RoadDamageDataset, collate_fn, CLASS_NAMES
from src.model import build_model

CLASS_COLORS = {
    1: '#e63946',  # D00 - Red
    2: '#f4a261',  # D10 - Orange
    3: '#2a9d8f',  # D20 - Teal
    4: '#457b9d',  # D40 - Blue
    5: '#9d4edd'   # D43/D44 - Purple
}

def visualize_predictions(model_path='models/best_model.pth', root_dir='.', split='val', num_samples=6, score_thresh=0.3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- GENERATING QUALITATIVE PREDICTION VISUALIZATIONS ON SPLIT: {split} ---", flush=True)
    
    dataset = RoadDamageDataset(root_dir=root_dir, split=split, augment=False)
    data_loader = DataLoader(dataset, batch_size=num_samples, shuffle=True, collate_fn=collate_fn)
    
    model = build_model(num_classes=6, pretrained=False)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {model_path}", flush=True)
    else:
        print(f"Warning: {model_path} not found. Running visualization with default model.", flush=True)
        
    model.to(device)
    model.eval()
    
    # Get a batch of samples
    images, targets = next(iter(data_loader))
    images_gpu = [img.to(device) for img in images]
    
    with torch.no_grad():
        predictions = model(images_gpu)
        
    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx in range(num_samples):
        if idx >= len(images):
            break
            
        img_tensor = images[idx]
        target = targets[idx]
        pred = predictions[idx]
        
        # Convert tensor to PIL Image [512, 512, 3]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)
        
        # Draw Ground Truth Boxes (Green outlines)
        gt_boxes = target['boxes'].numpy()
        gt_labels = target['labels'].numpy()
        for box, lbl in zip(gt_boxes, gt_labels):
            xmin, ymin, xmax, ymax = box
            draw.rectangle([xmin, ymin, xmax, ymax], outline='#2b9348', width=3)
            cls_name = CLASS_NAMES.get(int(lbl), f"Cls {lbl}").split(' ')[0]
            draw.text((xmin + 4, ymin + 4), f"GT: {cls_name}", fill='#2b9348')
            
        # Draw Predicted Boxes (Bright color matching class)
        p_boxes = pred['boxes'].cpu().numpy()
        p_labels = pred['labels'].cpu().numpy()
        p_scores = pred['scores'].cpu().numpy()
        
        for box, lbl, score in zip(p_boxes, p_labels, p_scores):
            if score >= score_thresh:
                xmin, ymin, xmax, ymax = box
                color = CLASS_COLORS.get(int(lbl), '#ff0055')
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)
                cls_name = CLASS_NAMES.get(int(lbl), f"Cls {lbl}").split(' ')[0]
                text = f"Pred: {cls_name} ({score:.2f})"
                draw.text((xmin + 4, ymax - 18 if ymax - 18 > ymin else ymin + 18), text, fill=color)
                
        ax = axes[idx]
        ax.imshow(img_pil)
        ax.set_title(f"Sample #{idx+1} | GT: Green, Pred: Colored", fontsize=11, fontweight='bold')
        ax.axis('off')
        
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
        
    plt.suptitle("Phase 2 Road Damage Model Qualitative Predictions (GT vs Prediction)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('phase2_predictions_visualization.png', dpi=300)
    plt.close()
    
    print("Saved qualitative prediction visualization to phase2_predictions_visualization.png", flush=True)

if __name__ == '__main__':
    visualize_predictions()
