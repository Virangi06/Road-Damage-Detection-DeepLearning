import os
import sys
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

# Set style for presentation graphics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif', 'figure.dpi': 300})

CLASS_MAPPING = {
    '0': 'D00 (Longitudinal Crack)',
    '1': 'D10 (Transverse Crack)',
    '2': 'D20 (Alligator Crack)',
    '3': 'D40 (Pothole)',
    '4': 'D43/D44 (Other Damage)'
}

CLASS_COLORS = {
    '0': '#e63946',  # Red
    '1': '#f4a261',  # Orange
    '2': '#2a9d8f',  # Teal
    '3': '#457b9d',  # Blue
    '4': '#9d4edd'   # Purple
}

def analyze_and_plot():
    print("--- STARTING PHASE 1: COMPREHENSIVE EDA & DATASET VALIDATION ---", flush=True)
    
    splits = ['train', 'val', 'test']
    stats = {}
    spatial_centers = []
    box_aspect_ratios = []

    for split in splits:
        print(f"Processing split: {split}...", flush=True)
        img_dir = os.path.join(split, 'images')
        lbl_dir = os.path.join(split, 'labels')
        
        img_files = os.listdir(img_dir) if os.path.exists(img_dir) else []
        lbl_files = os.listdir(lbl_dir) if os.path.exists(lbl_dir) else []
        
        class_counts = Counter()
        country_counts = Counter()
        total_boxes = 0
        empty_labels = 0
        box_sizes = {'small (<32x32)': 0, 'medium (32x32-96x96)': 0, 'large (>96x96)': 0}
        
        if os.path.exists(lbl_dir):
            with os.scandir(lbl_dir) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith('.txt'):
                        lbl_name = entry.name
                        country = lbl_name.split('_')[0]
                        country_counts[country] += 1
                        
                        try:
                            with open(entry.path, 'r', encoding='utf-8') as f:
                                lines = [line.strip() for line in f if line.strip()]
                                if not lines:
                                    empty_labels += 1
                                for line in lines:
                                    parts = line.split()
                                    if len(parts) >= 5:
                                        cls_id = parts[0]
                                        class_counts[cls_id] += 1
                                        total_boxes += 1
                                        
                                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                        abs_w, abs_h = w * 512, h * 512
                                        abs_area = abs_w * abs_h
                                        
                                        if len(spatial_centers) < 15000:
                                            spatial_centers.append((xc, yc))
                                            if abs_h > 0:
                                                box_aspect_ratios.append(abs_w / abs_h)
                                                
                                        if abs_area < 32 * 32:
                                            box_sizes['small (<32x32)'] += 1
                                        elif abs_area < 96 * 96:
                                            box_sizes['medium (32x32-96x96)'] += 1
                                        else:
                                            box_sizes['large (>96x96)'] += 1
                        except Exception as e:
                            print(f"Error reading {entry.path}: {e}", flush=True)
                            
        stats[split] = {
            'num_images': len(img_files),
            'num_labels': len(lbl_files),
            'empty_labels': empty_labels,
            'total_boxes': total_boxes,
            'class_counts': {CLASS_MAPPING.get(k, f'Class {k}'): v for k, v in class_counts.items()},
            'country_counts': dict(country_counts),
            'box_sizes': box_sizes
        }

    # Save stats to json
    with open('phase1_eda_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
        
    print("Dataset Stats JSON saved to phase1_eda_stats.json.", flush=True)
    
    # 1. Class Distribution Bar Chart
    plt.figure(figsize=(10, 6))
    train_cls = stats['train']['class_counts']
    val_cls = stats['val']['class_counts']
    test_cls = stats['test']['class_counts']
    
    all_cls_keys = list(CLASS_MAPPING.values())
    x = np.arange(len(all_cls_keys))
    width = 0.25
    
    train_vals = [train_cls.get(k, 0) for k in all_cls_keys]
    val_vals = [val_cls.get(k, 0) for k in all_cls_keys]
    test_vals = [test_cls.get(k, 0) for k in all_cls_keys]
    
    plt.bar(x - width, train_vals, width, label='Train', color='#1d3557')
    plt.bar(x, val_vals, width, label='Val', color='#457b9d')
    plt.bar(x + width, test_vals, width, label='Test', color='#a8dadc')
    
    plt.ylabel('Count of Bounding Boxes', fontsize=12, fontweight='bold')
    plt.title('Road Damage Class Distribution Across Splits', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, [k.split(' ')[0] + '\n' + ' '.join(k.split(' ')[1:]) for k in all_cls_keys], fontsize=10)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('phase1_class_distribution.png', dpi=300)
    plt.close()
    
    # 2. Country Source Distribution
    plt.figure(figsize=(8, 5))
    countries = stats['train']['country_counts']
    sns.barplot(x=list(countries.keys()), y=list(countries.values()), palette='viridis')
    plt.title('Image Count by Country Source (Train Dataset)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
    plt.xlabel('Country / Source Split', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('phase1_country_distribution.png', dpi=300)
    plt.close()

    # 3. Spatial Heatmap of Damage Bounding Box Centers
    plt.figure(figsize=(7, 7))
    xcs = [c[0] for c in spatial_centers]
    ycs = [c[1] for c in spatial_centers]
    plt.hexbin(xcs, ycs, gridsize=40, cmap='YlOrRd', mincnt=1)
    plt.colorbar(label='Damage Instance Density')
    plt.title('Spatial Heatmap of Damage Bounding Box Centers (Normalized)', fontsize=13, fontweight='bold')
    plt.xlabel('X Center (Normalized 0.0 - 1.0)', fontsize=11)
    plt.ylabel('Y Center (Normalized 0.0 - 1.0)', fontsize=11)
    plt.gca().invert_yaxis()  # Invert Y to match image coordinate system
    plt.tight_layout()
    plt.savefig('phase1_spatial_heatmap.png', dpi=300)
    plt.close()

    # 4. Box Size Categorization (COCO Standards)
    plt.figure(figsize=(8, 5))
    sizes = stats['train']['box_sizes']
    plt.pie(sizes.values(), labels=sizes.keys(), autopct='%1.1f%%', colors=['#e76f51', '#f4a261', '#2a9d8f'], startangle=140, explode=(0.05, 0, 0))
    plt.title('Object Size Distribution (COCO Standard)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('phase1_box_size_distribution.png', dpi=300)
    plt.close()

    # 5. Generate Annotated Sample Image Grid
    print("Generating Annotated Sample Visualizations...", flush=True)
    sample_lbl_files = [f for f in os.listdir('train/labels') if os.path.getsize(os.path.join('train/labels', f)) > 0][:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    for idx, lbl_file in enumerate(sample_lbl_files):
        img_file = lbl_file.replace('.txt', '.jpg')
        img_path = os.path.join('train/images', img_file)
        lbl_path = os.path.join('train/labels', lbl_file)
        
        if not os.path.exists(img_path):
            continue
            
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        w_img, h_img = img.size
        
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c_id, xc, yc, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    xmin = (xc - w/2) * w_img
                    ymin = (yc - h/2) * h_img
                    xmax = (xc + w/2) * w_img
                    ymax = (yc + h/2) * h_img
                    
                    color = CLASS_COLORS.get(c_id, '#ff0000')
                    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)
                    draw.text((xmin + 5, ymin + 5), CLASS_MAPPING.get(c_id, c_id).split(' ')[0], fill=color)
                    
        ax = axes[idx // 2, idx % 2]
        ax.imshow(img)
        ax.set_title(f"Sample Ground Truth: {lbl_file}", fontsize=10, fontweight='bold')
        ax.axis('off')
        
    plt.suptitle("Sample Annotated Road Damage Images (Phase 1 Ground Truth)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('phase1_sample_annotated_images.png', dpi=300)
    plt.close()

    print("--- PHASE 1 EDA COMPLETED SUCCESSFULLY! ---", flush=True)

if __name__ == '__main__':
    analyze_and_plot()
