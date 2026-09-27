import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

CLASS_MAPPING = {
    0: 'D00 (Longitudinal Crack)',
    1: 'D10 (Transverse Crack)',
    2: 'D20 (Alligator Crack)',
    3: 'D40 (Pothole)',
    4: 'D43/D44 (Other Damage)'
}

def apply_clahe(img_pil):
    img_np = np.array(img_pil)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_np)

class RoadDamageCropDataset(Dataset):
    """
    Extracts individual damage region crops from YOLO bounding boxes with 10% contextual padding border.
    Fast indexing using os.scandir and set lookups.
    """
    def __init__(self, root_dir='.', split='train', img_size=224, augment=False, max_samples=None, pad_ratio=0.10):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.pad_ratio = pad_ratio
        
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.lbl_dir = os.path.join(root_dir, split, 'labels')
        
        self.crop_records = []
        self._index_crops(max_samples)
        
        # PyTorch ImageNet Normalization
        self.norm_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _index_crops(self, max_samples=None):
        if not os.path.exists(self.lbl_dir) or not os.path.exists(self.img_dir):
            return
            
        img_files_set = set(os.listdir(self.img_dir))
        
        with os.scandir(self.lbl_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.txt'):
                    base_name = os.path.splitext(entry.name)[0]
                    img_name = base_name + '.jpg'
                    if img_name not in img_files_set:
                        img_name = base_name + '.png'
                        if img_name not in img_files_set:
                            continue
                            
                    img_path = os.path.join(self.img_dir, img_name)
                    
                    try:
                        with open(entry.path, 'r', encoding='utf-8') as f:
                            for box_idx, line in enumerate(f):
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    cls_id = int(parts[0])
                                    if 0 <= cls_id <= 4:
                                        xc, yc, w, h = map(float, parts[1:5])
                                        self.crop_records.append({
                                            'img_path': img_path,
                                            'cls_id': cls_id,
                                            'bbox': (xc, yc, w, h),
                                            'crop_id': f"{base_name}_{box_idx}"
                                        })
                                        if max_samples and len(self.crop_records) >= max_samples * 2:
                                            break
                    except Exception:
                        continue
                        
                if max_samples and len(self.crop_records) >= max_samples * 2:
                    break
                    
        if max_samples and len(self.crop_records) > max_samples:
            self.crop_records = self.crop_records[:max_samples]

    def __len__(self):
        return len(self.crop_records)
        
    def __getitem__(self, idx):
        record = self.crop_records[idx]
        img_path = record['img_path']
        cls_id = record['cls_id']
        xc, yc, w, h = record['bbox']
        
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        
        pad_w = w * self.pad_ratio
        pad_h = h * self.pad_ratio
        
        xmin = max(0, int((xc - w / 2.0 - pad_w) * img_w))
        ymin = max(0, int((yc - h / 2.0 - pad_h) * img_h))
        xmax = min(img_w, int((xc + w / 2.0 + pad_w) * img_w))
        ymax = min(img_h, int((yc + h / 2.0 + pad_h) * img_h))
        
        if xmax > xmin and ymax > ymin:
            crop = img.crop((xmin, ymin, xmax, ymax))
        else:
            crop = img
            
        crop = apply_clahe(crop)
        crop = crop.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        if self.augment:
            if torch.rand(1).item() > 0.5:
                crop = TF.hflip(crop)
            if torch.rand(1).item() > 0.5:
                color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                crop = color_jitter(crop)
                
        tensor_img = self.norm_transform(crop)
        return tensor_img, torch.tensor(cls_id, dtype=torch.long)

def get_crop_dataloader(root_dir='.', split='train', batch_size=32, shuffle=True, augment=True, max_samples=None):
    dataset = RoadDamageCropDataset(root_dir=root_dir, split=split, augment=augment, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)
    return loader

if __name__ == '__main__':
    import time
    t0 = time.time()
    ds = RoadDamageCropDataset(root_dir='.', split='train', max_samples=100)
    print(f"Indexed {len(ds)} damage crops in {time.time() - t0:.2f}s!")
