import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

CLASS_NAMES = {
    0: 'Background',
    1: 'D00 (Longitudinal Crack)',
    2: 'D10 (Transverse Crack)',
    3: 'D20 (Alligator Crack)',
    4: 'D40 (Pothole)',
    5: 'D43/D44 (Other Damage)'
}

class RoadDamageDataset(Dataset):
    """
    PyTorch Dataset for Road Damage Detection (RDD).
    Loads images and YOLO formatted bounding box annotations.
    Maps YOLO class IDs (0..4) to PyTorch Detection IDs (1..5), reserving 0 for background.
    """
    def __init__(self, root_dir, split='train', img_size=512, augment=False):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.lbl_dir = os.path.join(root_dir, split, 'labels')
        
        self.image_filenames = sorted(os.listdir(self.img_dir)) if os.path.exists(self.img_dir) else []
        
    def __len__(self):
        return len(self.image_filenames)
        
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = img.size
        
        # Resize image to target img_size
        if (w_orig, h_orig) != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Load corresponding label file
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(self.lbl_dir, lbl_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Map YOLO 0..4 -> Faster R-CNN 1..5
                        cls_id = int(parts[0]) + 1
                        xc, yc, w, h = map(float, parts[1:5])
                        
                        # Convert normalized YOLO (xc, yc, w, h) to pixel corner coords
                        xmin = max(0.0, (xc - w / 2.0) * self.img_size)
                        ymin = max(0.0, (yc - h / 2.0) * self.img_size)
                        xmax = min(float(self.img_size), (xc + w / 2.0) * self.img_size)
                        ymax = min(float(self.img_size), (yc + h / 2.0) * self.img_size)
                        
                        # Ensure box width and height are positive
                        if (xmax - xmin) > 1.0 and (ymax - ymin) > 1.0:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(cls_id)
                            
        # Optional Data Augmentations for training split
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                flipped_boxes = []
                for box in boxes:
                    # xmin_new = img_size - xmax, xmax_new = img_size - xmin
                    new_xmin = self.img_size - box[2]
                    new_xmax = self.img_size - box[0]
                    flipped_boxes.append([new_xmin, box[1], new_xmax, box[3]])
                boxes = flipped_boxes
            
            # Color jitter (brightness, contrast)
            if random.random() > 0.5:
                color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                img = color_jitter(img)

        # Convert image to Tensor [3, H, W] normalized to [0, 1]
        img_tensor = T.ToTensor()(img)
        
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            iscrowd_tensor = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)
            
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx]),
            'area': area_tensor,
            'iscrowd': iscrowd_tensor,
            'orig_size': torch.tensor([h_orig, w_orig])
        }
        
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloader(root_dir, split='train', batch_size=8, shuffle=True, num_workers=0, augment=False):
    dataset = RoadDamageDataset(root_dir, split=split, augment=augment)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return loader

if __name__ == '__main__':
    print("Testing RoadDamageDataset pipeline...")
    dataset = RoadDamageDataset(root_dir='.', split='train', augment=True)
    print(f"Total training samples: {len(dataset)}")
    sample_img, sample_target = dataset[0]
    print(f"Sample image shape: {sample_img.shape}")
    print(f"Sample target keys: {list(sample_target.keys())}")
    print(f"Sample target boxes count: {len(sample_target['boxes'])}")
    print(f"Sample target labels: {sample_target['labels']}")
    print("Dataset verification successful!")

