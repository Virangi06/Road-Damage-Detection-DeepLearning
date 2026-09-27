import torch
import torch.nn as nn
from models.cnn_model import CNNBaselineModel
from models.vit_model import ViTBaselineModel

class HybridCNNViTModel(nn.Module):
    """
    Hybrid CNN + Vision Transformer (Main Proposed Architecture).
    Extracts local visual features using CNN (EfficientNet-B0) and global contextual features
    using Vision Transformer (ViT). Concatenates feature vectors and feeds them into a
    BatchNorm / Dropout / Dense Classification Fusion Head.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(HybridCNNViTModel, self).__init__()
        
        self.cnn_stream = CNNBaselineModel(num_classes=num_classes, pretrained=pretrained)
        self.vit_stream = ViTBaselineModel(num_classes=num_classes, pretrained=pretrained)
        
        # Dimensions: CNN (1280) + ViT (192) = 1472
        cnn_feat_dim = 1280
        vit_feat_dim = self.vit_stream.num_features
        fusion_in_dim = cnn_feat_dim + vit_feat_dim
        
        # Feature Fusion Classification Head
        self.fusion_head = nn.Sequential(
            nn.BatchNorm1d(fusion_in_dim),
            nn.Dropout(p=0.3),
            nn.Linear(fusion_in_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        f_cnn = self.cnn_stream.extract_features(x)
        f_vit = self.vit_stream.extract_features(x)
        
        # Concatenate local + global features
        f_fused = torch.cat([f_cnn, f_vit], dim=1)
        
        logits = self.fusion_head(f_fused)
        return logits

if __name__ == '__main__':
    print("Testing HybridCNNViTModel Architecture...")
    model = HybridCNNViTModel(num_classes=5, pretrained=False)
    dummy_x = torch.randn(2, 3, 224, 224)
    out = model(dummy_x)
    print(f"Hybrid Output Logits Shape: {out.shape}")
    print("Hybrid CNN+ViT test successful!")
