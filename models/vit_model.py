import torch
import torch.nn as nn
import timm

class ViTBaselineModel(nn.Module):
    """
    Vision Transformer (ViT) Baseline Classifier for Road Damage Classification.
    Uses pretrained vit_tiny_patch16_224 via timm.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(ViTBaselineModel, self).__init__()
        # Load vit_tiny_patch16_224 (lightweight vision transformer for fast convergence)
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, num_classes=num_classes)
        self.num_features = self.vit.head.in_features
        
    def forward(self, x):
        return self.vit(x)

    def extract_features(self, x):
        """
        Extracts 192-dim global context feature vector from ViT transformer backbone.
        Used by the Hybrid CNN+ViT fusion model.
        """
        feats = self.vit.forward_features(x)
        if hasattr(self.vit, 'forward_head'):
            # For newer timm ViT models
            feats = self.vit.forward_head(feats, pre_logits=True)
        else:
            # Fallback for classification token
            feats = feats[:, 0]
        return feats

if __name__ == '__main__':
    print("Testing ViTBaselineModel Architecture...")
    model = ViTBaselineModel(num_classes=5, pretrained=False)
    dummy_x = torch.randn(2, 3, 224, 224)
    out = model(dummy_x)
    feats = model.extract_features(dummy_x)
    print(f"Output Logits Shape: {out.shape}")
    print(f"Extracted ViT Features Shape: {feats.shape}")
    print("ViT Baseline test successful!")
