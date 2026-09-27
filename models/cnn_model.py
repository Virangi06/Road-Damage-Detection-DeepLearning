import torch
import torch.nn as nn
import torchvision.models as models

class CNNBaselineModel(nn.Module):
    """
    CNN Baseline Classifier for Road Damage Classification.
    Uses pretrained EfficientNet-B0 backbone fine-tuned for 5 damage classes.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(CNNBaselineModel, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Extract in_features from default classifier linear layer
        in_features = self.backbone.classifier[1].in_features
        
        # Custom Classification Head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        """
        Extracts 1280-dim spatial feature vector before classification head.
        Used by the Hybrid CNN+ViT fusion model.
        """
        feat = self.backbone.features(x)
        feat = self.backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return feat

if __name__ == '__main__':
    print("Testing CNNBaselineModel Architecture...")
    model = CNNBaselineModel(num_classes=5, pretrained=False)
    dummy_x = torch.randn(2, 3, 224, 224)
    out = model(dummy_x)
    feats = model.extract_features(dummy_x)
    print(f"Output Logits Shape: {out.shape}")
    print(f"Extracted Features Shape: {feats.shape}")
    print("CNN Baseline test successful!")
