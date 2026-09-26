import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

def build_model(num_classes=6, pretrained=True):
    """
    Builds Faster R-CNN with ResNet-50 FPN backbone.
    num_classes: 6 (5 damage classes + 1 background class)
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    # Replace the classification head with new head matching num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

if __name__ == '__main__':
    print("Testing Model Architecture Construction...")
    model = build_model(num_classes=6, pretrained=False)
    model.eval()
    
    # Test dummy forward pass
    dummy_x = [torch.rand(3, 512, 512)]
    with torch.no_grad():
        predictions = model(dummy_x)
        
    print("Dummy Forward Pass Successful!")
    print(f"Prediction Keys: {list(predictions[0].keys())}")
