import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class GradCAMExplainer:
    """
    Gradient-Weighted Class Activation Mapping (Grad-CAM) Explainer
    for CNN backbones (EfficientNet-B0) in CNN Baseline and Hybrid models.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Locate target layer automatically if not specified
        if target_layer is None:
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
                # CNN Baseline Model
                target_layer = model.backbone.features[-1]
            elif hasattr(model, 'cnn_stream') and hasattr(model.cnn_stream.backbone, 'features'):
                # Hybrid CNN+ViT Model
                target_layer = model.cnn_stream.backbone.features[-1]
            else:
                raise ValueError("Could not automatically locate CNN features layer for Grad-CAM.")
                
        self.target_layer = target_layer
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generates Grad-CAM heatmap array normalized in [0, 1].
        
        Args:
            input_tensor (Tensor): Input image tensor (1, 3, H, W)
            target_class (int, optional): Target class index. If None, uses max logit.
            
        Returns:
            np.ndarray: CAM heatmap array (H, W) in [0, 1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured by hooks.")
            
        gradients = self.gradients[0]      # (C, H_feat, W_feat)
        activations = self.activations[0]  # (C, H_feat, W_feat)
        
        # Global Average Pooling of Gradients across spatial dimensions
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True) # (C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0) # (H_feat, W_feat)
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam_np = cam.cpu().numpy()
        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)
            
        # Resize to input tensor spatial dimensions
        h_in, w_in = input_tensor.shape[2:]
        cam_resized = cv2.resize(cam_np, (w_in, h_in))
        return cam_resized, target_class

    def overlay_heatmap(self, original_img_rgb, cam_heatmap, alpha=0.5):
        """
        Overlays CAM heatmap onto original RGB numpy image (H, W, 3) [0..255].
        """
        h, w = original_img_rgb.shape[:2]
        heatmap_resized = cv2.resize(cam_heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = np.float32(color_heatmap) * alpha + np.float32(original_img_rgb) * (1.0 - alpha)
        overlay = np.uint8(np.clip(overlay, 0, 255))
        return overlay

    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()


class ViTAttentionExplainer:
    """
    Vision Transformer (ViT) Self-Attention Map Explainer
    Extracts patch attention maps from final block multi-head self-attention.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.attention_matrix = None
        self.handler = None
        
        # Locate ViT model
        if hasattr(model, 'vit'):
            vit_module = model.vit
        elif hasattr(model, 'vit_stream') and hasattr(model.vit_stream, 'vit'):
            vit_module = model.vit_stream.vit
        else:
            raise ValueError("Could not automatically locate ViT module.")
            
        self.vit_module = vit_module
        self._register_hook()

    def _register_hook(self):
        # Register hook on final block's attn module
        target_attn_block = self.vit_module.blocks[-1].attn
        
        def hook_fn(module, input, output):
            # Output of timm attention layer: (B, N, C)
            # To get raw attention weights:
            B, N, C = input[0].shape
            num_heads = module.num_heads
            head_dim = C // num_heads
            
            qkv = module.qkv(input[0]).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1) # (B, num_heads, N, N)
            self.attention_matrix = attn.detach()

        self.handler = target_attn_block.register_forward_hook(hook_fn)

    def generate_attention_map(self, input_tensor):
        """
        Generates ViT self-attention heatmap relative to CLS token across patches.
        
        Args:
            input_tensor (Tensor): Input image tensor (1, 3, 224, 224)
            
        Returns:
            np.ndarray: Attention map (224, 224) in [0, 1]
        """
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        if self.attention_matrix is None:
            raise RuntimeError("Attention matrix was not captured.")
            
        # Attention shape: (1, num_heads, N_tokens, N_tokens)
        attn = self.attention_matrix[0] # (num_heads, N, N)
        
        # Mean attention across heads from [CLS] token (index 0) to patch tokens (indices 1:)
        attn_cls = attn[:, 0, 1:].mean(dim=0) # (N_patches,)
        
        num_patches = attn_cls.shape[0]
        grid_size = int(np.sqrt(num_patches))
        
        if grid_size * grid_size != num_patches:
            # Fallback if patch shape isn't square
            grid_size = 14
            attn_cls = attn_cls[:196]
            
        attn_grid = attn_cls.reshape(grid_size, grid_size).cpu().numpy()
        
        # Normalize in [0, 1]
        if attn_grid.max() > attn_grid.min():
            attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
        else:
            attn_grid = np.zeros_like(attn_grid)
            
        h_in, w_in = input_tensor.shape[2:]
        attn_resized = cv2.resize(attn_grid, (w_in, h_in), interpolation=cv2.INTER_CUBIC)
        return attn_resized

    def overlay_heatmap(self, original_img_rgb, attn_heatmap, alpha=0.5):
        """
        Overlays ViT Attention heatmap onto original RGB numpy image.
        """
        h, w = original_img_rgb.shape[:2]
        heatmap_resized = cv2.resize(attn_heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = np.float32(color_heatmap) * alpha + np.float32(original_img_rgb) * (1.0 - alpha)
        overlay = np.uint8(np.clip(overlay, 0, 255))
        return overlay

    def remove_hook(self):
        if self.handler:
            self.handler.remove()


if __name__ == '__main__':
    print("Testing GradCAMExplainer & ViTAttentionExplainer...")
    from models.cnn_model import CNNBaselineModel
    from models.vit_model import ViTBaselineModel
    from models.hybrid_model import HybridCNNViTModel

    dummy_x = torch.randn(1, 3, 224, 224)
    dummy_rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # Test CNN Grad-CAM
    cnn = CNNBaselineModel(pretrained=False)
    gradcam = GradCAMExplainer(cnn)
    cam, target_cls = gradcam.generate_cam(dummy_x)
    cam_overlay = gradcam.overlay_heatmap(dummy_rgb, cam)
    print(f"CNN Grad-CAM shape: {cam.shape}, target_class: {target_cls}, overlay shape: {cam_overlay.shape}")
    gradcam.remove_hooks()

    # Test ViT Attention
    vit = ViTBaselineModel(pretrained=False)
    vit_explainer = ViTAttentionExplainer(vit)
    attn_map = vit_explainer.generate_attention_map(dummy_x)
    attn_overlay = vit_explainer.overlay_heatmap(dummy_rgb, attn_map)
    print(f"ViT Attention map shape: {attn_map.shape}, overlay shape: {attn_overlay.shape}")
    vit_explainer.remove_hook()

    # Test Hybrid CNN + ViT Explainers
    hybrid = HybridCNNViTModel(pretrained=False)
    hybrid_gradcam = GradCAMExplainer(hybrid)
    h_cam, _ = hybrid_gradcam.generate_cam(dummy_x)
    hybrid_vit = ViTAttentionExplainer(hybrid)
    h_attn = hybrid_vit.generate_attention_map(dummy_x)
    print(f"Hybrid Grad-CAM shape: {h_cam.shape}, Hybrid ViT Attention shape: {h_attn.shape}")
    hybrid_gradcam.remove_hooks()
    hybrid_vit.remove_hook()

    print("All Explainability tests passed successfully!")
