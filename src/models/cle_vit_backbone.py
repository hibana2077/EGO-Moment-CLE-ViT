"""
CLE-ViT Backbone - Wrapper for timm models with CLE-ViT data augmentation

This module implements the CLE-ViT backbone using timm models and includes
the dual-view generation (anchor + positive with masking & shuffling) as described
in the CLE-ViT paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from typing import Tuple, Optional, Dict, Any
import math


class PositiveViewAugmentation:
    """
    Positive view augmentation for CLE-ViT: masking + 4x4 grid shuffling
    
    This creates the "positive" view by:
    1. Applying same standard augmentation as anchor
    2. Random rectangular masking (Cutout-style)  
    3. 4x4 grid shuffling to prevent trivial reconstruction
    
    Args:
        mask_ratio: Tuple of (min, max) masking ratios
        grid_size: Grid size for shuffling (4x4 as in paper)
        mask_value: Value to fill masked regions (0 for black)
    """
    
    def __init__(
        self,
        mask_ratio: Tuple[float, float] = (0.15, 0.45),
        grid_size: int = 4,
        mask_value: float = 0.0
    ):
        self.mask_ratio = mask_ratio
        self.grid_size = grid_size
        self.mask_value = mask_value
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply positive view augmentation
        
        Args:
            img: PIL Image after standard augmentation
            
        Returns:
            augmented_img: PIL Image with masking and shuffling
        """
        # Convert to numpy for processing
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # 1) Random rectangular masking
        ratio = random.uniform(*self.mask_ratio)
        mask_h = int(h * math.sqrt(ratio))
        mask_w = int(w * math.sqrt(ratio))
        
        # Random position for mask
        y0 = random.randint(0, max(1, h - mask_h))
        x0 = random.randint(0, max(1, w - mask_w))
        
        # Apply mask
        img_masked = img_array.copy()
        img_masked[y0:y0+mask_h, x0:x0+mask_w] = self.mask_value
        
        # 2) Grid shuffling (4x4)
        s = self.grid_size
        grid_h, grid_w = h // s, w // s
        
        # Extract tiles
        tiles = []
        for i in range(s):
            for j in range(s):
                tile = img_masked[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w].copy()
                tiles.append(tile)
        
        # Shuffle tiles
        random.shuffle(tiles)
        
        # Reconstruct image
        img_shuffled = np.zeros_like(img_masked)
        tile_idx = 0
        for i in range(s):
            for j in range(s):
                img_shuffled[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w] = tiles[tile_idx]
                tile_idx += 1
        
        return Image.fromarray(img_shuffled)


class CLEViTDataTransforms:
    """
    Data transforms for CLE-ViT dual-view generation
    
    Creates anchor and positive views according to CLE-ViT specification:
    - Resize to 600x600, then random/center crop to 448x448
    - Standard augmentations for both views
    - Additional masking + shuffling for positive view
    """
    
    def __init__(
        self,
        input_size: int = 448,
        resize_size: int = 600,
        is_training: bool = True,
        mask_ratio: Tuple[float, float] = (0.15, 0.45)
    ):
        self.input_size = input_size
        self.resize_size = resize_size
        self.is_training = is_training
        
        # Base transforms (common for both views)
        if is_training:
            self.base_transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(10),
            ])
        else:
            self.base_transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(input_size),
            ])
        
        # Positive view augmentation
        self.positive_aug = PositiveViewAugmentation(mask_ratio=mask_ratio)
        
        # Final tensor conversion
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate dual views for CLE-ViT
        
        Args:
            img: Input PIL Image
            
        Returns:
            anchor_tensor: Anchor view tensor [3, H, W]
            positive_tensor: Positive view tensor [3, H, W]
        """
        # Anchor view: standard augmentation only
        anchor_img = self.base_transform(img)
        anchor_tensor = self.to_tensor(anchor_img)
        
        # Positive view: standard augmentation + masking + shuffling
        if self.is_training:
            positive_img_base = self.base_transform(img)
            positive_img = self.positive_aug(positive_img_base)
            positive_tensor = self.to_tensor(positive_img)
        else:
            # During inference, positive view is same as anchor
            positive_tensor = anchor_tensor.clone()
        
        return anchor_tensor, positive_tensor


class CLEViTBackbone(nn.Module):
    """
    CLE-ViT backbone using timm models
    
    Wraps timm vision transformers for feature extraction with CLE-ViT dual-view processing.
    Supports both Swin Transformer and ViT architectures.
    
    Args:
        model_name: timm model name (e.g., 'swin_base_patch4_window7_224')
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes (0 for feature extraction)
        global_pool: Global pooling method ('avg', 'max', '')
        drop_rate: Dropout rate
    """
    
    def __init__(
        self,
        model_name: str = 'swin_base_patch4_window7_224',
        pretrained: bool = True,
        num_classes: int = 0,  # 0 for feature extraction
        global_pool: str = '',  # We'll handle pooling ourselves
        drop_rate: float = 0.0
    ):
        super().__init__()
        
        self.model_name = model_name
        
        # Create timm model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            global_pool=global_pool,
            drop_rate=drop_rate
        )
        
        # Get model info
        self.num_features = self.backbone.num_features
        self.patch_size = getattr(self.backbone, 'patch_size', None)
        
        # Check if model has cls_token (ViT-style) or uses global average pooling (Swin-style)
        self.has_cls_token = hasattr(self.backbone, 'cls_token')
        
        print(f"Created {model_name} backbone:")
        print(f"  - Num features: {self.num_features}")
        print(f"  - Has CLS token: {self.has_cls_token}")
        print(f"  - Patch size: {self.patch_size}")
    
    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patch tokens and global features
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            patch_tokens: [B, N, D] patch token features
            global_features: [B, D] global features
        """
        # Forward through backbone to get features
        features = self.backbone.forward_features(x)
        
        if self.has_cls_token:
            # ViT-style: features = [B, N+1, D] where first token is CLS
            cls_token = features[:, 0]  # [B, D]
            patch_tokens = features[:, 1:]  # [B, N, D]
            global_features = cls_token
        else:
            # Swin-style: features = [B, H', W', D] or [B, N, D]
            if features.dim() == 4:
                # [B, H', W', D] -> [B, N, D]
                B, H, W, D = features.shape
                patch_tokens = features.view(B, H * W, D)
            else:
                # Already [B, N, D]
                patch_tokens = features
            
            # Global average pooling for global features
            global_features = patch_tokens.mean(dim=1)  # [B, D]
        
        return patch_tokens, global_features
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning both tokens and global features
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary containing:
                - patch_tokens: [B, N, D] patch features
                - global_features: [B, D] global features  
        """
        patch_tokens, global_features = self.forward_features(x)
        
        return {
            'patch_tokens': patch_tokens,
            'global_features': global_features
        }


class CLEViTDualStream(nn.Module):
    """
    Dual-stream CLE-ViT for anchor and positive views
    
    Processes both anchor and positive views through shared backbone
    and returns features for both streams.
    """
    
    def __init__(
        self,
        model_name: str = 'swin_base_patch4_window7_224',
        pretrained: bool = True,
        drop_rate: float = 0.0
    ):
        super().__init__()
        
        self.backbone = CLEViTBackbone(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate
        )
        
        self.num_features = self.backbone.num_features
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for dual views
        
        Args:
            anchor: [B, 3, H, W] anchor view
            positive: [B, 3, H, W] positive view
            
        Returns:
            anchor_features: Dictionary with anchor features
            positive_features: Dictionary with positive features
        """
        anchor_features = self.backbone(anchor)
        positive_features = self.backbone(positive)
        
        return anchor_features, positive_features


def test_clevit_backbone():
    """Test function for CLE-ViT backbone"""
    print("Testing CLE-ViT Backbone...")
    
    # Test data transforms
    print("\n=== Testing Data Transforms ===")
    
    # Create dummy PIL image
    dummy_img = Image.new('RGB', (600, 400), color='red')
    
    # Test transforms
    transforms = CLEViTDataTransforms(
        input_size=448,
        resize_size=600,
        is_training=True
    )
    
    anchor, positive = transforms(dummy_img)
    print(f"Anchor shape: {anchor.shape}")
    print(f"Positive shape: {positive.shape}")
    
    # Test backbone
    print("\n=== Testing Backbone ===")
    
    backbone = CLEViTBackbone(
        model_name='swin_tiny_patch4_window7_224',  # Use smaller model for testing
        pretrained=False
    )
    
    # Test single view
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 448, 448)
    
    with torch.no_grad():
        output = backbone(dummy_input)
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Patch tokens shape: {output['patch_tokens'].shape}")
        print(f"Global features shape: {output['global_features'].shape}")
    
    # Test dual stream
    print("\n=== Testing Dual Stream ===")
    
    dual_stream = CLEViTDualStream(
        model_name='swin_tiny_patch4_window7_224',
        pretrained=False
    )
    
    anchor_input = torch.randn(batch_size, 3, 448, 448)
    positive_input = torch.randn(batch_size, 3, 448, 448)
    
    with torch.no_grad():
        anchor_feat, positive_feat = dual_stream(anchor_input, positive_input)
        
        print(f"Anchor tokens shape: {anchor_feat['patch_tokens'].shape}")
        print(f"Positive tokens shape: {positive_feat['patch_tokens'].shape}")
        print(f"Anchor global shape: {anchor_feat['global_features'].shape}")
        print(f"Positive global shape: {positive_feat['global_features'].shape}")
    
    print("CLE-ViT Backbone test completed!")


if __name__ == "__main__":
    test_clevit_backbone()
