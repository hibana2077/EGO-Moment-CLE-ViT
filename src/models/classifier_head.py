"""
Classifier Head - Fusion of CLS token and moment features for classification

This module combines CLS token features from CLE-ViT backbone with 
high-order moment features from the moment head for final classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassifierHead(nn.Module):
    """
    Classifier head that fuses CLS token features with moment features
    
    Takes global features (CLS token or GAP from backbone) and moment features
    from graph-weighted pooling, then produces classification logits.
    
    Args:
        d_cls: Dimension of CLS/global features from backbone
        d_moment: Dimension of moment features from moment head
        num_classes: Number of output classes
        hidden_dim: Hidden dimension for fusion network
        dropout: Dropout rate
        fusion_type: Type of feature fusion ('concat', 'add', 'bilinear')
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        d_cls: int,
        d_moment: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        fusion_type: str = 'concat',
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.d_cls = d_cls
        self.d_moment = d_moment
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.use_batch_norm = use_batch_norm
        
        # Determine fusion dimension
        if fusion_type == 'concat':
            fusion_dim = d_cls + d_moment
        elif fusion_type == 'add':
            # Requires same dimensions or projection
            if d_cls != d_moment:
                self.cls_proj = nn.Linear(d_cls, d_moment)
                self.moment_proj = nn.Linear(d_moment, d_moment)
                fusion_dim = d_moment
            else:
                fusion_dim = d_cls
        elif fusion_type == 'bilinear':
            # Bilinear fusion
            fusion_dim = d_cls * d_moment
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Set hidden dimension
        if hidden_dim is None:
            hidden_dim = max(fusion_dim // 2, 256)
        
        # Feature fusion layers
        if fusion_type == 'bilinear':
            self.bilinear = nn.Bilinear(d_cls, d_moment, hidden_dim)
            
        # Classification network
        layers = []
        
        # Input layer
        if fusion_type == 'bilinear':
            input_dim = hidden_dim
        else:
            input_dim = fusion_dim
            
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim // 2))
            
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim // 2, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def _fuse_features(self, cls_features: torch.Tensor, moment_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse CLS and moment features according to fusion type
        
        Args:
            cls_features: [B, d_cls] CLS token features
            moment_features: [B, d_moment] moment features
            
        Returns:
            fused_features: [B, fusion_dim] fused features
        """
        if self.fusion_type == 'concat':
            # Simple concatenation
            fused = torch.cat([cls_features, moment_features], dim=-1)
            
        elif self.fusion_type == 'add':
            # Element-wise addition (with optional projection)
            if hasattr(self, 'cls_proj'):
                cls_proj = self.cls_proj(cls_features)
                moment_proj = self.moment_proj(moment_features)
                fused = cls_proj + moment_proj
            else:
                fused = cls_features + moment_features
                
        elif self.fusion_type == 'bilinear':
            # Bilinear fusion
            fused = self.bilinear(cls_features, moment_features)
            
        return fused
    
    def forward(self, cls_features: torch.Tensor, moment_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of classifier head
        
        Args:
            cls_features: [B, d_cls] global features from backbone
            moment_features: [B, d_moment] moment features from moment head
            
        Returns:
            logits: [B, num_classes] classification logits
        """
        # Fuse features
        fused_features = self._fuse_features(cls_features, moment_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits


class MultiScaleClassifierHead(nn.Module):
    """
    Multi-scale classifier head with attention mechanism
    
    Processes CLS and moment features at multiple scales and uses
    attention to combine them for robust classification.
    """
    
    def __init__(
        self,
        d_cls: int,
        d_moment: int,
        num_classes: int,
        num_scales: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_cls = d_cls
        self.d_moment = d_moment
        self.num_classes = num_classes
        self.num_scales = num_scales
        
        # Multi-scale projections
        self.cls_projections = nn.ModuleList([
            nn.Linear(d_cls, d_cls // (2**i)) for i in range(num_scales)
        ])
        
        self.moment_projections = nn.ModuleList([
            nn.Linear(d_moment, d_moment // (2**i)) for i in range(num_scales) 
        ])
        
        # Scale dimensions
        scale_dims = [d_cls // (2**i) + d_moment // (2**i) for i in range(num_scales)]
        
        # Scale-specific classifiers
        self.scale_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(scale_dims[i], scale_dims[i] // 2),
                nn.BatchNorm1d(scale_dims[i] // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(scale_dims[i] // 2, num_classes)
            ) for i in range(num_scales)
        ])
        
        # Attention for scale fusion
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=num_classes,
            num_heads=1,
            batch_first=True
        )
        
    def forward(self, cls_features: torch.Tensor, moment_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale processing
        
        Args:
            cls_features: [B, d_cls] global features
            moment_features: [B, d_moment] moment features
            
        Returns:
            logits: [B, num_classes] final logits
        """
        batch_size = cls_features.shape[0]
        scale_logits = []
        
        # Process each scale
        for i in range(self.num_scales):
            cls_proj = self.cls_projections[i](cls_features)
            moment_proj = self.moment_projections[i](moment_features)
            
            # Concatenate and classify
            scale_input = torch.cat([cls_proj, moment_proj], dim=-1)
            scale_output = self.scale_classifiers[i](scale_input)
            scale_logits.append(scale_output)
        
        # Stack scale logits for attention
        scale_logits = torch.stack(scale_logits, dim=1)  # [B, num_scales, num_classes]
        
        # Apply attention across scales
        attended_logits, _ = self.scale_attention(scale_logits, scale_logits, scale_logits)
        
        # Average across scales
        final_logits = attended_logits.mean(dim=1)  # [B, num_classes]
        
        return final_logits


class AdaptiveClassifierHead(nn.Module):
    """
    Adaptive classifier head that learns feature importance dynamically
    
    Uses squeeze-and-excitation style attention to adaptively weight
    the contribution of CLS and moment features.
    """
    
    def __init__(
        self,
        d_cls: int,
        d_moment: int,
        num_classes: int,
        reduction_ratio: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_cls = d_cls
        self.d_moment = d_moment
        self.num_classes = num_classes
        
        fusion_dim = d_cls + d_moment
        
        # Squeeze-and-Excitation for adaptive weighting
        self.se_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(fusion_dim, fusion_dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim // reduction_ratio, fusion_dim),
            nn.Sigmoid()
        )
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.BatchNorm1d(fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 4, num_classes)
        )
        
    def forward(self, cls_features: torch.Tensor, moment_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive feature weighting
        
        Args:
            cls_features: [B, d_cls] global features  
            moment_features: [B, d_moment] moment features
            
        Returns:
            logits: [B, num_classes] classification logits
        """
        # Concatenate features
        fused_features = torch.cat([cls_features, moment_features], dim=-1)  # [B, d_cls + d_moment]
        
        # Adaptive weighting via SE
        weights = self.se_layer(fused_features.unsqueeze(-1)).squeeze(-1)  # [B, d_cls + d_moment]
        weighted_features = fused_features * weights
        
        # Classification
        logits = self.classifier(weighted_features)
        
        return logits


def test_classifier_head():
    """Test function for classifier heads"""
    print("Testing Classifier Heads...")
    
    # Test parameters
    batch_size = 4
    d_cls = 768
    d_moment = 1024
    num_classes = 100
    
    # Create dummy features
    cls_features = torch.randn(batch_size, d_cls)
    moment_features = torch.randn(batch_size, d_moment)
    
    print(f"CLS features shape: {cls_features.shape}")
    print(f"Moment features shape: {moment_features.shape}")
    
    # Test basic classifier head
    print("\n=== Testing Basic Classifier ===")
    
    for fusion_type in ['concat', 'add', 'bilinear']:
        classifier = ClassifierHead(
            d_cls=d_cls,
            d_moment=d_moment,
            num_classes=num_classes,
            fusion_type=fusion_type
        )
        
        with torch.no_grad():
            logits = classifier(cls_features, moment_features)
            print(f"{fusion_type.capitalize()} fusion - Logits shape: {logits.shape}")
    
    # Test multi-scale classifier
    print("\n=== Testing Multi-Scale Classifier ===")
    
    multiscale_classifier = MultiScaleClassifierHead(
        d_cls=d_cls,
        d_moment=d_moment,
        num_classes=num_classes,
        num_scales=3
    )
    
    with torch.no_grad():
        logits = multiscale_classifier(cls_features, moment_features)
        print(f"Multi-scale logits shape: {logits.shape}")
    
    # Test adaptive classifier
    print("\n=== Testing Adaptive Classifier ===")
    
    adaptive_classifier = AdaptiveClassifierHead(
        d_cls=d_cls,
        d_moment=d_moment,
        num_classes=num_classes
    )
    
    with torch.no_grad():
        logits = adaptive_classifier(cls_features, moment_features)
        print(f"Adaptive logits shape: {logits.shape}")
    
    print("Classifier Head tests completed!")


if __name__ == "__main__":
    test_classifier_head()
