"""
Models module for EGO-Moment-CLE-ViT

This module contains all model components:
- CLE-ViT backbone with dual-view processing
- Graph Polynomial Fusion (GPF) for relation graph construction  
- Moment Head for high-order statistical pooling
- Classifier Head for feature fusion and classification
- Main EGO-Moment-CLE-ViT integration model
"""

from .cle_vit_backbone import (
    CLEViTBackbone,
    CLEViTDualStream,
    CLEViTDataTransforms,
    PositiveViewAugmentation
)

from .gpf_kernel import (
    GraphPolynomialFusion,
    AdaptiveGraphPolynomialFusion
)

from .moment_head import (
    MomentHead,
    NewtonSchulzSqrtm,
    TensorSketch
)

from .classifier_head import (
    ClassifierHead,
    MultiScaleClassifierHead,
    AdaptiveClassifierHead
)

from .ego_moment_clevit import (
    EGOMomentCLEViT
)

__all__ = [
    # Backbone
    'CLEViTBackbone',
    'CLEViTDualStream', 
    'CLEViTDataTransforms',
    'PositiveViewAugmentation',
    
    # Graph Polynomial Fusion
    'GraphPolynomialFusion',
    'AdaptiveGraphPolynomialFusion',
    
    # Moment pooling
    'MomentHead',
    'NewtonSchulzSqrtm',
    'TensorSketch',
    
    # Classification
    'ClassifierHead',
    'MultiScaleClassifierHead',
    'AdaptiveClassifierHead',
    
    # Main model
    'EGOMomentCLEViT',
]
