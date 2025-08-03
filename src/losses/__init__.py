"""
Losses module for EGO-Moment-CLE-ViT

This module contains loss functions used in the model:
- Triplet losses for instance-level contrastive learning
- Kernel alignment losses for graph regularization
"""

from .triplet_loss import (
    TripletLoss,
    HardTripletLoss,
    MultiViewTripletLoss
)

from .kernel_alignment import (
    KernelAlignmentLoss,
    ContrastiveAlignmentLoss,
    HierarchicalAlignmentLoss
)

__all__ = [
    # Triplet losses
    'TripletLoss',
    'HardTripletLoss', 
    'MultiViewTripletLoss',
    
    # Alignment losses
    'KernelAlignmentLoss',
    'ContrastiveAlignmentLoss',
    'HierarchicalAlignmentLoss',
]
