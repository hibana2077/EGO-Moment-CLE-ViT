"""
Triplet Loss for CLE-ViT style instance-level contrastive learning

This module implements the triplet loss used in CLE-ViT for instance-level
contrastive learning between anchor, positive, and negative views.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TripletLoss(nn.Module):
    """
    Triplet loss for instance-level contrastive learning
    
    Implements the triplet loss used in CLE-ViT:
    L = max(||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin, 0)
    
    Where:
    - a: anchor view features
    - p: positive view features (same instance, different augmentation)
    - n: negative view features (different instance)
    
    Args:
        margin: Margin for triplet loss
        p_norm: Norm to use for distance calculation (2 for L2)
        normalize: Whether to L2 normalize features before computing distance
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        p_norm: int = 2,
        normalize: bool = True,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.margin = margin
        self.p_norm = p_norm
        self.normalize = normalize
        self.reduction = reduction
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor: [B, D] anchor features
            positive: [B, D] positive features  
            negative: [B, D] negative features
            
        Returns:
            loss: Triplet loss value
        """
        if self.normalize:
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)
        
        # Compute distances
        pos_dist = torch.norm(anchor - positive, p=self.p_norm, dim=1)
        neg_dist = torch.norm(anchor - negative, p=self.p_norm, dim=1)
        
        # Triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HardTripletLoss(nn.Module):
    """
    Hard triplet loss with online hard negative mining
    
    Automatically finds the hardest negative examples within each batch
    for more effective training.
    
    Args:
        margin: Margin for triplet loss
        normalize: Whether to L2 normalize features
        hard_positive: Whether to also mine hard positives
        reduction: Reduction method
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        normalize: bool = True,
        hard_positive: bool = False,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
        self.hard_positive = hard_positive
        self.reduction = reduction
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hard triplet loss with mining
        
        Args:
            embeddings: [B, D] feature embeddings
            labels: [B] class labels
            
        Returns:
            loss: Hard triplet loss
        """
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # Create masks for positive and negative pairs
        labels = labels.unsqueeze(0)
        pos_mask = labels == labels.t()
        neg_mask = labels != labels.t()
        
        # Remove diagonal (self-distances)
        pos_mask.fill_diagonal_(False)
        
        batch_size = embeddings.shape[0]
        triplet_losses = []
        
        for i in range(batch_size):
            # Get positive and negative distances for anchor i
            pos_dists = dist_matrix[i][pos_mask[i]]
            neg_dists = dist_matrix[i][neg_mask[i]]
            
            if len(pos_dists) == 0 or len(neg_dists) == 0:
                continue
                
            # Hard positive: furthest positive
            if self.hard_positive:
                hardest_pos_dist = pos_dists.max()
            else:
                hardest_pos_dist = pos_dists.mean()
                
            # Hard negative: closest negative
            hardest_neg_dist = neg_dists.min()
            
            # Triplet loss
            loss = torch.clamp(hardest_pos_dist - hardest_neg_dist + self.margin, min=0.0)
            triplet_losses.append(loss)
        
        if len(triplet_losses) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        triplet_losses = torch.stack(triplet_losses)
        
        if self.reduction == 'mean':
            return triplet_losses.mean()
        elif self.reduction == 'sum':
            return triplet_losses.sum()
        else:
            return triplet_losses


class MultiViewTripletLoss(nn.Module):
    """
    Multi-view triplet loss for CLE-ViT with multiple positive views
    
    Extends basic triplet loss to handle multiple positive views (e.g., different
    augmentations of the same image) and multiple negative sampling strategies.
    
    Args:
        margin: Margin for triplet loss
        normalize: Whether to normalize features
        num_positives: Number of positive views per anchor
        negative_sampling: Strategy for negative sampling ('random', 'hard', 'semi-hard')
        temperature: Temperature for soft triplet variants
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        normalize: bool = True,
        num_positives: int = 1,
        negative_sampling: str = 'random',
        temperature: float = 0.1
    ):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
        self.num_positives = num_positives
        self.negative_sampling = negative_sampling
        self.temperature = temperature
        
    def _sample_negatives(
        self,
        anchor_idx: int,
        dist_matrix: torch.Tensor,
        labels: torch.Tensor,
        strategy: str = 'random'
    ) -> torch.Tensor:
        """Sample negative examples for given anchor"""
        neg_mask = labels != labels[anchor_idx]
        neg_indices = torch.where(neg_mask)[0]
        
        if len(neg_indices) == 0:
            return torch.tensor([], device=dist_matrix.device)
            
        neg_dists = dist_matrix[anchor_idx][neg_indices]
        
        if strategy == 'random':
            # Random negative
            idx = torch.randint(0, len(neg_indices), (1,))
            return neg_dists[idx]
        elif strategy == 'hard':
            # Hardest (closest) negative
            return neg_dists.min().unsqueeze(0)
        elif strategy == 'semi-hard':
            # Semi-hard negatives (closer than furthest positive)
            pos_mask = (labels == labels[anchor_idx]) & (torch.arange(len(labels)) != anchor_idx)
            if pos_mask.any():
                furthest_pos = dist_matrix[anchor_idx][pos_mask].max()
                semi_hard_mask = (neg_dists > furthest_pos - self.margin) & (neg_dists < furthest_pos)
                if semi_hard_mask.any():
                    return neg_dists[semi_hard_mask].min().unsqueeze(0)
            # Fall back to hardest if no semi-hard found
            return neg_dists.min().unsqueeze(0)
        else:
            raise ValueError(f"Unknown negative sampling strategy: {strategy}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-view triplet loss
        
        Args:
            anchor: [B, D] anchor features
            positive: [B, D] or [B, num_positives, D] positive features
            labels: [B] labels
            
        Returns:
            loss: Multi-view triplet loss
        """
        if self.normalize:
            anchor = F.normalize(anchor, p=2, dim=1)
            if positive.dim() == 3:
                positive = F.normalize(positive, p=2, dim=-1)
            else:
                positive = F.normalize(positive, p=2, dim=1)
        
        batch_size = anchor.shape[0]
        
        # Handle multiple positives
        if positive.dim() == 3:
            # [B, num_positives, D] -> compute loss for each positive
            losses = []
            for p_idx in range(positive.shape[1]):
                pos_view = positive[:, p_idx]  # [B, D]
                
                # Compute distance matrix
                all_features = torch.cat([anchor, pos_view], dim=0)
                all_labels = torch.cat([labels, labels], dim=0)
                dist_matrix = torch.cdist(all_features, all_features, p=2)
                
                # Compute triplet loss
                for i in range(batch_size):
                    pos_dist = dist_matrix[i, batch_size + i]  # Distance to corresponding positive
                    
                    # Sample negative
                    neg_dists = self._sample_negatives(i, dist_matrix, all_labels, self.negative_sampling)
                    if len(neg_dists) > 0:
                        neg_dist = neg_dists[0]
                        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
                        losses.append(loss)
            
            if len(losses) > 0:
                return torch.stack(losses).mean()
            else:
                return torch.tensor(0.0, device=anchor.device, requires_grad=True)
        else:
            # Standard case: [B, D] positive
            triplet_loss = TripletLoss(
                margin=self.margin,
                normalize=False,  # Already normalized
                reduction='mean'
            )
            
            # Simple negative sampling by shifting
            if self.negative_sampling == 'random':
                negative = anchor[torch.randperm(batch_size)]
            else:
                # Use distance-based sampling
                dist_matrix = torch.cdist(anchor, anchor, p=2)
                losses = []
                for i in range(batch_size):
                    neg_dists = self._sample_negatives(i, dist_matrix, labels, self.negative_sampling)
                    if len(neg_dists) > 0:
                        pos_dist = torch.norm(anchor[i] - positive[i], p=2)
                        neg_dist = neg_dists[0]
                        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
                        losses.append(loss)
                
                if len(losses) > 0:
                    return torch.stack(losses).mean()
                else:
                    return torch.tensor(0.0, device=anchor.device, requires_grad=True)
            
            negative = anchor[torch.randperm(batch_size)]
            return triplet_loss(anchor, positive, negative)


def test_triplet_losses():
    """Test triplet loss implementations"""
    print("Testing Triplet Losses...")
    
    # Test data
    batch_size = 8
    feature_dim = 128
    num_classes = 4
    
    anchor = torch.randn(batch_size, feature_dim)
    positive = torch.randn(batch_size, feature_dim)
    negative = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"Test data shapes:")
    print(f"  Anchor: {anchor.shape}")
    print(f"  Positive: {positive.shape}")
    print(f"  Negative: {negative.shape}")
    print(f"  Labels: {labels.shape}")
    
    # Test basic triplet loss
    print("\n=== Basic Triplet Loss ===")
    triplet_loss = TripletLoss(margin=1.0, normalize=True)
    loss = triplet_loss(anchor, positive, negative)
    print(f"Triplet loss: {loss.item():.4f}")
    
    # Test hard triplet loss
    print("\n=== Hard Triplet Loss ===")
    hard_triplet = HardTripletLoss(margin=1.0, normalize=True)
    
    # Combine anchor and positive as embeddings
    embeddings = torch.cat([anchor, positive], dim=0)
    combined_labels = torch.cat([labels, labels], dim=0)
    
    hard_loss = hard_triplet(embeddings, combined_labels)
    print(f"Hard triplet loss: {hard_loss.item():.4f}")
    
    # Test multi-view triplet loss
    print("\n=== Multi-View Triplet Loss ===")
    
    # Test with multiple positives
    num_positives = 3
    multi_positive = torch.randn(batch_size, num_positives, feature_dim)
    
    multi_triplet = MultiViewTripletLoss(
        margin=1.0,
        normalize=True,
        num_positives=num_positives,
        negative_sampling='hard'
    )
    
    multi_loss = multi_triplet(anchor, multi_positive, labels)
    print(f"Multi-view triplet loss: {multi_loss.item():.4f}")
    
    # Test backward pass
    print("\n=== Testing Backward Pass ===")
    loss.backward()
    print("Backward pass successful!")
    
    print("Triplet loss tests completed!")


if __name__ == "__main__":
    test_triplet_losses()
