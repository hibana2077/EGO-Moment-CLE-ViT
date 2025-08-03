"""
Kernel Alignment Loss for Graph Regularization

This module implements kernel alignment loss to encourage the learned
relation graph to align with label-based similarity structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class KernelAlignmentLoss(nn.Module):
    """
    Kernel alignment loss for graph regularization
    
    Encourages the learned relation graph G to align with target similarity
    structures based on class labels. This helps ensure that same-class
    samples have higher similarity in the learned graph.
    
    The alignment loss is computed as:
    L_align = 1 - <G, Y> / (||G||_F * ||Y||_F)
    
    Where:
    - G: Learned relation graph [B, N, N] or [B, B] 
    - Y: Target similarity matrix based on labels
    - <·,·>: Frobenius inner product
    - ||·||_F: Frobenius norm
    
    Args:
        alignment_type: Type of alignment ('centered', 'normalized', 'cosine')
        temperature: Temperature for softmax normalization
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        alignment_type: str = 'centered',
        temperature: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alignment_type = alignment_type
        self.temperature = temperature
        self.reduction = reduction
        
    def _create_label_similarity_matrix(
        self,
        labels: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Create similarity matrix from labels
        
        Args:
            labels: [B] class labels
            normalize: Whether to normalize the matrix
            
        Returns:
            label_sim: [B, B] label similarity matrix
        """
        batch_size = labels.shape[0]
        
        # Create binary similarity matrix: same class = 1, different class = 0
        labels_expanded = labels.unsqueeze(0)  # [1, B]
        label_sim = (labels_expanded == labels_expanded.t()).float()  # [B, B]
        
        if normalize:
            # Normalize to have unit Frobenius norm
            frobenius_norm = torch.norm(label_sim, p='fro')
            if frobenius_norm > 0:
                label_sim = label_sim / frobenius_norm
                
        return label_sim
    
    def _graph_to_global_similarity(
        self,
        graph: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert spatial relation graph to global sample similarity
        
        Args:
            graph: [B, N, N] spatial relation graphs
            
        Returns:
            global_sim: [B, B] global similarity matrix between samples
        """
        if graph.dim() == 2:
            # Already global similarity matrix
            return graph
        elif graph.dim() == 3:
            # Spatial relation graphs -> aggregate to global similarities
            batch_size = graph.shape[0]
            
            # Method 1: Average spatial similarities as global similarity measure
            global_similarities = graph.mean(dim=(1, 2))  # [B] - mean similarity per sample
            
            # Create pairwise similarity matrix using dot product
            global_similarities = global_similarities.unsqueeze(1)  # [B, 1]
            global_sim = torch.mm(global_similarities, global_similarities.t())  # [B, B]
            
            return global_sim
        else:
            raise ValueError(f"Unsupported graph dimension: {graph.dim()}")
    
    def _centered_kernel_alignment(
        self,
        K1: torch.Tensor,
        K2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute centered kernel alignment (CKA)
        
        Args:
            K1: [B, B] first kernel matrix
            K2: [B, B] second kernel matrix
            
        Returns:
            cka: Centered kernel alignment score
        """
        batch_size = K1.shape[0]
        
        # Center the kernel matrices
        H = torch.eye(batch_size, device=K1.device) - torch.ones(batch_size, batch_size, device=K1.device) / batch_size
        
        K1_centered = torch.mm(torch.mm(H, K1), H)
        K2_centered = torch.mm(torch.mm(H, K2), H)
        
        # Compute CKA
        numerator = torch.trace(torch.mm(K1_centered, K2_centered))
        denominator = torch.sqrt(torch.trace(torch.mm(K1_centered, K1_centered)) * 
                                torch.trace(torch.mm(K2_centered, K2_centered)))
        
        if denominator > 0:
            cka = numerator / denominator
        else:
            cka = torch.tensor(0.0, device=K1.device)
            
        return cka
    
    def forward(
        self,
        graph: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute kernel alignment loss
        
        Args:
            graph: [B, N, N] relation graphs or [B, B] global similarity
            labels: [B] class labels
            
        Returns:
            loss: Kernel alignment loss
        """
        # Convert graph to global similarity if needed
        graph_sim = self._graph_to_global_similarity(graph)
        
        # Create target similarity matrix
        label_sim = self._create_label_similarity_matrix(labels, normalize=True)
        
        if self.alignment_type == 'centered':
            # Centered Kernel Alignment (CKA)
            alignment = self._centered_kernel_alignment(graph_sim, label_sim)
            loss = 1.0 - alignment  # Minimize negative alignment
            
        elif self.alignment_type == 'normalized':
            # Normalized cross-correlation
            graph_norm = torch.norm(graph_sim, p='fro')
            label_norm = torch.norm(label_sim, p='fro')
            
            if graph_norm > 0 and label_norm > 0:
                graph_normalized = graph_sim / graph_norm
                label_normalized = label_sim / label_norm
                
                # Frobenius inner product
                alignment = torch.sum(graph_normalized * label_normalized)
                loss = 1.0 - alignment
            else:
                loss = torch.tensor(1.0, device=graph.device)
                
        elif self.alignment_type == 'cosine':
            # Cosine similarity between flattened matrices
            graph_flat = graph_sim.view(-1)
            label_flat = label_sim.view(-1)
            
            cosine_sim = F.cosine_similarity(graph_flat, label_flat, dim=0)
            loss = 1.0 - cosine_sim
            
        else:
            raise ValueError(f"Unknown alignment type: {self.alignment_type}")
        
        return loss


class ContrastiveAlignmentLoss(nn.Module):
    """
    Contrastive alignment loss for graph regularization
    
    Uses contrastive learning principles to encourage same-class samples
    to have high graph similarity and different-class samples to have low similarity.
    
    Args:
        temperature: Temperature for contrastive loss
        margin: Margin for contrastive pairs
        positive_weight: Weight for positive pairs
        negative_weight: Weight for negative pairs
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.5,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        
    def forward(
        self,
        graph: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive alignment loss
        
        Args:
            graph: [B, N, N] relation graphs
            labels: [B] class labels
            
        Returns:
            loss: Contrastive alignment loss
        """
        if graph.dim() == 3:
            # Convert to global similarities
            global_sim = graph.mean(dim=(1, 2))  # [B]
        else:
            global_sim = torch.diagonal(graph)  # [B]
            
        batch_size = labels.shape[0]
        loss = 0.0
        num_pairs = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                sim_ij = global_sim[i] * global_sim[j]  # Similarity measure
                
                if labels[i] == labels[j]:
                    # Positive pair: encourage high similarity
                    pos_loss = torch.clamp(self.margin - sim_ij, min=0.0)
                    loss += self.positive_weight * pos_loss
                else:
                    # Negative pair: encourage low similarity  
                    neg_loss = torch.clamp(sim_ij - (1.0 - self.margin), min=0.0)
                    loss += self.negative_weight * neg_loss
                    
                num_pairs += 1
        
        if num_pairs > 0:
            loss = loss / num_pairs
            
        return loss


class HierarchicalAlignmentLoss(nn.Module):
    """
    Hierarchical alignment loss for multi-scale graph regularization
    
    Applies alignment loss at multiple scales of the relation graph,
    from fine-grained spatial relationships to global similarities.
    
    Args:
        scales: List of spatial scales to consider
        scale_weights: Weights for each scale
        base_alignment: Base alignment loss module
    """
    
    def __init__(
        self,
        scales: Optional[list] = None,
        scale_weights: Optional[list] = None,
        base_alignment: Optional[nn.Module] = None
    ):
        super().__init__()
        
        if scales is None:
            scales = [1, 2, 4]  # Different pooling scales
        if scale_weights is None:
            scale_weights = [1.0] * len(scales)
        if base_alignment is None:
            base_alignment = KernelAlignmentLoss()
            
        self.scales = scales
        self.scale_weights = scale_weights
        self.base_alignment = base_alignment
        
    def _multiscale_pooling(
        self,
        graph: torch.Tensor,
        scale: int
    ) -> torch.Tensor:
        """
        Apply pooling to relation graph at given scale
        
        Args:
            graph: [B, N, N] relation graph (assuming N = H*W for spatial layout)
            scale: Pooling scale
            
        Returns:
            pooled_graph: [B, N', N'] pooled relation graph
        """
        if scale == 1:
            return graph
            
        batch_size, N, _ = graph.shape
        
        # Assume square spatial layout
        H = W = int(N ** 0.5)
        if H * W != N:
            # If not perfect square, just return original
            return graph
            
        # Reshape to spatial format
        graph_spatial = graph.view(batch_size, H, W, H, W)
        
        # Apply average pooling
        pooled_H, pooled_W = H // scale, W // scale
        
        # Simple pooling by taking every scale-th element
        pooled_graph = graph_spatial[::scale, ::scale, ::scale, ::scale]
        
        return pooled_graph.view(batch_size, pooled_H * pooled_W, pooled_H * pooled_W)
    
    def forward(
        self,
        graph: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hierarchical alignment loss
        
        Args:
            graph: [B, N, N] relation graph
            labels: [B] class labels
            
        Returns:
            loss: Weighted sum of multi-scale alignment losses
        """
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.scale_weights):
            pooled_graph = self._multiscale_pooling(graph, scale)
            scale_loss = self.base_alignment(pooled_graph, labels)
            total_loss += weight * scale_loss
            
        return total_loss


def test_alignment_losses():
    """Test alignment loss implementations"""
    print("Testing Alignment Losses...")
    
    # Test data
    batch_size = 6
    num_tokens = 16  # 4x4 spatial layout
    num_classes = 3
    
    # Create dummy relation graphs
    graphs = torch.randn(batch_size, num_tokens, num_tokens)
    graphs = torch.bmm(graphs, graphs.transpose(-2, -1))  # Make PSD
    graphs = graphs / graphs.norm(dim=(1, 2), keepdim=True)  # Normalize
    
    # Create labels with some structure
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    
    print(f"Graph shape: {graphs.shape}")
    print(f"Labels: {labels}")
    
    # Test kernel alignment loss
    print("\n=== Kernel Alignment Loss ===")
    
    for alignment_type in ['centered', 'normalized', 'cosine']:
        ka_loss = KernelAlignmentLoss(alignment_type=alignment_type)
        loss = ka_loss(graphs, labels)
        print(f"{alignment_type.capitalize()} alignment loss: {loss.item():.4f}")
    
    # Test contrastive alignment loss
    print("\n=== Contrastive Alignment Loss ===")
    
    contrastive_loss = ContrastiveAlignmentLoss(temperature=0.1, margin=0.5)
    loss = contrastive_loss(graphs, labels)
    print(f"Contrastive alignment loss: {loss.item():.4f}")
    
    # Test hierarchical alignment loss
    print("\n=== Hierarchical Alignment Loss ===")
    
    hierarchical_loss = HierarchicalAlignmentLoss(
        scales=[1, 2],
        scale_weights=[0.7, 0.3]
    )
    loss = hierarchical_loss(graphs, labels)
    print(f"Hierarchical alignment loss: {loss.item():.4f}")
    
    # Test backward pass
    print("\n=== Testing Backward Pass ===")
    loss.backward()
    print("Backward pass successful!")
    
    print("Alignment loss tests completed!")


if __name__ == "__main__":
    test_alignment_losses()
