"""
EGO-Moment-CLE-ViT - Main model integrating all components

This module combines:
1. CLE-ViT backbone for dual-view feature extraction
2. Graph Polynomial Fusion (GPF) for relation graph construction  
3. Moment Head for high-order statistical pooling
4. Classifier Head for final classification

The model maintains CLE-ViT's dual-view training while adding graph-based
high-order moment features for enhanced ultra-fine-grained classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import warnings

from .cle_vit_backbone import CLEViTDualStream
from .gpf_kernel import GraphPolynomialFusion
from .moment_head import MomentHead
from .classifier_head import ClassifierHead


class EGOMomentCLEViT(nn.Module):
    """
    EGO-Moment-CLE-ViT: Integration of graph polynomial fusion and moment pooling with CLE-ViT
    
    This model extends CLE-ViT with:
    - Graph Polynomial Fusion (EGO-like) for dual-view relation graphs
    - Graph-weighted high-order moment pooling (2nd + 3rd order)
    - Feature fusion for enhanced ultra-fine-grained classification
    
    Args:
        num_classes: Number of classification classes
        backbone_name: timm model name for backbone
        pretrained: Whether to use pretrained backbone weights
        gpf_degree_p: Polynomial degree for anchor view in GPF
        gpf_degree_q: Polynomial degree for positive view in GPF
        moment_d_out: Output dimension of moment head
        use_third_order: Whether to include third-order moments
        classifier_fusion: Feature fusion type in classifier
        lambda_triplet: Weight for triplet loss
        lambda_align: Weight for graph alignment loss
        margin: Margin for triplet loss
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = 'swin_base_patch4_window7_224',
        pretrained: bool = True,
        gpf_degree_p: int = 2,
        gpf_degree_q: int = 2,
        gpf_similarity: str = 'cosine',
        moment_d_out: int = 1024,
        use_third_order: bool = True,
        isqrt_iterations: int = 5,
        sketch_dim: int = 4096,
        classifier_fusion: str = 'concat',
        classifier_hidden: Optional[int] = None,
        lambda_triplet: float = 1.0,
        lambda_align: float = 0.1,
        margin: float = 0.3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.lambda_triplet = lambda_triplet
        self.lambda_align = lambda_align
        self.margin = margin
        
        # 1. CLE-ViT Backbone (dual-stream)
        self.backbone = CLEViTDualStream(
            model_name=backbone_name,
            pretrained=pretrained,
            drop_rate=dropout
        )
        
        backbone_dim = self.backbone.num_features
        
        # 2. Graph Polynomial Fusion
        self.gpf = GraphPolynomialFusion(
            degree_p=gpf_degree_p,
            degree_q=gpf_degree_q,
            similarity=gpf_similarity,
            symmetric_enforce=True
        )
        
        # 3. Moment Head
        self.moment_head = MomentHead(
            d_in=backbone_dim,
            d_out=moment_d_out,
            use_third_order=use_third_order,
            isqrt_iterations=isqrt_iterations,
            sketch_dim=sketch_dim
        )
        
        # 4. Classifier Head
        self.classifier = ClassifierHead(
            d_cls=backbone_dim,
            d_moment=moment_d_out,
            num_classes=num_classes,
            hidden_dim=classifier_hidden,
            dropout=dropout,
            fusion_type=classifier_fusion
        )
        
        # 5. Standard classifier for CLE-ViT baseline (for ablation)
        self.cls_only_classifier = nn.Linear(backbone_dim, num_classes)
        
        print(f"Created EGO-Moment-CLE-ViT:")
        print(f"  - Backbone: {backbone_name} ({backbone_dim} features)")
        print(f"  - GPF: degrees ({gpf_degree_p}, {gpf_degree_q}), similarity: {gpf_similarity}")
        print(f"  - Moments: {moment_d_out}D, third-order: {use_third_order}")
        print(f"  - Classifier: {classifier_fusion} fusion")
        print(f"  - Classes: {num_classes}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of EGO-Moment-CLE-ViT
        
        Args:
            anchor: [B, 3, H, W] anchor view images
            positive: [B, 3, H, W] positive view images (masked & shuffled)
            labels: [B] ground truth labels (optional, for loss computation)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
                - logits: [B, num_classes] main classification logits
                - logits_anchor: [B, num_classes] anchor-only logits (for CLE-ViT loss)
                - logits_positive: [B, num_classes] positive-only logits (for CLE-ViT loss)
                - loss_dict: Dictionary of losses (if labels provided)
                - features: Intermediate features (if return_features=True)
        """
        batch_size = anchor.shape[0]
        
        # 1. Extract features from both views
        anchor_features, positive_features = self.backbone(anchor, positive)
        
        anchor_tokens = anchor_features['patch_tokens']  # [B, N, D]
        positive_tokens = positive_features['patch_tokens']  # [B, N, D]
        anchor_global = anchor_features['global_features']  # [B, D]
        positive_global = positive_features['global_features']  # [B, D]
        
        # 2. Graph Polynomial Fusion
        fused_graph = self.gpf(anchor_tokens, positive_tokens)  # [B, N, N]
        
        # 3. Moment pooling (using anchor tokens with fused graph)
        moment_features = self.moment_head(anchor_tokens, fused_graph)  # [B, d_moment]
        
        # 4. Main classification (CLS + moments)
        main_logits = self.classifier(anchor_global, moment_features)  # [B, num_classes]
        
        # 5. CLE-ViT style individual view classification
        anchor_logits = self.cls_only_classifier(anchor_global)  # [B, num_classes]
        positive_logits = self.cls_only_classifier(positive_global)  # [B, num_classes]
        
        # Prepare output
        output = {
            'logits': main_logits,
            'logits_anchor': anchor_logits,
            'logits_positive': positive_logits,
        }
        
        # Compute losses if labels provided
        if labels is not None:
            loss_dict = self._compute_losses(
                main_logits=main_logits,
                anchor_logits=anchor_logits,
                positive_logits=positive_logits,
                anchor_global=anchor_global,
                positive_global=positive_global,
                fused_graph=fused_graph,
                labels=labels
            )
            output['loss_dict'] = loss_dict
            output['loss'] = sum(loss_dict.values())
        
        # Return intermediate features if requested
        if return_features:
            output['features'] = {
                'anchor_tokens': anchor_tokens,
                'positive_tokens': positive_tokens,
                'anchor_global': anchor_global,
                'positive_global': positive_global,
                'fused_graph': fused_graph,
                'moment_features': moment_features,
                'gpf_coefficients': self.gpf.get_coefficient_matrix()
            }
        
        return output
    
    def _compute_losses(
        self,
        main_logits: torch.Tensor,
        anchor_logits: torch.Tensor,
        positive_logits: torch.Tensor,
        anchor_global: torch.Tensor,
        positive_global: torch.Tensor,
        fused_graph: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components
        
        Returns:
            Dictionary of losses:
                - loss_main_ce: Main classification loss
                - loss_anchor_ce: Anchor view classification loss
                - loss_positive_ce: Positive view classification loss  
                - loss_triplet: Instance-level triplet loss
                - loss_align: Graph alignment loss (optional)
        """
        loss_dict = {}
        
        # 1. Classification losses (Cross-Entropy)
        loss_dict['loss_main_ce'] = F.cross_entropy(main_logits, labels)
        loss_dict['loss_anchor_ce'] = F.cross_entropy(anchor_logits, labels)
        loss_dict['loss_positive_ce'] = F.cross_entropy(positive_logits, labels)
        
        # 2. Instance-level triplet loss (CLE-ViT style)
        # Use anchor as query, positive as positive, and shifted anchor as negative
        negative_global = anchor_global.roll(shifts=1, dims=0)  # Simple negative sampling
        loss_dict['loss_triplet'] = self.lambda_triplet * self._triplet_loss(
            anchor_global, positive_global, negative_global, margin=self.margin
        )
        
        # 3. Graph alignment loss (optional)
        if self.lambda_align > 0:
            loss_dict['loss_align'] = self.lambda_align * self._graph_alignment_loss(
                fused_graph, labels
            )
        
        return loss_dict
    
    def _triplet_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        margin: float
    ) -> torch.Tensor:
        """
        Compute triplet loss with L2 normalization
        
        Args:
            anchor: [B, D] anchor features
            positive: [B, D] positive features
            negative: [B, D] negative features
            margin: Triplet margin
            
        Returns:
            triplet_loss: Scalar loss
        """
        # L2 normalize features
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Compute distances
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        
        # Triplet loss
        loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        return loss.mean()
    
    def _graph_alignment_loss(
        self,
        fused_graph: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute graph alignment loss to encourage same-class high similarity
        
        Args:
            fused_graph: [B, N, N] fused relation graphs
            labels: [B] class labels
            
        Returns:
            alignment_loss: Scalar loss
        """
        batch_size = fused_graph.shape[0]
        
        # Create label similarity matrix (same class = 1, different class = 0)
        label_sim = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [B, B]
        
        # Global similarity from fused graphs (average over spatial dimensions)
        # Take diagonal as self-similarity and off-diagonal as cross-similarity measures
        graph_global_sim = fused_graph.mean(dim=(1, 2))  # [B] - mean similarity per sample
        
        # Compute pairwise global similarities between samples
        graph_sim_matrix = torch.zeros_like(label_sim)
        for i in range(batch_size):
            for j in range(batch_size):
                # Use some measure of graph similarity between samples i and j
                # For simplicity, use dot product of their global similarities
                graph_sim_matrix[i, j] = graph_global_sim[i] * graph_global_sim[j]
        
        # Normalize similarities to [0, 1]
        graph_sim_matrix = torch.sigmoid(graph_sim_matrix)
        
        # Alignment loss: encourage graph similarity to match label similarity
        alignment_loss = F.mse_loss(graph_sim_matrix, label_sim)
        
        return alignment_loss
    
    def inference(self, images: torch.Tensor) -> torch.Tensor:
        """
        Inference mode - single view input
        
        Args:
            images: [B, 3, H, W] input images
            
        Returns:
            logits: [B, num_classes] classification logits
        """
        # In inference, use same image for both views (no positive augmentation)
        with torch.no_grad():
            output = self.forward(images, images, labels=None, return_features=False)
            return output['logits']


def test_ego_moment_clevit():
    """Test function for EGO-Moment-CLE-ViT"""
    print("Testing EGO-Moment-CLE-ViT...")
    
    # Test parameters
    batch_size = 2
    num_classes = 10
    input_size = 224  # Smaller for testing
    
    # Create model
    model = EGOMomentCLEViT(
        num_classes=num_classes,
        backbone_name='swin_tiny_patch4_window7_224',  # Smaller model for testing
        pretrained=False,
        gpf_degree_p=2,
        gpf_degree_q=2,
        moment_d_out=512,  # Smaller for testing
        use_third_order=True,
        sketch_dim=1024
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data
    anchor = torch.randn(batch_size, 3, input_size, input_size)
    positive = torch.randn(batch_size, 3, input_size, input_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"Input shapes: anchor {anchor.shape}, positive {positive.shape}")
    print(f"Labels: {labels}")
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    
    with torch.no_grad():
        output = model(anchor, positive, labels, return_features=True)
        
        print(f"Main logits shape: {output['logits'].shape}")
        print(f"Anchor logits shape: {output['logits_anchor'].shape}")
        print(f"Positive logits shape: {output['logits_positive'].shape}")
        print(f"Total loss: {output['loss'].item():.4f}")
        
        # Print loss components
        print("\nLoss components:")
        for loss_name, loss_value in output['loss_dict'].items():
            print(f"  {loss_name}: {loss_value.item():.4f}")
        
        # Print feature shapes
        print("\nFeature shapes:")
        features = output['features']
        for feat_name, feat_value in features.items():
            if isinstance(feat_value, torch.Tensor):
                print(f"  {feat_name}: {feat_value.shape}")
            else:
                print(f"  {feat_name}: {feat_value}")
    
    # Test inference mode
    print("\n=== Testing Inference Mode ===")
    
    with torch.no_grad():
        inference_logits = model.inference(anchor)
        print(f"Inference logits shape: {inference_logits.shape}")
    
    # Test backward pass
    print("\n=== Testing Backward Pass ===")
    
    model.train()
    output = model(anchor, positive, labels)
    loss = output['loss']
    
    loss.backward()
    print(f"Backward pass successful. Loss: {loss.item():.4f}")
    
    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"Parameters with gradients: {has_grad}/{total_params}")
    
    print("EGO-Moment-CLE-ViT test completed!")


if __name__ == "__main__":
    test_ego_moment_clevit()
