"""
Graph Polynomial Fusion (GPF) - EGO-like polynomial expansion of relation graphs

This module implements the Graph Polynomial Fusion mechanism based on EGO principles,
which combines dual-view token similarity graphs through learnable polynomial expansions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class GraphPolynomialFusion(nn.Module):
    """
    Graph Polynomial Fusion module that combines dual-view similarity graphs
    using learnable polynomial coefficients.
    
    Based on EGO paper: performs polynomial expansion G(A) = Σ A_pq * R_a^p ⊙ R_p^q
    where ⊙ denotes Hadamard (element-wise) product and ^p denotes element-wise power.
    
    Args:
        degree_p (int): Maximum degree for anchor view polynomial
        degree_q (int): Maximum degree for positive view polynomial  
        similarity (str): Similarity function - 'cosine' or 'dot'
        eps (float): Small constant for numerical stability
        symmetric_enforce (bool): Whether to enforce symmetric output
        coeff_init (str): Coefficient initialization method
    """
    
    def __init__(
        self,
        degree_p: int = 2,
        degree_q: int = 2, 
        similarity: str = 'cosine',
        eps: float = 1e-6,
        symmetric_enforce: bool = True,
        coeff_init: str = 'uniform'
    ):
        super().__init__()
        
        self.degree_p = degree_p
        self.degree_q = degree_q
        self.similarity = similarity
        self.eps = eps
        self.symmetric_enforce = symmetric_enforce
        
        # Total number of polynomial terms
        self.num_terms = (degree_p + 1) * (degree_q + 1)
        
        # Learnable coefficients - use softplus to ensure non-negativity
        # A_pq = softplus(alpha_pq) to maintain PSD property
        self.alpha_coeffs = nn.Parameter(torch.zeros(degree_p + 1, degree_q + 1))
        
        self._init_coefficients(coeff_init)
        
    def _init_coefficients(self, init_method: str):
        """Initialize polynomial coefficients"""
        if init_method == 'uniform':
            # Small positive values to start
            nn.init.uniform_(self.alpha_coeffs, 0.0, 0.1)
        elif init_method == 'xavier':
            nn.init.xavier_uniform_(self.alpha_coeffs)
        elif init_method == 'identity':
            # Emphasize identity terms (p=0, q=0) and (p=1, q=1)
            self.alpha_coeffs.data.fill_(0.01)
            if self.degree_p >= 0 and self.degree_q >= 0:
                self.alpha_coeffs.data[0, 0] = 0.5  # R_a^0 ⊙ R_p^0 = all ones
            if self.degree_p >= 1 and self.degree_q >= 1:
                self.alpha_coeffs.data[1, 1] = 0.5  # R_a^1 ⊙ R_p^1 = element-wise product
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
    
    def _compute_similarity(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise similarity matrix for tokens
        
        Args:
            tokens: [B, N, D] token features
            
        Returns:
            similarity: [B, N, N] pairwise similarity matrix
        """
        if self.similarity == 'cosine':
            # L2 normalize tokens
            tokens_norm = F.normalize(tokens, p=2, dim=-1, eps=self.eps)
            similarity = torch.bmm(tokens_norm, tokens_norm.transpose(-2, -1))
        elif self.similarity == 'dot':
            similarity = torch.bmm(tokens, tokens.transpose(-2, -1))
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity}")
            
        return similarity
    
    def _hadamard_power(self, matrix: torch.Tensor, power: int) -> torch.Tensor:
        """
        Compute element-wise power of matrix
        
        Args:
            matrix: [B, N, N] input matrix
            power: integer power
            
        Returns:
            result: [B, N, N] matrix with elements raised to power
        """
        if power == 0:
            # R^0 = matrix of ones (identity for Hadamard product)
            return torch.ones_like(matrix)
        elif power == 1:
            return matrix
        else:
            # For numerical stability, use: exp(power * log(abs(x) + eps)) * sign(x)^power
            # But since similarity matrices should be non-negative, we can use direct power
            return torch.pow(torch.clamp(matrix, min=0.0), power)
    
    def forward(self, tokens_anchor: torch.Tensor, tokens_positive: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Graph Polynomial Fusion
        
        Args:
            tokens_anchor: [B, N, D] anchor view tokens
            tokens_positive: [B, N, D] positive view tokens
            
        Returns:
            fused_graph: [B, N, N] fused relation graph
        """
        # Compute similarity matrices for both views
        R_a = self._compute_similarity(tokens_anchor)  # [B, N, N]
        R_p = self._compute_similarity(tokens_positive)  # [B, N, N] 
        
        # Get non-negative coefficients
        A_coeffs = F.softplus(self.alpha_coeffs)  # [P+1, Q+1]
        
        # Initialize fused graph
        B, N, _ = R_a.shape
        fused_graph = torch.zeros_like(R_a)
        
        # Compute polynomial expansion: G(A) = Σ A_pq * R_a^p ⊙ R_p^q
        for p in range(self.degree_p + 1):
            for q in range(self.degree_q + 1):
                coeff = A_coeffs[p, q]
                
                # Compute R_a^p and R_p^q  
                R_a_power = self._hadamard_power(R_a, p)  # [B, N, N]
                R_p_power = self._hadamard_power(R_p, q)  # [B, N, N]
                
                # Hadamard product and weighted sum
                term = coeff * (R_a_power * R_p_power)  # [B, N, N]
                fused_graph = fused_graph + term
        
        # Enforce symmetry to maintain PSD property
        if self.symmetric_enforce:
            fused_graph = 0.5 * (fused_graph + fused_graph.transpose(-2, -1))
        
        # Ensure non-negative values (clamp to maintain PSD)
        fused_graph = torch.clamp(fused_graph, min=0.0)
        
        return fused_graph
    
    def get_coefficient_matrix(self) -> torch.Tensor:
        """Get current coefficient matrix A_pq = softplus(alpha_pq)"""
        return F.softplus(self.alpha_coeffs)
    
    def get_sparsity_loss(self, lambda_sparse: float = 0.01) -> torch.Tensor:
        """
        Compute sparsity regularization on coefficients
        
        Args:
            lambda_sparse: sparsity regularization weight
            
        Returns:
            sparsity_loss: L1 penalty on coefficients
        """
        coeffs = F.softplus(self.alpha_coeffs)
        return lambda_sparse * torch.sum(torch.abs(coeffs))


class AdaptiveGraphPolynomialFusion(GraphPolynomialFusion):
    """
    Adaptive version of GPF that learns separate coefficients for each spatial position
    or uses attention-based coefficient modulation.
    """
    
    def __init__(
        self,
        degree_p: int = 2,
        degree_q: int = 2,
        similarity: str = 'cosine', 
        eps: float = 1e-6,
        symmetric_enforce: bool = True,
        coeff_init: str = 'uniform',
        adaptive_type: str = 'global'  # 'global', 'spatial', 'attention'
    ):
        super().__init__(degree_p, degree_q, similarity, eps, symmetric_enforce, coeff_init)
        
        self.adaptive_type = adaptive_type
        
        if adaptive_type == 'attention':
            # Attention-based coefficient modulation
            self.coeff_attention = nn.MultiheadAttention(
                embed_dim=self.num_terms,
                num_heads=1,
                batch_first=True
            )
        elif adaptive_type == 'spatial':
            # Spatial-specific coefficients (for each token position)
            # This would require knowing the spatial layout (e.g., 14x14 for ViT)
            pass
    
    def forward(self, tokens_anchor: torch.Tensor, tokens_positive: torch.Tensor) -> torch.Tensor:
        """Forward with adaptive coefficient modulation"""
        if self.adaptive_type == 'global':
            return super().forward(tokens_anchor, tokens_positive)
        else:
            # TODO: Implement adaptive variants
            return super().forward(tokens_anchor, tokens_positive)


def test_gpf():
    """Test function for Graph Polynomial Fusion"""
    print("Testing Graph Polynomial Fusion...")
    
    # Test parameters
    batch_size = 2
    num_tokens = 196  # 14x14 patches for ViT
    token_dim = 768
    
    # Create dummy tokens
    tokens_anchor = torch.randn(batch_size, num_tokens, token_dim)
    tokens_positive = torch.randn(batch_size, num_tokens, token_dim)
    
    # Test GPF module
    gpf = GraphPolynomialFusion(degree_p=2, degree_q=2, similarity='cosine')
    
    with torch.no_grad():
        fused_graph = gpf(tokens_anchor, tokens_positive)
        
        print(f"Input tokens shape: {tokens_anchor.shape}")
        print(f"Fused graph shape: {fused_graph.shape}")
        print(f"Fused graph range: [{fused_graph.min().item():.4f}, {fused_graph.max().item():.4f}]")
        
        # Check symmetry
        is_symmetric = torch.allclose(fused_graph, fused_graph.transpose(-2, -1), atol=1e-6)
        print(f"Is symmetric: {is_symmetric}")
        
        # Check PSD (positive eigenvalues)
        eigenvals = torch.linalg.eigvals(fused_graph)
        min_eigenval = eigenvals.real.min().item()
        print(f"Minimum eigenvalue: {min_eigenval:.6f}")
        
        # Check coefficients
        coeffs = gpf.get_coefficient_matrix()
        print(f"Coefficients shape: {coeffs.shape}")
        print(f"Coefficients:\n{coeffs}")
    
    print("GPF test completed!")


if __name__ == "__main__":
    test_gpf()
