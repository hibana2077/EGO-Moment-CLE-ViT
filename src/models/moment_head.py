"""
Moment Head - Graph-weighted high-order statistical moment pooling

This module implements graph-weighted second and third-order moment pooling
with iSQRT-COV normalization and Tensor-Sketch approximation for computational efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class NewtonSchulzSqrtm(nn.Module):
    """
    Newton-Schulz iteration for matrix square root (iSQRT-COV)
    
    Computes M^(-1/2) using iterative method for covariance normalization.
    More stable than direct eigendecomposition for backpropagation.
    """
    
    def __init__(self, num_iterations: int = 5, eps: float = 1e-5):
        super().__init__()
        self.num_iterations = num_iterations
        self.eps = eps
    
    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute matrix^(-1/2) using Newton-Schulz iteration
        
        Args:
            matrix: [B, D, D] positive definite matrices
            
        Returns:
            isqrt: [B, D, D] inverse square root matrices
        """
        batch_size, dim, _ = matrix.shape
        device = matrix.device
        
        # Trace normalization for better numerical stability
        trace = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1, keepdim=True)  # [B, 1]
        trace = trace.unsqueeze(-1)  # [B, 1, 1]
        matrix_normalized = matrix / (trace + self.eps)
        
        # Initialize Y_0 = I, Z_0 = M_normalized
        I = torch.eye(dim, device=device, dtype=matrix.dtype).expand(batch_size, -1, -1)
        Y = I.clone()
        Z = matrix_normalized.clone()
        
        # Newton-Schulz iteration
        for i in range(self.num_iterations):
            # Y_{k+1} = 0.5 * Y_k * (3*I - Z_k * Y_k)
            # Z_{k+1} = 0.5 * (3*I - Y_k * Z_k) * Z_k
            ZY = torch.bmm(Z, Y)
            YZ = torch.bmm(Y, Z)
            
            Y = 0.5 * torch.bmm(Y, 3.0 * I - ZY)
            Z = 0.5 * torch.bmm(3.0 * I - YZ, Z)
        
        # Scale back
        sqrt_trace = torch.sqrt(trace + self.eps)
        isqrt = Y / sqrt_trace
        
        return isqrt


class TensorSketch(nn.Module):
    """
    Tensor-Sketch for efficient third-order moment approximation
    
    Approximates third-order tensor X ⊗ X ⊗ X using random projections
    to reduce computational complexity from O(D^3) to O(D * sketch_dim).
    """
    
    def __init__(self, input_dim: int, sketch_dim: int = 4096, seed: int = 42):
        super().__init__()
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        
        # Fixed random projections for consistency
        torch.manual_seed(seed)
        
        # Random hash functions for Count-Sketch
        self.register_buffer('hash1', torch.randint(0, sketch_dim, (input_dim,)))
        self.register_buffer('hash2', torch.randint(0, sketch_dim, (input_dim,)))
        self.register_buffer('hash3', torch.randint(0, sketch_dim, (input_dim,)))
        
        # Random signs
        self.register_buffer('sign1', torch.randint(0, 2, (input_dim,)) * 2 - 1)
        self.register_buffer('sign2', torch.randint(0, 2, (input_dim,)) * 2 - 1) 
        self.register_buffer('sign3', torch.randint(0, 2, (input_dim,)) * 2 - 1)
    
    def _count_sketch(self, x: torch.Tensor, hash_idx: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        """Apply Count-Sketch to input tensor"""
        batch_size = x.shape[0]
        sketched = torch.zeros(batch_size, self.sketch_dim, device=x.device, dtype=x.dtype)
        
        # Apply signs
        x_signed = x * signs.unsqueeze(0)  # [B, D]
        
        # Scatter to sketch
        sketched.scatter_add_(1, hash_idx.unsqueeze(0).expand(batch_size, -1), x_signed)
        
        return sketched
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute third-order tensor sketch
        
        Args:
            x: [B, D] input features (typically centered: x - μ)
            
        Returns:
            sketch: [B, sketch_dim] approximated third-order features  
        """
        # Apply Count-Sketch to each copy
        sketch1 = self._count_sketch(x, self.hash1, self.sign1)  # [B, sketch_dim]
        sketch2 = self._count_sketch(x, self.hash2, self.sign2)  # [B, sketch_dim]
        sketch3 = self._count_sketch(x, self.hash3, self.sign3)  # [B, sketch_dim]
        
        # Element-wise product to approximate tensor product
        # This approximates the trilinear form x ⊗ x ⊗ x
        sketch = sketch1 * sketch2 * sketch3  # [B, sketch_dim]
        
        return sketch


class MomentHead(nn.Module):
    """
    Graph-weighted high-order moment pooling head
    
    Computes second-order (covariance) and optionally third-order moments
    using graph weighting from GPF output, with efficient approximations.
    
    Args:
        d_in: Input token dimension
        d_out: Output feature dimension
        use_third_order: Whether to include third-order moments
        isqrt_iterations: Number of Newton-Schulz iterations for iSQRT-COV
        sketch_dim: Dimension for tensor sketch approximation
        eps: Numerical stability constant
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int = 1024,
        use_third_order: bool = True,
        isqrt_iterations: int = 5,
        sketch_dim: int = 4096,
        eps: float = 1e-5
    ):
        super().__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        self.use_third_order = use_third_order
        self.eps = eps
        
        # iSQRT-COV for second-order moment normalization
        self.isqrt_cov = NewtonSchulzSqrtm(num_iterations=isqrt_iterations, eps=eps)
        
        # Tensor-Sketch for third-order approximation
        if use_third_order:
            self.tensor_sketch = TensorSketch(d_in, sketch_dim)
        
        # Dimensions for each component
        if use_third_order:
            # Half dimensions for second and third order
            self.d_second = d_out // 2
            self.d_third = d_out - self.d_second
        else:
            self.d_second = d_out
            self.d_third = 0
        
        # Second-order processing network
        # Input: upper triangular part of symmetric matrix
        second_input_dim = (d_in * (d_in + 1)) // 2  # Upper triangular + diagonal
        self.second_net = nn.Sequential(
            nn.Linear(second_input_dim, self.d_second * 2),
            nn.BatchNorm1d(self.d_second * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_second * 2, self.d_second)
        )
        
        # Third-order processing network (if enabled)
        if use_third_order:
            self.third_net = nn.Sequential(
                nn.Linear(sketch_dim, self.d_third * 2),
                nn.BatchNorm1d(self.d_third * 2), 
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_third * 2, self.d_third)
            )
    
    def _half_vectorize(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Extract upper triangular part of symmetric matrix including diagonal
        
        Args:
            matrix: [B, D, D] symmetric matrices
            
        Returns:
            vector: [B, D*(D+1)/2] half-vectorized matrices
        """
        batch_size, dim, _ = matrix.shape
        
        # Get upper triangular indices
        triu_indices = torch.triu_indices(dim, dim, offset=0)
        
        # Extract upper triangular elements
        triu_elements = matrix[:, triu_indices[0], triu_indices[1]]  # [B, D*(D+1)/2]
        
        return triu_elements
    
    def _graph_weighted_mean(self, tokens: torch.Tensor, weight_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute graph-weighted mean: μ = (Z^T W 1) / tr(W)
        
        Args:
            tokens: [B, N, D] token features
            weight_matrix: [B, N, N] graph weight matrix
            
        Returns:
            weighted_mean: [B, D] graph-weighted mean features
        """
        batch_size = tokens.shape[0]
        # Compute weighted sum: Z^T @ W @ 1
        ones = torch.ones(batch_size, tokens.shape[1], 1, device=tokens.device, dtype=tokens.dtype)  # [B, N, 1]
        weighted_sum = torch.bmm(tokens.transpose(-2, -1), torch.bmm(weight_matrix, ones))  # [B, D, 1]
        
        # Compute normalization: tr(W)
        trace_w = torch.diagonal(weight_matrix, dim1=-2, dim2=-1).sum(-1, keepdim=True)  # [B, 1]
        
        # Weighted mean
        weighted_mean = weighted_sum.squeeze(-1) / (trace_w + self.eps)  # [B, D]
        
        return weighted_mean
    
    def _normalize_weight_matrix(self, graph: torch.Tensor) -> torch.Tensor:
        """
        Normalize graph to proper weight matrix: W = D^(-1/2) G D^(-1/2)
        
        Args:
            graph: [B, N, N] fused graph from GPF
            
        Returns:
            weight_matrix: [B, N, N] normalized weight matrix
        """
        # Compute degree matrix D = diag(G @ 1)
        batch_size = graph.shape[0]
        ones = torch.ones(batch_size, graph.shape[-1], 1, device=graph.device, dtype=graph.dtype)  # [B, N, 1]
        degrees = torch.bmm(graph, ones)  # [B, N, 1]
        degrees = degrees.squeeze(-1)  # [B, N]
        
        # Add eps for numerical stability
        degrees = torch.clamp(degrees, min=self.eps)
        
        # D^(-1/2)
        inv_sqrt_degrees = 1.0 / torch.sqrt(degrees)  # [B, N]
        
        # W = D^(-1/2) G D^(-1/2)
        weight_matrix = graph * inv_sqrt_degrees.unsqueeze(-1) * inv_sqrt_degrees.unsqueeze(-2)
        
        return weight_matrix
    
    def forward(self, tokens: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of moment pooling
        
        Args:
            tokens: [B, N, D] token features (typically from anchor view)
            graph: [B, N, N] fused relation graph from GPF
            
        Returns:
            moment_features: [B, d_out] aggregated moment features
        """
        batch_size, num_tokens, token_dim = tokens.shape
        
        # Normalize graph to weight matrix  
        W = self._normalize_weight_matrix(graph)  # [B, N, N]
        
        # Compute graph-weighted mean
        mu = self._graph_weighted_mean(tokens, W)  # [B, D]
        
        # Center tokens: Z_centered = Z - μ
        tokens_centered = tokens - mu.unsqueeze(1)  # [B, N, D]
        
        # === Second-order moment ===
        # M2 = (Z - μ)^T W (Z - μ)
        weighted_tokens = torch.bmm(W, tokens_centered)  # [B, N, D]
        M2 = torch.bmm(tokens_centered.transpose(-2, -1), weighted_tokens)  # [B, D, D]
        
        # iSQRT-COV normalization 
        M2_normalized = self.isqrt_cov(M2)  # [B, D, D]
        
        # Half-vectorize and process
        M2_vec = self._half_vectorize(M2_normalized)  # [B, D*(D+1)/2]
        second_features = self.second_net(M2_vec)  # [B, d_second]
        
        features_list = [second_features]
        
        # === Third-order moment (if enabled) ===
        if self.use_third_order:
            # Compute weighted average of centered tokens for third-order
            trace_w = torch.diagonal(W, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
            
            # Weight each token by its graph connectivity
            token_weights = torch.bmm(W, torch.ones_like(tokens_centered))  # [B, N, D]
            weighted_centered = (tokens_centered * token_weights).sum(dim=1) / (trace_w.squeeze(-1) + self.eps)  # [B, D]
            
            # Apply tensor sketch
            third_sketch = self.tensor_sketch(weighted_centered)  # [B, sketch_dim]
            third_features = self.third_net(third_sketch)  # [B, d_third]
            
            features_list.append(third_features)
        
        # Concatenate all moment features
        moment_features = torch.cat(features_list, dim=-1)  # [B, d_out]
        
        return moment_features


def test_moment_head():
    """Test function for Moment Head"""
    print("Testing Moment Head...")
    
    # Test parameters
    batch_size = 2
    num_tokens = 196
    token_dim = 768
    d_out = 1024
    
    # Create dummy data
    tokens = torch.randn(batch_size, num_tokens, token_dim)
    
    # Create dummy graph (PSD matrix)
    graph = torch.randn(batch_size, num_tokens, num_tokens)
    graph = torch.bmm(graph, graph.transpose(-2, -1))  # Ensure PSD
    graph = 0.5 * (graph + graph.transpose(-2, -1))  # Ensure symmetric
    
    # Test moment head
    moment_head = MomentHead(
        d_in=token_dim,
        d_out=d_out,
        use_third_order=True,
        isqrt_iterations=3,
        sketch_dim=2048
    )
    
    with torch.no_grad():
        moment_features = moment_head(tokens, graph)
        
        print(f"Input tokens shape: {tokens.shape}")
        print(f"Input graph shape: {graph.shape}")
        print(f"Output moment features shape: {moment_features.shape}")
        print(f"Feature range: [{moment_features.min().item():.4f}, {moment_features.max().item():.4f}]")
        
        # Test without third-order
        moment_head_2nd = MomentHead(
            d_in=token_dim,
            d_out=d_out,
            use_third_order=False
        )
        
        moment_features_2nd = moment_head_2nd(tokens, graph)
        print(f"Second-order only shape: {moment_features_2nd.shape}")
    
    print("Moment Head test completed!")


if __name__ == "__main__":
    test_moment_head()
