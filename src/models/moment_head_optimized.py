"""
Memory-Optimized Moment Head - Graph-weighted high-order statistical moment pooling

This module implements memory-efficient graph-weighted second and third-order moment pooling
with gradient checkpointing, in-place operations, and chunked processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
from torch.utils.checkpoint import checkpoint


class MemoryEfficientNewtonSchulz(nn.Module):
    """
    Memory-efficient Newton-Schulz iteration with in-place operations
    """
    
    def __init__(self, num_iterations: int = 5, eps: float = 1e-5):
        super().__init__()
        self.num_iterations = num_iterations
        self.eps = eps
    
    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient matrix^(-1/2) computation
        """
        batch_size, dim, _ = matrix.shape
        device = matrix.device
        
        # Trace normalization
        with torch.no_grad():
            trace = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        
        # In-place normalization to save memory
        matrix.div_(trace + self.eps)
        
        # Initialize Y = I (reuse matrix memory when possible)
        Y = torch.eye(dim, device=device, dtype=matrix.dtype).expand_as(matrix).contiguous()
        
        # Newton-Schulz iteration with in-place operations
        for i in range(self.num_iterations):
            # Use gradient checkpointing for memory efficiency during backprop
            if self.training:
                Y = checkpoint(self._newton_step, Y, matrix, use_reentrant=False)
            else:
                Y = self._newton_step(Y, matrix)
        
        # Scale back in-place
        sqrt_trace = torch.sqrt(trace + self.eps)
        Y.div_(sqrt_trace)
        
        return Y
    
    def _newton_step(self, Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Single Newton-Schulz step with memory optimization"""
        # Create identity once
        I = torch.eye(Y.shape[-1], device=Y.device, dtype=Y.dtype)
        I = I.expand_as(Y)
        
        # Compute products (reuse tensors where possible)
        ZY = torch.bmm(Z, Y)
        
        # Y_{k+1} = 0.5 * Y * (3*I - ZY)
        temp = 3.0 * I - ZY
        Y_new = 0.5 * torch.bmm(Y, temp)
        
        # Update Z in-place: Z_{k+1} = 0.5 * (3*I - Y*Z) * Z
        YZ = torch.bmm(Y, Z)
        temp = 3.0 * I - YZ
        Z = 0.5 * torch.bmm(temp, Z)
        
        return Y_new


class CompactTensorSketch(nn.Module):
    """
    Memory-efficient tensor sketch with reduced sketch dimensions
    """
    
    def __init__(self, input_dim: int, sketch_dim: int = 2048, seed: int = 42):
        super().__init__()
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        
        # Use smaller sketch dimension for memory efficiency
        effective_sketch_dim = min(sketch_dim, input_dim * 2)
        self.effective_sketch_dim = effective_sketch_dim
        
        torch.manual_seed(seed)
        
        # Store hash functions more efficiently
        self.register_buffer('hash_indices', torch.randint(0, effective_sketch_dim, (3, input_dim)))
        self.register_buffer('signs', torch.randint(0, 2, (3, input_dim)) * 2 - 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient third-order sketch computation
        """
        batch_size = x.shape[0]
        
        # Process sketches sequentially to save memory
        sketch_result = torch.ones(batch_size, self.effective_sketch_dim, 
                                 device=x.device, dtype=x.dtype)
        
        for i in range(3):
            # Apply count sketch
            sketched = torch.zeros_like(sketch_result)
            x_signed = x * self.signs[i].unsqueeze(0)
            sketched.scatter_add_(1, self.hash_indices[i].unsqueeze(0).expand(batch_size, -1), x_signed)
            
            # Multiply in-place to save memory
            sketch_result.mul_(sketched)
        
        # Pad to original sketch_dim if needed
        if self.effective_sketch_dim < self.sketch_dim:
            padding = torch.zeros(batch_size, self.sketch_dim - self.effective_sketch_dim,
                                device=x.device, dtype=x.dtype)
            sketch_result = torch.cat([sketch_result, padding], dim=1)
        
        return sketch_result


class MemoryOptimizedMomentHead(nn.Module):
    """
    Memory-optimized graph-weighted moment pooling head
    
    Optimizations:
    1. Gradient checkpointing for memory-intensive operations
    2. In-place operations where possible
    3. Chunked processing for large batches
    4. Reduced intermediate tensor allocations
    5. Optional mixed precision support
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int = 1024,
        use_third_order: bool = True,
        isqrt_iterations: int = 3,  # Reduced iterations
        sketch_dim: int = 2048,     # Reduced sketch dimension
        chunk_size: int = None,     # For chunked processing
        use_mixed_precision: bool = True,
        eps: float = 1e-5
    ):
        super().__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        self.use_third_order = use_third_order
        self.chunk_size = chunk_size
        self.use_mixed_precision = use_mixed_precision
        self.eps = eps
        
        # Memory-efficient components
        self.isqrt_cov = MemoryEfficientNewtonSchulz(num_iterations=isqrt_iterations, eps=eps)
        
        if use_third_order:
            self.tensor_sketch = CompactTensorSketch(d_in, sketch_dim)
        
        # Dimension allocation
        if use_third_order:
            self.d_second = d_out // 2
            self.d_third = d_out - self.d_second
        else:
            self.d_second = d_out
            self.d_third = 0
        
        # Smaller networks to reduce memory
        second_input_dim = (d_in * (d_in + 1)) // 2
        self.second_net = nn.Sequential(
            nn.Linear(second_input_dim, self.d_second),
            nn.BatchNorm1d(self.d_second),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        if use_third_order:
            self.third_net = nn.Sequential(
                nn.Linear(sketch_dim, self.d_third),
                nn.BatchNorm1d(self.d_third),
                nn.GELU(),
                nn.Dropout(0.1)
            )
    
    def _chunked_bmm(self, a: torch.Tensor, b: torch.Tensor, chunk_size: int = None) -> torch.Tensor:
        """Memory-efficient batch matrix multiplication"""
        if chunk_size is None or a.shape[0] <= chunk_size:
            return torch.bmm(a, b)
        
        results = []
        for i in range(0, a.shape[0], chunk_size):
            end_idx = min(i + chunk_size, a.shape[0])
            chunk_result = torch.bmm(a[i:end_idx], b[i:end_idx])
            results.append(chunk_result)
        
        return torch.cat(results, dim=0)
    
    def _memory_efficient_half_vectorize(self, matrix: torch.Tensor) -> torch.Tensor:
        """Memory-efficient upper triangular extraction"""
        batch_size, dim, _ = matrix.shape
        
        # Use more memory-efficient indexing
        indices = torch.triu_indices(dim, dim, offset=0)
        result = matrix[:, indices[0], indices[1]]
        
        return result
    
    @torch.amp.autocast('cuda', enabled=False)  # Disable autocast for precision-sensitive ops
    def _compute_second_order_moment(self, tokens_centered: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Compute second-order moment with memory optimization"""
        if self.chunk_size:
            # Chunked computation for large batches
            weighted_tokens = self._chunked_bmm(W, tokens_centered, self.chunk_size)
            M2 = self._chunked_bmm(tokens_centered.transpose(-2, -1), weighted_tokens, self.chunk_size)
        else:
            # Standard computation
            weighted_tokens = torch.bmm(W, tokens_centered)
            M2 = torch.bmm(tokens_centered.transpose(-2, -1), weighted_tokens)
        
        # Use gradient checkpointing for iSQRT
        if self.training:
            M2_normalized = checkpoint(self.isqrt_cov, M2, use_reentrant=False)
        else:
            M2_normalized = self.isqrt_cov(M2)
        
        return M2_normalized
    
    def forward(self, tokens: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Memory-optimized forward pass
        """
        batch_size, num_tokens, token_dim = tokens.shape
        
        # Use mixed precision context if enabled
        if self.use_mixed_precision and torch.cuda.is_available():
            autocast_context = torch.amp.autocast('cuda')
        else:
            autocast_context = torch.no_grad()
        
        with autocast_context:
            # Normalize graph to weight matrix (in-place when possible)
            ones = torch.ones(batch_size, num_tokens, 1, device=graph.device, dtype=graph.dtype)
            degrees = torch.bmm(graph, ones).squeeze(-1)
            degrees = torch.clamp(degrees, min=self.eps)
            
            inv_sqrt_degrees = 1.0 / torch.sqrt(degrees)
            W = graph * inv_sqrt_degrees.unsqueeze(-1) * inv_sqrt_degrees.unsqueeze(-2)
            
            # Compute weighted mean efficiently
            weighted_sum = torch.bmm(tokens.transpose(-2, -1), torch.bmm(W, ones))
            trace_w = torch.diagonal(W, dim1=-2, dim2=-1).sum(-1, keepdim=True)
            mu = weighted_sum.squeeze(-1) / (trace_w + self.eps)
            
            # Center tokens
            tokens_centered = tokens - mu.unsqueeze(1)
            
            # Second-order moment computation
            M2_normalized = self._compute_second_order_moment(tokens_centered, W)
            
            # Process second-order features
            M2_vec = self._memory_efficient_half_vectorize(M2_normalized)
            second_features = self.second_net(M2_vec)
            
            features_list = [second_features]
            
            # Third-order moment (if enabled)
            if self.use_third_order:
                # More memory-efficient third-order computation
                W_sum = torch.bmm(W, torch.ones_like(tokens_centered))
                weighted_centered = (tokens_centered * W_sum).sum(dim=1) / (trace_w.squeeze(-1) + self.eps)
                
                # Apply tensor sketch
                third_sketch = self.tensor_sketch(weighted_centered)
                third_features = self.third_net(third_sketch)
                
                features_list.append(third_features)
            
            # Concatenate features
            moment_features = torch.cat(features_list, dim=-1)
        
        return moment_features


def test_memory_usage():
    """Test memory usage comparison"""
    import torch.profiler
    
    print("Testing Memory Usage...")
    
    # Test parameters
    batch_size = 4
    num_tokens = 196
    token_dim = 768
    d_out = 1024
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    tokens = torch.randn(batch_size, num_tokens, token_dim, device=device)
    graph = torch.randn(batch_size, num_tokens, num_tokens, device=device)
    graph = torch.bmm(graph, graph.transpose(-2, -1))
    graph = 0.5 * (graph + graph.transpose(-2, -1))
    
    # Test optimized version
    optimized_head = MemoryOptimizedMomentHead(
        d_in=token_dim,
        d_out=d_out,
        use_third_order=True,
        isqrt_iterations=3,
        sketch_dim=2048,
        chunk_size=2,
        use_mixed_precision=True
    ).to(device)
    
    # Warmup
    with torch.no_grad():
        _ = optimized_head(tokens, graph)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        start_memory = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            result = optimized_head(tokens, graph)
        
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        
        print(f"Peak GPU memory usage: {(peak_memory - start_memory) / 1024**2:.2f} MB")
        print(f"Output shape: {result.shape}")
    else:
        with torch.no_grad():
            result = optimized_head(tokens, graph)
        print(f"Output shape: {result.shape}")
    
    print("Memory test completed!")


if __name__ == "__main__":
    test_memory_usage()
