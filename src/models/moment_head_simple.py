"""
Simplified Memory-Optimized Moment Head

Focus on practical memory optimizations without complex features
that might introduce bugs or overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class SimplifiedMomentHead(nn.Module):
    """
    Simplified memory-optimized moment head with key optimizations:
    
    1. Reduced default parameters
    2. In-place operations where safe
    3. Simplified Newton-Schulz iteration
    4. Optional gradient checkpointing
    5. Smaller network architectures
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int = 512,
        use_third_order: bool = False,
        isqrt_iterations: int = 3,
        sketch_dim: int = 1024,  # Much smaller
        use_gradient_checkpointing: bool = True,
        eps: float = 1e-5
    ):
        super().__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        self.use_third_order = use_third_order
        self.isqrt_iterations = isqrt_iterations
        self.eps = eps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Dimension allocation
        if use_third_order:
            self.d_second = d_out // 2
            self.d_third = d_out - self.d_second
            
            # Simple third-order approximation via random projection
            self.register_buffer('random_proj', torch.randn(d_in, sketch_dim) / math.sqrt(d_in))
        else:
            self.d_second = d_out
            self.d_third = 0
        
        # Simplified networks
        second_input_dim = (d_in * (d_in + 1)) // 2
        self.second_net = nn.Sequential(
            nn.Linear(second_input_dim, self.d_second),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        if use_third_order:
            self.third_net = nn.Sequential(
                nn.Linear(sketch_dim, self.d_third),
                nn.GELU(),
                nn.Dropout(0.1)
            )
    
    def _simplified_isqrt(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Simplified iSQRT-COV with minimal memory overhead
        """
        # Trace normalization
        trace = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        matrix = matrix / (trace + self.eps)
        
        # Initialize
        dim = matrix.shape[-1]
        Y = torch.eye(dim, device=matrix.device, dtype=matrix.dtype).expand_as(matrix)
        
        # Simplified Newton-Schulz (fewer iterations, simpler updates)
        for _ in range(self.isqrt_iterations):
            # Y = 0.5 * Y * (3*I - matrix @ Y)
            MY = torch.bmm(matrix, Y)
            I = torch.eye(dim, device=matrix.device, dtype=matrix.dtype).expand_as(matrix)
            Y = 0.5 * torch.bmm(Y, 3.0 * I - MY)
        
        # Scale back
        Y = Y / torch.sqrt(trace + self.eps)
        return Y
    
    def _extract_upper_triangular(self, matrix: torch.Tensor) -> torch.Tensor:
        """Extract upper triangular part efficiently"""
        dim = matrix.shape[-1]
        indices = torch.triu_indices(dim, dim, device=matrix.device)
        return matrix[:, indices[0], indices[1]]
    
    def _compute_graph_weighted_stats(self, tokens: torch.Tensor, graph: torch.Tensor):
        """Compute weighted mean and second moment efficiently"""
        # Normalize graph to weights
        degrees = graph.sum(dim=-1, keepdim=True)  # [B, N, 1]
        weights = graph / (degrees + self.eps)  # Row-normalized
        
        # Weighted mean: μ = Σ(w_i * z_i)
        weighted_mean = torch.bmm(weights.sum(dim=1, keepdim=True), tokens).squeeze(1)  # [B, D]
        
        # Center tokens
        tokens_centered = tokens - weighted_mean.unsqueeze(1)  # [B, N, D]
        
        # Weighted second moment: M2 = Σ(w_i * (z_i - μ)(z_i - μ)^T)
        # Use batch-wise computation to save memory
        M2 = torch.zeros(tokens.shape[0], self.d_in, self.d_in, 
                        device=tokens.device, dtype=tokens.dtype)
        
        for i in range(tokens.shape[0]):
            # For each sample in batch
            w = weights[i]  # [N, N]
            z_c = tokens_centered[i]  # [N, D]
            
            # Weighted covariance: z_c^T @ w @ z_c
            weighted_z = torch.mm(w, z_c)  # [N, D]
            M2[i] = torch.mm(z_c.t(), weighted_z)  # [D, D]
        
        return weighted_mean, M2
    
    def forward(self, tokens: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Memory-optimized forward pass
        """
        # Compute weighted statistics
        weighted_mean, M2 = self._compute_graph_weighted_stats(tokens, graph)
        
        # Normalize second moment
        if self.use_gradient_checkpointing and self.training:
            M2_normalized = torch.utils.checkpoint.checkpoint(
                self._simplified_isqrt, M2, use_reentrant=False
            )
        else:
            M2_normalized = self._simplified_isqrt(M2)
        
        # Extract features
        M2_vec = self._extract_upper_triangular(M2_normalized)
        second_features = self.second_net(M2_vec)
        
        features = [second_features]
        
        # Third-order features (if enabled)
        if self.use_third_order:
            # Simple third-order approximation using random projection
            # Approximate E[(x-μ)^3] via random projection
            centered_mean = weighted_mean - tokens.mean(dim=1)  # [B, D]
            
            # Project to lower dimension
            projected = torch.mm(centered_mean, self.random_proj)  # [B, sketch_dim]
            
            # Cube to approximate third moment
            third_approx = projected.pow(3)  # Element-wise cubing
            
            third_features = self.third_net(third_approx)
            features.append(third_features)
        
        return torch.cat(features, dim=-1)


def compare_memory_usage():
    """Compare memory usage with different configurations"""
    print("=== Simplified Memory Optimization Test ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test parameters
    batch_size = 4
    num_tokens = 196
    token_dim = 768
    
    # Create test data
    tokens = torch.randn(batch_size, num_tokens, token_dim, device=device)
    graph = torch.randn(batch_size, num_tokens, num_tokens, device=device)
    graph = torch.bmm(graph, graph.transpose(-2, -1))
    graph = 0.5 * (graph + graph.transpose(-2, -1))
    
    configs = [
        {
            'name': 'Minimal (d_out=256, 2nd only)',
            'model': SimplifiedMomentHead(
                d_in=token_dim,
                d_out=256,
                use_third_order=False,
                isqrt_iterations=2
            )
        },
        {
            'name': 'Standard (d_out=512, 2nd only)', 
            'model': SimplifiedMomentHead(
                d_in=token_dim,
                d_out=512,
                use_third_order=False,
                isqrt_iterations=3
            )
        },
        {
            'name': 'With 3rd order (d_out=512)',
            'model': SimplifiedMomentHead(
                d_in=token_dim,
                d_out=512,
                use_third_order=True,
                sketch_dim=512,
                isqrt_iterations=3
            )
        }
    ]
    
    for config in configs:
        model = config['model'].to(device)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Warmup
            with torch.no_grad():
                _ = model(tokens, graph)
            
            torch.cuda.empty_cache() 
            torch.cuda.reset_peak_memory_stats()
            
            start_memory = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                output = model(tokens, graph)
            
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = (peak_memory - start_memory) / 1024**2
            
            print(f"{config['name']}:")
            print(f"  Memory: {memory_used:.2f} MB")
            print(f"  Output: {output.shape}")
            print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
            print()
        else:
            with torch.no_grad():
                output = model(tokens, graph)
            print(f"{config['name']}: {output.shape} (CPU)")


if __name__ == "__main__":
    compare_memory_usage()
