"""
Utility operations for EGO-Moment-CLE-ViT

This module contains utility functions for matrix operations, numerical stability,
and other common operations used throughout the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Optional, Union, Dict, Any
import math


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a PyTorch model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }
    
    return info


def print_model_info(model: nn.Module, input_size: Tuple[int, ...] = None):
    """
    Print detailed information about a model
    
    Args:
        model: PyTorch model
        input_size: Input tensor size for forward pass estimation
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"Model Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    if input_size is not None:
        # Estimate model size
        dummy_input = torch.randn(1, *input_size)
        try:
            with torch.no_grad():
                _ = model(dummy_input)
            print(f"  Input size: {input_size}")
        except Exception as e:
            print(f"  Could not run forward pass: {e}")


def half_vectorize_symmetric(matrix: torch.Tensor) -> torch.Tensor:
    """
    Extract upper triangular part of symmetric matrix including diagonal
    
    Args:
        matrix: [B, D, D] symmetric matrices
        
    Returns:
        vector: [B, D*(D+1)/2] half-vectorized matrices
    """
    batch_size, dim, _ = matrix.shape
    device = matrix.device
    
    # Get upper triangular indices
    triu_indices = torch.triu_indices(dim, dim, offset=0, device=device)
    
    # Extract upper triangular elements
    triu_elements = matrix[:, triu_indices[0], triu_indices[1]]  # [B, D*(D+1)/2]
    
    return triu_elements


def matrix_sqrt_newton_schulz(
    matrix: torch.Tensor,
    num_iterations: int = 5,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Compute matrix square root using Newton-Schulz iteration
    
    More numerically stable than eigendecomposition for backpropagation.
    
    Args:
        matrix: [B, D, D] positive definite matrices
        num_iterations: Number of iterations
        eps: Numerical stability constant
        
    Returns:
        sqrt_matrix: [B, D, D] matrix square roots
    """
    batch_size, dim, _ = matrix.shape
    device = matrix.device
    dtype = matrix.dtype
    
    # Trace normalization
    trace = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
    matrix_normalized = matrix / (trace + eps)
    
    # Initialize Y_0 = I, Z_0 = M_normalized
    I = torch.eye(dim, device=device, dtype=dtype).expand(batch_size, -1, -1)
    Y = I.clone()
    Z = matrix_normalized.clone()
    
    # Newton-Schulz iteration: Y_{k+1} = 0.5 * Y_k * (3*I - Z_k * Y_k)
    for _ in range(num_iterations):
        ZY = torch.bmm(Z, Y)
        YZ = torch.bmm(Y, Z)
        
        Y = 0.5 * torch.bmm(Y, 3.0 * I - ZY)
        Z = 0.5 * torch.bmm(3.0 * I - YZ, Z)
    
    # Scale back
    sqrt_trace = torch.sqrt(trace + eps)
    sqrt_matrix = Y * sqrt_trace
    
    return sqrt_matrix


def matrix_power_eigen(matrix: torch.Tensor, power: float) -> torch.Tensor:
    """
    Compute matrix power using eigendecomposition
    
    Args:
        matrix: [B, D, D] symmetric positive definite matrices
        power: Power to raise matrix to
        
    Returns:
        matrix_power: [B, D, D] matrix raised to power
    """
    # Eigendecomposition
    eigenvals, eigenvecs = torch.linalg.eigh(matrix)
    
    # Clamp eigenvalues to avoid numerical issues
    eigenvals = torch.clamp(eigenvals, min=1e-8)
    
    # Compute power
    eigenvals_power = torch.pow(eigenvals, power)
    
    # Reconstruct matrix
    matrix_power = torch.bmm(
        torch.bmm(eigenvecs, torch.diag_embed(eigenvals_power)),
        eigenvecs.transpose(-2, -1)
    )
    
    return matrix_power


def check_psd(matrix: torch.Tensor, tol: float = 1e-6) -> bool:
    """
    Check if matrices are positive semi-definite
    
    Args:
        matrix: [B, D, D] matrices to check
        tol: Tolerance for numerical errors
        
    Returns:
        is_psd: True if all matrices are PSD
    """
    try:
        eigenvals = torch.linalg.eigvals(matrix)
        min_eigenval = eigenvals.real.min().item()
        return min_eigenval >= -tol
    except:
        return False


def ensure_psd(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Ensure matrix is positive semi-definite by clamping eigenvalues
    
    Args:
        matrix: [B, D, D] input matrices
        eps: Minimum eigenvalue
        
    Returns:
        psd_matrix: [B, D, D] PSD matrices
    """
    eigenvals, eigenvecs = torch.linalg.eigh(matrix)
    eigenvals = torch.clamp(eigenvals, min=eps)
    
    psd_matrix = torch.bmm(
        torch.bmm(eigenvecs, torch.diag_embed(eigenvals)),
        eigenvecs.transpose(-2, -1)
    )
    
    return psd_matrix


def normalize_graph(graph: torch.Tensor, method: str = 'symmetric') -> torch.Tensor:
    """
    Normalize graph Laplacian or adjacency matrix
    
    Args:
        graph: [B, N, N] graph matrices
        method: Normalization method ('symmetric', 'random_walk', 'none')
        
    Returns:
        normalized_graph: [B, N, N] normalized graph matrices
    """
    if method == 'none':
        return graph
    
    eps = 1e-8
    
    # Compute degree matrix
    degrees = graph.sum(dim=-1)  # [B, N]
    degrees = torch.clamp(degrees, min=eps)
    
    if method == 'symmetric':
        # D^(-1/2) A D^(-1/2)
        inv_sqrt_degrees = 1.0 / torch.sqrt(degrees)  # [B, N]
        normalized_graph = graph * inv_sqrt_degrees.unsqueeze(-1) * inv_sqrt_degrees.unsqueeze(-2)
        
    elif method == 'random_walk':
        # D^(-1) A
        inv_degrees = 1.0 / degrees  # [B, N]
        normalized_graph = graph * inv_degrees.unsqueeze(-1)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_graph


def compute_graph_statistics(graph: torch.Tensor) -> Dict[str, float]:
    """
    Compute various statistics of graph matrices
    
    Args:
        graph: [B, N, N] graph matrices
        
    Returns:
        stats: Dictionary of statistics
    """
    stats = {}
    
    with torch.no_grad():
        # Basic statistics
        stats['mean'] = graph.mean().item()
        stats['std'] = graph.std().item()
        stats['min'] = graph.min().item()
        stats['max'] = graph.max().item()
        
        # Symmetry check
        symmetry_error = (graph - graph.transpose(-2, -1)).abs().max().item()
        stats['symmetry_error'] = symmetry_error
        stats['is_symmetric'] = symmetry_error < 1e-5
        
        # Eigenvalue statistics
        try:
            eigenvals = torch.linalg.eigvals(graph).real
            stats['min_eigenval'] = eigenvals.min().item()
            stats['max_eigenval'] = eigenvals.max().item()
            stats['eigenval_ratio'] = (eigenvals.max() / torch.clamp(eigenvals.min(), min=1e-8)).item()
            stats['is_psd'] = stats['min_eigenval'] >= -1e-6
        except:
            stats['eigenval_error'] = True
        
        # Sparsity (assuming sparse graphs should have many small values)
        threshold = 0.1 * stats['max']
        sparsity = (graph.abs() < threshold).float().mean().item()
        stats['sparsity'] = sparsity
    
    return stats


def batch_trace(matrices: torch.Tensor) -> torch.Tensor:
    """
    Compute trace for batch of matrices
    
    Args:
        matrices: [B, D, D] batch of square matrices
        
    Returns:
        traces: [B] traces of each matrix
    """
    return torch.diagonal(matrices, dim1=-2, dim2=-1).sum(-1)


def batch_logdet(matrices: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute log determinant for batch of matrices
    
    Args:
        matrices: [B, D, D] batch of square matrices
        eps: Small value to add to diagonal for numerical stability
        
    Returns:
        logdets: [B] log determinants
    """
    # Add eps to diagonal for numerical stability
    eye = torch.eye(matrices.shape[-1], device=matrices.device, dtype=matrices.dtype)
    stabilized = matrices + eps * eye.unsqueeze(0)
    
    try:
        logdets = torch.logdet(stabilized)
    except:
        # Fallback using eigenvalues
        eigenvals = torch.linalg.eigvals(stabilized).real
        eigenvals = torch.clamp(eigenvals, min=eps)
        logdets = torch.log(eigenvals).sum(-1)
    
    return logdets


def cosine_similarity_matrix(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix
    
    Args:
        features: [B, N, D] or [N, D] feature matrices
        eps: Small value for numerical stability
        
    Returns:
        similarity: [B, N, N] or [N, N] cosine similarity matrices
    """
    if features.dim() == 2:
        features = features.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # L2 normalize
    features_norm = F.normalize(features, p=2, dim=-1, eps=eps)
    
    # Compute similarity
    similarity = torch.bmm(features_norm, features_norm.transpose(-2, -1))
    
    if squeeze_output:
        similarity = similarity.squeeze(0)
    
    return similarity


def test_ops():
    """Test utility operations"""
    print("Testing Utility Operations...")
    
    # Test data
    batch_size = 3
    dim = 4
    
    # Create test matrices
    A = torch.randn(batch_size, dim, dim)
    symmetric_A = torch.bmm(A, A.transpose(-2, -1))  # Make PSD
    
    print(f"Test matrix shape: {symmetric_A.shape}")
    
    # Test half vectorization
    print("\n=== Half Vectorization ===")
    half_vec = half_vectorize_symmetric(symmetric_A)
    print(f"Half vectorized shape: {half_vec.shape}")
    print(f"Expected shape: {batch_size} x {dim * (dim + 1) // 2}")
    
    # Test matrix square root
    print("\n=== Matrix Square Root ===")
    sqrt_A = matrix_sqrt_newton_schulz(symmetric_A, num_iterations=3)
    print(f"Square root shape: {sqrt_A.shape}")
    
    # Verify: sqrt(A) * sqrt(A) ≈ A
    reconstructed = torch.bmm(sqrt_A, sqrt_A)
    error = (reconstructed - symmetric_A).abs().max().item()
    print(f"Reconstruction error: {error:.6f}")
    
    # Test PSD check
    print("\n=== PSD Check ===")
    is_psd = check_psd(symmetric_A)
    print(f"Is PSD: {is_psd}")
    
    # Test graph normalization
    print("\n=== Graph Normalization ===")
    # Make adjacency-like matrix (non-negative)
    adj_matrix = torch.abs(symmetric_A)
    normalized = normalize_graph(adj_matrix, method='symmetric')
    print(f"Normalized graph shape: {normalized.shape}")
    
    # Test graph statistics
    print("\n=== Graph Statistics ===")
    stats = compute_graph_statistics(adj_matrix)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test cosine similarity
    print("\n=== Cosine Similarity ===")
    features = torch.randn(batch_size, 5, 8)
    cos_sim = cosine_similarity_matrix(features)
    print(f"Cosine similarity shape: {cos_sim.shape}")
    print(f"Similarity range: [{cos_sim.min().item():.3f}, {cos_sim.max().item():.3f}]")
    
    print("Utility operations test completed!")


if __name__ == "__main__":
    test_ops()
