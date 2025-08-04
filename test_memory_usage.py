"""
Memory usage comparison between original and optimized MomentHead
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.moment_head import MomentHead
from src.models.moment_head_optimized import MemoryOptimizedMomentHead


def measure_memory_usage(model, tokens, graph, device, name):
    """Measure peak memory usage of a model"""
    model = model.to(device)
    tokens = tokens.to(device)
    graph = graph.to(device)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Warmup
        with torch.no_grad():
            _ = model(tokens, graph)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        with torch.no_grad():
            output = model(tokens, graph)
        
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = (peak_memory - start_memory) / 1024**2  # MB
        
        print(f"{name}:")
        print(f"  Peak memory: {memory_used:.2f} MB")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return memory_used
    else:
        with torch.no_grad():
            output = model(tokens, graph)
        print(f"{name} - Output shape: {output.shape} (CPU mode)")
        return 0


def run_memory_benchmark():
    """Run comprehensive memory benchmark"""
    print("=== Memory Usage Benchmark ===\n")
    
    # Test configurations
    configs = [
        {"batch_size": 2, "num_tokens": 196, "token_dim": 768, "name": "Small (B=2, N=196, D=768)"},
        {"batch_size": 4, "num_tokens": 196, "token_dim": 768, "name": "Medium (B=4, N=196, D=768)"},
        {"batch_size": 8, "num_tokens": 196, "token_dim": 768, "name": "Large (B=8, N=196, D=768)"},
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    results = []
    
    for config in configs:
        print(f"Testing {config['name']}:")
        print("-" * 50)
        
        batch_size = config["batch_size"]
        num_tokens = config["num_tokens"] 
        token_dim = config["token_dim"]
        
        # Create test data
        tokens = torch.randn(batch_size, num_tokens, token_dim)
        graph = torch.randn(batch_size, num_tokens, num_tokens)
        graph = torch.bmm(graph, graph.transpose(-2, -1))
        graph = 0.5 * (graph + graph.transpose(-2, -1))
        
        # Original model (with optimizations applied)
        original_model = MomentHead(
            d_in=token_dim,
            d_out=512,  # Reduced from 1024
            use_third_order=False,  # Disabled to save memory
            isqrt_iterations=3,
            sketch_dim=2048
        )
        
        # Fully optimized model  
        optimized_model = MemoryOptimizedMomentHead(
            d_in=token_dim,
            d_out=512,
            use_third_order=False,
            isqrt_iterations=3,
            sketch_dim=2048,
            chunk_size=batch_size//2 if batch_size > 2 else None,
            use_mixed_precision=True
        )
        
        # Measure memory usage
        original_memory = measure_memory_usage(original_model, tokens, graph, device, "Original (Optimized)")
        optimized_memory = measure_memory_usage(optimized_model, tokens, graph, device, "Fully Optimized")
        
        if device.type == 'cuda':
            memory_reduction = ((original_memory - optimized_memory) / original_memory) * 100
            print(f"  Memory reduction: {memory_reduction:.1f}%")
        
        print()
        
        results.append({
            'config': config['name'],
            'original_memory': original_memory,
            'optimized_memory': optimized_memory
        })
    
    # Summary
    print("=== Summary ===")
    for result in results:
        if device.type == 'cuda':
            reduction = ((result['original_memory'] - result['optimized_memory']) / result['original_memory']) * 100
            print(f"{result['config']}: {reduction:.1f}% memory reduction")
        else:
            print(f"{result['config']}: Tested on CPU")


def test_third_order_impact():
    """Test memory impact of third-order moments"""
    print("\n=== Third-Order Memory Impact ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 4
    num_tokens = 196
    token_dim = 768
    
    tokens = torch.randn(batch_size, num_tokens, token_dim)
    graph = torch.randn(batch_size, num_tokens, num_tokens)
    graph = torch.bmm(graph, graph.transpose(-2, -1))
    graph = 0.5 * (graph + graph.transpose(-2, -1))
    
    # Test with different third-order settings
    print("Second-order only:")
    model_2nd = MemoryOptimizedMomentHead(
        d_in=token_dim,
        d_out=512,
        use_third_order=False,
        sketch_dim=2048
    )
    measure_memory_usage(model_2nd, tokens, graph, device, "  2nd order only")
    
    print("\nSecond + Third order:")
    model_3rd = MemoryOptimizedMomentHead(
        d_in=token_dim,
        d_out=512,
        use_third_order=True,
        sketch_dim=2048
    )
    measure_memory_usage(model_3rd, tokens, graph, device, "  2nd + 3rd order")


if __name__ == "__main__":
    run_memory_benchmark()
    test_third_order_impact()
