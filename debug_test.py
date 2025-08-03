"""
Simplified test to debug specific issues
"""

import os
import sys
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_gpf():
    """Test GPF in isolation"""
    from models import GraphPolynomialFusion
    
    gpf = GraphPolynomialFusion(degree_p=2, degree_q=2)
    
    batch_size = 2
    num_patches = 10
    embed_dim = 64
    
    anchor = torch.randn(batch_size, num_patches, embed_dim)
    positive = torch.randn(batch_size, num_patches, embed_dim)
    
    graph = gpf(anchor, positive)
    print(f"GPF test passed: {graph.shape}")

def test_moment_simple():
    """Test moment head with simpler config"""
    from models import MomentHead
    
    moment_head = MomentHead(
        d_in=64,
        d_out=128,
        use_third_order=False  # Disable third order for now
    )
    
    batch_size = 2
    num_patches = 10
    embed_dim = 64
    
    tokens = torch.randn(batch_size, num_patches, embed_dim)
    graph = torch.randn(batch_size, num_patches, num_patches)
    
    features = moment_head(tokens, graph)
    print(f"Moment head test passed: {features.shape}")

def test_model_simple():
    """Test model with minimal config"""
    from models import EGOMomentCLEViT
    
    model = EGOMomentCLEViT(
        num_classes=5,
        backbone_name='vit_tiny_patch16_224',
        pretrained=False,
        moment_d_out=128,
        use_third_order=False  # Disable third order
    )
    
    batch_size = 2
    anchor = torch.randn(batch_size, 3, 224, 224)
    positive = torch.randn(batch_size, 3, 224, 224)
    
    output = model(anchor, positive)
    print(f"Model test passed: {output['logits'].shape}")

if __name__ == "__main__":
    try:
        print("Testing GPF...")
        test_gpf()
        
        print("Testing Moment Head...")
        test_moment_simple()
        
        print("Testing Full Model...")
        test_model_simple()
        
        print("All simplified tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
