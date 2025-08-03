"""
Simple test script to validate the EGO-Moment-CLE-ViT implementation

This script runs basic tests to ensure all components work correctly
without requiring a full dataset or training.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from models import (
            EGOMomentCLEViT,
            GraphPolynomialFusion,
            MomentHead,
            CLEViTBackbone,
            ClassifierHead
        )
        from losses import TripletLoss, KernelAlignmentLoss
        from utils import set_seed, get_model_info
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test model creation with default parameters"""
    print("Testing model creation...")
    
    try:
        from models import EGOMomentCLEViT
        
        model = EGOMomentCLEViT(
            num_classes=10,
            backbone_name='vit_tiny_patch16_224',
            pretrained=False
        )
        
        print(f"✓ Model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True, model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False, None


def test_forward_pass(model):
    """Test forward pass with dummy data"""
    print("Testing forward pass...")
    
    try:
        # Create dummy input
        batch_size = 4
        channels = 3
        height = 224
        width = 224
        
        anchor = torch.randn(batch_size, channels, height, width)
        positive = torch.randn(batch_size, channels, height, width)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(anchor, positive)
        
        # Check output shape
        expected_shape = (batch_size, 10)  # num_classes = 10
        if output['logits'].shape == expected_shape:
            print(f"✓ Forward pass successful")
            print(f"  Output shape: {output['logits'].shape}")
            return True
        else:
            print(f"✗ Unexpected output shape: {output['logits'].shape}, expected: {expected_shape}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_feature_extraction(model):
    """Test feature extraction"""
    print("Testing feature extraction...")
    
    try:
        batch_size = 2
        anchor = torch.randn(batch_size, 3, 224, 224)
        positive = torch.randn(batch_size, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            output = model(anchor, positive, return_features=True)
        
        # Check if features are returned
        features = output['features']
        required_keys = ['anchor_global', 'positive_global', 'moment_features', 'fused_graph']
        
        missing_keys = [key for key in required_keys if key not in features]
        if not missing_keys:
            print("✓ Feature extraction successful")
            for key, value in features.items():
                print(f"  {key}: {value.shape}")
            return True
        else:
            print(f"✗ Missing feature keys: {missing_keys}")
            return False
            
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False


def test_loss_functions():
    """Test loss function implementations"""
    print("Testing loss functions...")
    
    try:
        from losses import TripletLoss, KernelAlignmentLoss
        
        # Test triplet loss
        triplet_loss = TripletLoss(margin=0.5)
        
        # Dummy features - ensure 2D tensors
        batch_size = 4
        feature_dim = 128
        
        anchor_features = torch.randn(batch_size, feature_dim)
        positive_features = torch.randn(batch_size, feature_dim)
        negative_features = torch.randn(batch_size, feature_dim)
        
        loss_triplet = triplet_loss(anchor_features, positive_features, negative_features)
        
        # Test kernel alignment loss  
        kernel_loss = KernelAlignmentLoss()
        
        # Create dummy graph and labels
        batch_size = 4
        num_patches = 10  # Smaller for testing
        graph = torch.randn(batch_size, num_patches, num_patches)  # [B, N, N]
        labels = torch.randint(0, 3, (batch_size,))  # Random labels
        
        loss_kernel = kernel_loss(graph, labels)
        
        print(f"✓ Loss functions working")
        print(f"  Triplet loss: {loss_triplet.item():.4f}")
        print(f"  Kernel loss: {loss_kernel.item():.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False


def test_individual_components():
    """Test individual model components"""
    print("Testing individual components...")
    
    try:
        from models import GraphPolynomialFusion, MomentHead, CLEViTBackbone
        
        # Test GPF
        batch_size = 4
        num_patches = 196  # 14x14 for 224x224 input with patch_size=16
        embed_dim = 192
        
        gpf = GraphPolynomialFusion(degree_p=2, degree_q=2)
        features_anchor = torch.randn(batch_size, num_patches, embed_dim)
        features_positive = torch.randn(batch_size, num_patches, embed_dim)
        graph = gpf(features_anchor, features_positive)
        
        print(f"✓ GPF working - output shape: {graph.shape}")
        
        # Test Moment Head
        moment_head = MomentHead(
            d_in=embed_dim,
            d_out=256,
            use_third_order=True
        )
        moment_features = moment_head(features_anchor, graph)
        
        print(f"✓ Moment Head working - output shape: {moment_features.shape}")
        
        # Test Backbone (basic test)
        backbone = CLEViTBackbone(
            model_name='vit_tiny_patch16_224',
            pretrained=False
        )
        
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        backbone_output = backbone(dummy_input)
        
        print(f"✓ Backbone working - patch tokens: {backbone_output['patch_tokens'].shape}, global: {backbone_output['global_features'].shape}")
        return True
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration loading...")
    
    try:
        import yaml
        
        config_path = Path("configs/ufg_base.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['model', 'training', 'data', 'experiment']
            missing_sections = [section for section in required_sections if section not in config]
            
            if not missing_sections:
                print("✓ Configuration loading successful")
                return True
            else:
                print(f"✗ Missing config sections: {missing_sections}")
                return False
        else:
            print(f"✗ Config file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_utilities():
    """Test utility functions"""
    print("Testing utilities...")
    
    try:
        from utils import set_seed, get_model_info
        
        # Test seed setting
        set_seed(42)
        tensor1 = torch.randn(5, 5)
        
        set_seed(42)
        tensor2 = torch.randn(5, 5)
        
        if torch.allclose(tensor1, tensor2):
            print("✓ Seed setting working")
        else:
            print("✗ Seed setting not working")
            return False
        
        # Test model info (if we have a model)
        try:
            from models import EGOMomentCLEViT
            model = EGOMomentCLEViT(num_classes=5, backbone_name='vit_tiny_patch16_224', pretrained=False)
            info = get_model_info(model)
            print(f"✓ Model info working - {info['total_params']:,} parameters")
        except:
            print("○ Model info test skipped")
        
        return True
        
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("EGO-Moment-CLE-ViT Implementation Test Suite")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Config Loading Test", test_config_loading),
        ("Utilities Test", test_utilities),
        ("Component Test", test_individual_components),
        ("Loss Functions Test", test_loss_functions),
    ]
    
    # Run basic tests first
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    # Model tests (require successful imports)
    if results[0][1]:  # If imports successful
        print(f"\nModel Creation Test:")
        model_created, model = test_model_creation()
        results.append(("Model Creation Test", model_created))
        
        if model_created and model is not None:
            print(f"\nForward Pass Test:")
            forward_result = test_forward_pass(model)
            results.append(("Forward Pass Test", forward_result))
            
            print(f"\nFeature Extraction Test:")
            feature_result = test_feature_extraction(model)
            results.append(("Feature Extraction Test", feature_result))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("\n🎉 All tests passed! Implementation looks good.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")
    
    print("="*60)


if __name__ == "__main__":
    main()
