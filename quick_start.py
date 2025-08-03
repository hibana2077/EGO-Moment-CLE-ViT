"""
Quick start script for EGO-Moment-CLE-ViT

This script demonstrates basic usage and provides a quick way to get started.
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import EGOMomentCLEViT, CLEViTDataTransforms
from utils import set_seed, get_model_info


def quick_demo():
    """Run a quick demonstration of the model"""
    print("🚀 EGO-Moment-CLE-ViT Quick Start Demo")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load configuration
    config_path = Path("configs/ufg_base.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration loaded")
    else:
        print("⚠️  Config file not found, using defaults")
        config = {}
    
    # Create model
    print("\n📦 Creating Model...")
    model = EGOMomentCLEViT(
        num_classes=10,
        backbone_name='vit_tiny_patch16_224',
        pretrained=False  # Set to True for pretrained weights
    )
    
    # Model info
    info = get_model_info(model)
    print(f"✓ Model created with {info['total_params']:,} parameters")
    print(f"  - Trainable: {info['trainable_params']:,}")
    print(f"  - Model size: {info['model_size_mb']:.1f} MB")
    
    # Create dummy data
    print("\n🎯 Testing Forward Pass...")
    batch_size = 4
    anchor = torch.randn(batch_size, 3, 224, 224)
    positive = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 10, (batch_size,))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(anchor, positive, labels=labels, return_features=True)
    
    print(f"✓ Forward pass successful")
    print(f"  - Logits shape: {output['logits'].shape}")
    print(f"  - Features available: {list(output['features'].keys())}")
    
    # Show loss components if available
    if 'loss_dict' in output:
        print(f"  - Loss components:")
        for loss_name, loss_value in output['loss_dict'].items():
            print(f"    • {loss_name}: {loss_value.item():.4f}")
    
    print("\n🎨 Data Transforms...")
    transforms = CLEViTDataTransforms(
        input_size=224,
        is_training=True
    )
    print("✓ Data transforms created")
    
    print("\n✨ Quick demo completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python train.py --config configs/ufg_base.yaml' to train")
    print("2. Run 'python eval.py --config configs/ufg_base.yaml --checkpoint path/to/checkpoint.pth' to evaluate")
    print("3. Check 'docs/note.md' for implementation details")


def show_architecture():
    """Show model architecture overview"""
    print("\n🏗️  Architecture Overview")
    print("=" * 50)
    
    model = EGOMomentCLEViT(
        num_classes=10,
        backbone_name='vit_tiny_patch16_224',
        pretrained=False
    )
    
    # Print key components
    print("📋 Key Components:")
    print(f"  1. Backbone: {model.backbone}")
    print(f"  2. GPF: {model.gpf}")
    print(f"  3. Moment Head: {model.moment_head}")
    print(f"  4. Classifier: {model.classifier}")
    
    # Show configuration
    print(f"\n⚙️  Configuration:")
    print(f"  - GPF degrees: (p={model.gpf.degree_p}, q={model.gpf.degree_q})")
    print(f"  - Moment output dim: {model.moment_head.d_out}")
    print(f"  - Use third-order: {model.moment_head.use_third_order}")
    print(f"  - Classifier fusion: {model.classifier.fusion_type}")


def check_requirements():
    """Check if all required packages are available"""
    print("\n🔍 Checking Requirements...")
    print("=" * 50)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('timm', 'timm (Torch Image Models)'),
        ('einops', 'einops'),
        ('yaml', 'PyYAML'),
        ('numpy', 'NumPy'),
    ]
    
    optional_packages = [
        ('sklearn', 'scikit-learn (for evaluation)'),
        ('matplotlib', 'matplotlib (for visualization)'),
        ('seaborn', 'seaborn (for plotting)'),
    ]
    
    all_good = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - REQUIRED")
            all_good = False
    
    print("\nOptional packages:")
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"○ {name} - optional")
    
    if all_good:
        print("\n🎉 All required packages are available!")
    else:
        print("\n⚠️  Some required packages are missing. Please install them.")
        print("Run: pip install -r requirements.txt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EGO-Moment-CLE-ViT Quick Start")
    parser.add_argument('--demo', action='store_true', help='Run quick demo')
    parser.add_argument('--arch', action='store_true', help='Show architecture overview')
    parser.add_argument('--check', action='store_true', help='Check requirements')
    
    args = parser.parse_args()
    
    if args.check or (not args.demo and not args.arch):
        check_requirements()
    
    if args.arch or (not args.demo and not args.check):
        show_architecture()
    
    if args.demo or (not args.arch and not args.check):
        quick_demo()
