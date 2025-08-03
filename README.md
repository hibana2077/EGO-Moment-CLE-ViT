# EGO-Moment-CLE-ViT

A PyTorch implementation of EGO-Moment-CLE-ViT, combining CLE-ViT (dual-view self-supervised Vision Transformer) with EGO-style graph polynomial fusion and high-order moment pooling for fine-grained image classification.

## Architecture Overview

The model integrates three key components:
- **CLE-ViT Backbone**: Dual-stream Vision Transformer with anchor/positive view generation
- **Graph Polynomial Fusion (GPF)**: EGO-style learnable graph fusion with non-negative polynomial coefficients
- **Moment Head**: Graph-weighted high-order moment pooling with iSQRT-COV and Tensor-Sketch

## Project Structure

```
EGO-Moment-CLE-ViT/
├── src/
│   ├── models/
│   │   ├── gpf_kernel.py          # Graph Polynomial Fusion
│   │   ├── moment_head.py         # Moment pooling with iSQRT-COV
│   │   ├── cle_vit_backbone.py    # CLE-ViT dual-stream backbone
│   │   ├── classifier_head.py     # Classification head with fusion
│   │   ├── ego_moment_clevit.py   # Main model integration
│   │   └── __init__.py
│   ├── losses/
│   │   ├── triplet_loss.py        # Instance-level triplet loss
│   │   ├── kernel_alignment.py    # CKA-based alignment loss
│   │   └── __init__.py
│   ├── utils/
│   │   ├── ops.py                 # Matrix operations and utilities
│   │   ├── viz.py                 # Visualization tools
│   │   └── __init__.py
│   └── dataset/
│       └── ufgvc.py              # UFG dataset loader
├── configs/
│   └── ufg_base.yaml             # Base configuration
├── docs/
│   ├── note.md                   # Implementation notes
│   ├── abs.md                    # Abstract and theory
│   ├── spec.md                   # Technical specifications
│   ├── cle-vit.md               # CLE-ViT implementation guide
│   └── ego.md                    # EGO fusion guide
├── train.py                      # Training script
├── eval.py                       # Evaluation script
├── test_implementation.py        # Implementation tests
└── requirements.txt
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd EGO-Moment-CLE-ViT

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Implementation

```bash
# Run basic implementation tests
python test_implementation.py
```

### 3. Training

```bash
# Train with default configuration
python train.py --config configs/ufg_base.yaml

# Train with custom settings
python train.py --config configs/ufg_base.yaml --batch_size 32 --learning_rate 0.001
```

### 4. Evaluation

```bash
# Evaluate trained model
python eval.py --config configs/ufg_base.yaml --checkpoint path/to/checkpoint.pth
```

## Configuration

The model is configured via YAML files. Key sections include:

- `model`: Architecture parameters (backbone, GPF degrees, moment pooling settings)
- `training`: Training hyperparameters (learning rate, batch size, epochs)
- `data`: Data loading and augmentation settings
- `experiment`: Logging, checkpointing, and output directories

## Key Features

### Graph Polynomial Fusion (GPF)
- Learnable non-negative polynomial coefficients
- Hadamard powers for element-wise graph operations  
- PSD enforcement for stability

### Moment Pooling
- Graph-weighted second and third-order moment computation
- iSQRT-COV for robust covariance estimation
- Tensor-Sketch for efficient high-dimensional operations

### CLE-ViT Backbone
- Dual-stream processing with anchor/positive views
- Positive view generation via masking and shuffling
- Support for various ViT architectures (Swin, DeiT, etc.)

### Multi-view Learning
- Instance-level triplet loss for feature alignment
- Kernel alignment loss (CKA) for representation consistency
- Hierarchical feature fusion

## Model Variants

The implementation supports multiple configurations:

- **Adaptive GPF**: Dynamic polynomial degree selection
- **Multi-scale Classifier**: Multiple fusion strategies
- **Third-order Moments**: Optional higher-order statistics
- **Various Backbones**: Support for different ViT architectures

## Dependencies

- PyTorch 2.2+
- timm (for pre-trained models)
- einops (for tensor operations)
- scikit-learn (for evaluation metrics)
- matplotlib/seaborn (for visualization)
- PyYAML (for configuration)

## Dataset

The implementation uses the UFG dataset format. The dataset loader (`src/dataset/ufgvc.py`) handles:
- Multiple split support (train/val/test)
- Automatic data downloading
- Flexible augmentation pipelines
- Class balancing options

## Evaluation

The evaluation script provides:
- Standard classification metrics (accuracy, per-class accuracy)
- Feature visualization (t-SNE, UMAP)
- Confusion matrices
- Graph weight analysis
- Polynomial coefficient visualization
- Ablation study support

## Implementation Notes

See `docs/note.md` for detailed implementation notes including:
- Architecture design decisions
- Key implementation details
- Module interactions
- Performance considerations

## Citation

If you use this code, please cite the relevant papers for CLE-ViT, EGO, and moment pooling methods.

## License

Please refer to the individual component licenses and ensure compliance with all requirements.