# Implementation Notes for EGO-Moment-CLE-ViT

## Project Overview
Based on the documentation analysis, this project aims to integrate:
1. **CLE-ViT (Contrastive Learning Encoded ViT)** - self-supervised learning with dual-view (anchor + positive with masking & shuffling)
2. **EGO (Graph Operator Expansion)** - polynomial fusion of relation graphs 
3. **High-order statistical moments** - second and third-order moment pooling
4. **UFG dataset** - Ultra-fine-grained visual categorization

## Key Implementation Components Needed

### 1. Model Architecture (src/models/)
- [ ] `cle_vit_backbone.py` - CLE-ViT backbone wrapper using timm
- [ ] `gpf_kernel.py` - Graph Polynomial Fusion (EGO-like) 
- [ ] `moment_head.py` - Weighted high-order moment pooling
- [ ] `classifier_head.py` - CLS + moment feature fusion
- [ ] `ego_moment_clevit.py` - Main model integration

### 2. Loss Functions (src/losses/)
- [ ] `triplet_loss.py` - Instance-level triplet loss for CLE-ViT
- [ ] `kernel_alignment.py` - Graph alignment regularization

### 3. Data Processing 
- [x] `src/dataset/ufgvc.py` - Already implemented UFG dataset
- [ ] Extend for CLE-ViT dual-view generation (anchor + positive with masking/shuffling)

### 4. Training Pipeline
- [ ] `train.py` - Main training script
- [ ] `eval.py` - Evaluation script
- [ ] Configuration management (YAML configs)

## Critical Implementation Details

### Graph Polynomial Fusion (GPF)
- Use softplus parameterization: A_pq = softplus(alpha_pq) to ensure A_pq >= 0
- Implement symmetric enforcement: G = 0.5 * (G + G.T) 
- Handle numerical stability in D^(-1/2) computation with eps clamping
- Polynomial degrees: start with P=Q=2 (9 terms)

### Moment Pooling Head
- **Second-order**: Weighted covariance with iSQRT-COV normalization
- **Third-order**: Tensor-Sketch/Compact Bilinear approximation 
- Use Newton-Schulz iteration (3-5 steps) for matrix square root
- Half-vectorization for symmetric matrices

### CLE-ViT Integration
- Dual-view generation: anchor (standard augment) + positive (mask + 4x4 shuffle)
- Mask ratio: α ∈ [0.15, 0.45] 
- Input size: resize 600 → crop 448
- Maintain original CLE-ViT losses: CE + triplet

### Mathematical Properties to Maintain
- **PSD preservation**: Schur product of PSD matrices remains PSD
- **Kernel validity**: G(A) represents valid kernel through polynomial expansion
- **Differentiability**: All operations must be differentiable for backprop

## Hyperparameters (Initial Settings)
- Backbone: Swin-B or ViT-Base from timm
- Batch size: 64 (with AMP)
- Learning rate: 3e-4 (backbone: 1e-4)
- Optimizer: AdamW with cosine scheduling
- Triplet margin: 0.3
- Alignment regularization: λ = 0.1
- Training epochs: 120 with 5-epoch warmup

## Implementation Priority
1. Core model components (GPF + Moment Head)
2. CLE-ViT backbone integration 
3. Loss functions and training loop
4. Evaluation and ablation studies
5. Visualization tools

## Potential Issues to Watch
- Memory usage with O(N²) operations (N=196 tokens)
- Numerical stability in matrix operations
- Gradient flow through complex polynomial operations
- Integration with existing CLE-ViT preprocessing

## Datasets Available (from ufgvc.py)
- cotton80: Cotton classification (80 classes)  
- soybean: Soybean classification
- soy_ageing_r1/r3/r4: Soybean aging datasets
- All datasets support automatic download from HuggingFace