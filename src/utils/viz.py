"""
Visualization utilities for EGO-Moment-CLE-ViT

This module provides functions for visualizing model features, attention maps,
similarity matrices, and training progress.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import warnings

# Handle missing optional dependencies
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not available. Some visualization functions will be disabled.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("opencv-python not available. Some visualization functions will be disabled.")


def plot_similarity_matrix(
    similarity_matrix: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    title: str = "Similarity Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot similarity matrix with optional label annotations
    
    Args:
        similarity_matrix: [N, N] similarity matrix
        labels: [N] optional labels for annotation
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Convert to numpy
    if isinstance(similarity_matrix, torch.Tensor):
        sim_np = similarity_matrix.detach().cpu().numpy()
    else:
        sim_np = similarity_matrix
    
    # Create heatmap
    sns.heatmap(
        sim_np,
        annot=False,
        cmap='RdBu_r',
        center=0,
        square=True,
        cbar_kws={'label': 'Similarity'}
    )
    
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    
    # Add label annotations if provided
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = labels
            
        # Add color bar for different classes
        unique_labels = np.unique(labels_np)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        # Add colored patches to indicate class boundaries
        for i, label in enumerate(unique_labels):
            indices = np.where(labels_np == label)[0]
            if len(indices) > 0:
                plt.axhline(y=indices[0], color=colors[i], linewidth=2, alpha=0.7)
                plt.axhline(y=indices[-1]+1, color=colors[i], linewidth=2, alpha=0.7)
                plt.axvline(x=indices[0], color=colors[i], linewidth=2, alpha=0.7)
                plt.axvline(x=indices[-1]+1, color=colors[i], linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_graph_weights(
    graph: torch.Tensor,
    spatial_layout: Optional[Tuple[int, int]] = None,
    title: str = "Graph Weights",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Visualize graph weights as spatial connectivity patterns
    
    Args:
        graph: [N, N] or [B, N, N] graph weight matrix
        spatial_layout: (H, W) spatial layout of tokens (e.g., (14, 14) for ViT)
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    if isinstance(graph, torch.Tensor):
        graph_np = graph.detach().cpu().numpy()
    else:
        graph_np = graph
    
    # Handle batch dimension
    if graph_np.ndim == 3:
        graph_np = graph_np[0]  # Take first sample
    
    N = graph_np.shape[0]
    
    if spatial_layout is None:
        # Assume square layout
        H = W = int(np.sqrt(N))
        if H * W != N:
            # Fall back to matrix visualization
            plot_similarity_matrix(graph_np, title=title, save_path=save_path, figsize=figsize)
            return
    else:
        H, W = spatial_layout
        if H * W != N:
            raise ValueError(f"Spatial layout {spatial_layout} doesn't match graph size {N}")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Full adjacency matrix
    sns.heatmap(graph_np, ax=axes[0, 0], cmap='viridis', square=True)
    axes[0, 0].set_title(f'{title} - Full Matrix')
    
    # 2. Average connectivity per spatial position
    connectivity_strength = graph_np.sum(axis=1).reshape(H, W)
    im1 = axes[0, 1].imshow(connectivity_strength, cmap='hot', interpolation='nearest')
    axes[0, 1].set_title('Average Connectivity Strength')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 3. Connectivity pattern for center token
    center_idx = N // 2
    center_pattern = graph_np[center_idx].reshape(H, W)
    im2 = axes[1, 0].imshow(center_pattern, cmap='Blues', interpolation='nearest')
    axes[1, 0].set_title(f'Connectivity from Center Token (idx {center_idx})')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 4. Eigenvalue spectrum
    eigenvals = np.linalg.eigvals(graph_np).real
    eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
    axes[1, 1].plot(eigenvals, 'o-', markersize=4)
    axes[1, 1].set_title('Eigenvalue Spectrum')
    axes[1, 1].set_xlabel('Eigenvalue Index')
    axes[1, 1].set_ylabel('Eigenvalue')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_polynomial_coefficients(
    coefficients: torch.Tensor,
    title: str = "Polynomial Coefficients",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Visualize learned polynomial coefficients from GPF
    
    Args:
        coefficients: [P+1, Q+1] coefficient matrix A_pq
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    if isinstance(coefficients, torch.Tensor):
        coeff_np = coefficients.detach().cpu().numpy()
    else:
        coeff_np = coefficients
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        coeff_np,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        square=True,
        cbar_kws={'label': 'Coefficient Value'}
    )
    
    plt.title(title)
    plt.xlabel('Degree q (Positive View)')
    plt.ylabel('Degree p (Anchor View)')
    
    # Add polynomial term labels
    P, Q = coeff_np.shape
    for p in range(P):
        for q in range(Q):
            plt.text(q + 0.5, p + 0.5, f'R_a^{p}⊙R_p^{q}', 
                    ha='center', va='bottom', fontsize=8, color='blue')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_embeddings(
    features: torch.Tensor,
    labels: torch.Tensor,
    method: str = 'tsne',
    title: str = "Feature Embeddings",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot 2D embedding of high-dimensional features
    
    Args:
        features: [N, D] feature vectors
        labels: [N] class labels
        method: Dimensionality reduction method ('tsne', 'pca')
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    if not HAS_SKLEARN:
        print("sklearn not available. Cannot plot feature embeddings.")
        return
    
    # Convert to numpy
    if isinstance(features, torch.Tensor):
        features_np = features.detach().cpu().numpy()
    else:
        features_np = features
        
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_np) - 1))
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings_2d = reducer.fit_transform(features_np)
    
    # Plot
    plt.figure(figsize=figsize)
    
    unique_labels = np.unique(labels_np)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.7,
            s=50
        )
    
    plt.title(f'{title} ({method.upper()})')
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot training and validation curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        train_accs: List of training accuracies (optional)
        val_accs: List of validation accuracies (optional)
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    n_plots = 1 + (train_accs is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses is not None:
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracies if available
    if train_accs is not None:
        axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
        if val_accs is not None:
            axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
        
        axes[1].set_title('Accuracy Curves')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        normalize: Whether to normalize the matrix
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    if not HAS_SKLEARN:
        print("sklearn not available. Cannot plot confusion matrix.")
        return
    
    from sklearn.metrics import confusion_matrix
    
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        square=True,
        xticklabels=class_names if class_names else range(cm.shape[1]),
        yticklabels=class_names if class_names else range(cm.shape[0])
    )
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_moment_features(
    moment_features: torch.Tensor,
    labels: torch.Tensor,
    feature_type: str = "moment",
    save_path: Optional[str] = None
):
    """
    Comprehensive visualization of moment features
    
    Args:
        moment_features: [B, D] moment feature vectors
        labels: [B] class labels
        feature_type: Type of features for title
        save_path: Base path for saving plots
    """
    print(f"Visualizing {feature_type} features...")
    
    # Feature statistics
    print(f"Feature shape: {moment_features.shape}")
    print(f"Feature range: [{moment_features.min().item():.3f}, {moment_features.max().item():.3f}]")
    print(f"Feature mean: {moment_features.mean().item():.3f}")
    print(f"Feature std: {moment_features.std().item():.3f}")
    
    # Plot feature distribution
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(moment_features.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.title(f'{feature_type.capitalize()} Feature Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    feature_norms = torch.norm(moment_features, p=2, dim=1)
    plt.hist(feature_norms.detach().cpu().numpy(), bins=20, alpha=0.7)
    plt.title('Feature L2 Norms')
    plt.xlabel('L2 Norm')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_{feature_type}_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # t-SNE visualization
    if HAS_SKLEARN and len(moment_features) >= 5:
        plot_feature_embeddings(
            moment_features,
            labels,
            method='tsne',
            title=f'{feature_type.capitalize()} Features (t-SNE)',
            save_path=f"{save_path}_{feature_type}_tsne.png" if save_path else None
        )


def test_visualization():
    """Test visualization functions"""
    print("Testing Visualization Functions...")
    
    # Create dummy data
    batch_size = 50
    num_classes = 5
    feature_dim = 128
    graph_size = 16
    
    # Random features and labels
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Random similarity matrix
    similarity = torch.randn(batch_size, batch_size)
    similarity = (similarity + similarity.t()) / 2  # Make symmetric
    
    # Random graph
    graph = torch.randn(graph_size, graph_size)
    graph = torch.mm(graph, graph.t())  # Make PSD
    
    # Random polynomial coefficients
    coeffs = torch.rand(3, 3)
    
    print("Testing similarity matrix plot...")
    plot_similarity_matrix(similarity[:10, :10], labels[:10], title="Test Similarity")
    
    print("Testing graph weights plot...")
    plot_graph_weights(graph, spatial_layout=(4, 4), title="Test Graph")
    
    print("Testing polynomial coefficients plot...")
    plot_polynomial_coefficients(coeffs, title="Test Coefficients")
    
    if HAS_SKLEARN:
        print("Testing feature embeddings plot...")
        plot_feature_embeddings(features, labels, method='pca', title="Test Features")
    
    print("Testing training curves plot...")
    train_losses = [1.0 - 0.1 * i + 0.05 * np.sin(i) for i in range(20)]
    val_losses = [1.1 - 0.08 * i + 0.03 * np.sin(i + 1) for i in range(20)]
    train_accs = [20 + 3 * i + 2 * np.sin(i) for i in range(20)]
    val_accs = [18 + 2.5 * i + 1.5 * np.sin(i + 1) for i in range(20)]
    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    print("Visualization tests completed!")


if __name__ == "__main__":
    test_visualization()
