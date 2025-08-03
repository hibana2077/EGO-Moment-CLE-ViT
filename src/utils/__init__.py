"""
Utils module for EGO-Moment-CLE-ViT

This module contains utility functions for:
- Mathematical operations (matrix operations, numerical stability)
- Visualization (plots, charts, feature visualization)
- General utilities (seed setting, model info, etc.)
"""

from .ops import (
    set_seed,
    count_parameters,
    get_model_info,
    print_model_info,
    half_vectorize_symmetric,
    matrix_sqrt_newton_schulz,
    matrix_power_eigen,
    check_psd,
    ensure_psd,
    normalize_graph,
    compute_graph_statistics,
    batch_trace,
    batch_logdet,
    cosine_similarity_matrix
)

from .viz import (
    plot_similarity_matrix,
    plot_graph_weights,
    plot_polynomial_coefficients,
    plot_feature_embeddings,
    plot_training_curves,
    plot_confusion_matrix,
    visualize_moment_features
)

__all__ = [
    # Mathematical operations
    'set_seed',
    'count_parameters',
    'get_model_info',
    'print_model_info',
    'half_vectorize_symmetric',
    'matrix_sqrt_newton_schulz',
    'matrix_power_eigen',
    'check_psd',
    'ensure_psd',
    'normalize_graph',
    'compute_graph_statistics',
    'batch_trace',
    'batch_logdet',
    'cosine_similarity_matrix',
    
    # Visualization
    'plot_similarity_matrix',
    'plot_graph_weights',
    'plot_polynomial_coefficients',
    'plot_feature_embeddings',
    'plot_training_curves',
    'plot_confusion_matrix',
    'visualize_moment_features',
]
