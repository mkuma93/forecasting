"""
Hierarchical Attention Architecture for Intermittent Demand Forecasting.

Two main implementations:
1. TabNet-based Hierarchical Attention (components.py) - Full feature attention with TabNet encoder
2. Lightweight Hierarchical Attention (components_lightweight.py) - Optimized for production
"""

# Loss functions
from .losses import composite_loss, weighted_composite_loss, mae_loss

# Lightweight implementation (optimized for production)
from .components_lightweight import build_hierarchical_model_lightweight

# TabNet-based hierarchical attention (full feature attention)
from .components import (
    DeepSequencePWLHierarchical,
    HierarchicalAttentionIntermittentHandler,
    TrendComponentBuilder,
    SeasonalComponentBuilder,
    HolidayComponentBuilder,
    RegressorComponentBuilder
)

# TabNet encoder components
from .tabnet import (
    TabNetEncoder,
    GhostBatchNormalization,
    GLUBlock,
    AttentiveTransformer,
    FeatureTransformer
)

# Model creation utilities
from .model import (
    create_hierarchical_model,
    compile_hierarchical_model,
    get_training_callbacks
)

__all__ = [
    # Loss functions
    'composite_loss',
    'weighted_composite_loss',
    'mae_loss',
    
    # Lightweight implementation
    'build_hierarchical_model_lightweight',
    
    # TabNet-based hierarchical attention
    'DeepSequencePWLHierarchical',
    'HierarchicalAttentionIntermittentHandler',
    'TrendComponentBuilder',
    'SeasonalComponentBuilder',
    'HolidayComponentBuilder',
    'RegressorComponentBuilder',
    
    # TabNet components
    'TabNetEncoder',
    'GhostBatchNormalization',
    'GLUBlock',
    'AttentiveTransformer',
    'FeatureTransformer',
    
    # Model utilities
    'create_hierarchical_model',
    'compile_hierarchical_model',
    'get_training_callbacks'
]
