"""
Hierarchical Attention Architecture for Intermittent Demand Forecasting.

This module implements a sophisticated three-level attention mechanism:

Level 1 (PWL-level): Attention on piecewise linear calibration outputs
    - Trend: Attention on changepoint features (which time regions matter)
    - Holiday: Attention on distance range features (which distances from holidays matter)

Level 2 (Feature-level): Attention on hidden features within each component
    - Applied to all components (trend, seasonal, holiday, regressor)
    - Selects which hidden dimensions are important per SKU

Level 3 (Component-level): Attention across all components
    - Learns importance of trend vs seasonal vs holiday vs regressor
    - Different SKUs can have different component importance

Key Features:
- TensorFlow-native sparsemax activation (can output exact zeros)
- PWL calibration for non-linear transformations
- SKU-specific attention through embeddings
- Hierarchical sparsity for interpretability
"""

from .components import (
    Entmax15,
    TrendComponentBuilder,
    SeasonalComponentBuilder,
    HolidayComponentBuilder,
    RegressorComponentBuilder,
    HierarchicalAttentionIntermittentHandler,
    DeepSequencePWLHierarchical
)

from .model import (
    create_hierarchical_model,
    compile_hierarchical_model,
    get_training_callbacks
)

__all__ = [
    'Entmax15',
    'TrendComponentBuilder',
    'SeasonalComponentBuilder',
    'HolidayComponentBuilder',
    'RegressorComponentBuilder',
    'HierarchicalAttentionIntermittentHandler',
    'DeepSequencePWLHierarchical',
    'create_hierarchical_model',
    'compile_hierarchical_model',
    'get_training_callbacks'
]
