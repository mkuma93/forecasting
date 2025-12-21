"""
DeepSequence PWL Implementation

Piecewise Linear (PWL) calibration-based components for intermittent demand forecasting.

Note: Old PWL components (trend, seasonal, holiday, regressor) use tf_keras and are commented out.
For new projects, use the hierarchical_attention module which uses tensorflow.keras.
"""

# Old PWL components (commented out - use hierarchical_attention instead)
# from .trend_component import TrendComponent
# from .seasonal_component import SeasonalComponent
# from .holiday_component import HolidayComponent
# from .regressor_component import RegressorComponent
# from .intermittent_handler import IntermittentHandler
# from .combination_layer import CombinationLayer
# from .model import DeepSequencePWL
# from . import config
# from .utils import *

__version__ = "1.0.0"

# Use hierarchical_attention module for new implementations
# from src.deepsequence_pwl.hierarchical_attention.components import DeepSequencePWLHierarchical