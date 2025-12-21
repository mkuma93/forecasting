"""
Inference helpers for DeepSequence Hierarchical Attention.

Provides utilities to convert continuous non-zero probabilities into
thresholded decisions for reporting while keeping training differentiable.
"""

from typing import Tuple
import numpy as np

def threshold_gate(base_forecast: np.ndarray,
                   non_zero_probability: np.ndarray,
                   threshold: float = 0.5,
                   clip_negative: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a hard decision gate at inference time.

    - decision = 1 if P >= threshold else 0
    - final_forecast = decision * base_forecast

    Args:
        base_forecast: Array of predicted base magnitudes (shape [n,] or [n,1])
        non_zero_probability: Array of P(non-zero) in [0,1] (same shape)
        threshold: Decision threshold for classifying non-zero
        clip_negative: If True, clip final_forecast at 0 to avoid negatives

    Returns:
        final_forecast_thresh, decision
    """
    base = np.asarray(base_forecast).squeeze()
    p = np.asarray(non_zero_probability).squeeze()
    decision = (p >= threshold).astype(base.dtype)
    final_forecast = decision * base
    if clip_negative:
        final_forecast = np.maximum(final_forecast, 0.0)
    return final_forecast, decision

def smooth_gate(base_forecast: np.ndarray,
                non_zero_probability: np.ndarray,
                clip_negative: bool = True) -> np.ndarray:
    """
    Continuous gating used during training:
    final_forecast = P * base_forecast
    """
    base = np.asarray(base_forecast).squeeze()
    p = np.asarray(non_zero_probability).squeeze()
    final = p * base
    if clip_negative:
        final = np.maximum(final, 0.0)
    return final
