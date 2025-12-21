"""Utilities for saving/loading models with adaptive loss weights."""

from tensorflow import keras
import numpy as np


def save_model_with_uncertainties(model, filepath):
    """
    Save model along with its learnable uncertainty weights.
    
    Args:
        model: Keras model with forecast_uncertainty and classification_uncertainty
        filepath: Base path (without extension) to save model
    """
    # Save the base model architecture and weights
    model.save(f'{filepath}.keras')
    
    # Save uncertainty weights separately
    if hasattr(model, 'forecast_uncertainty'):
        forecast_log_var = model.forecast_uncertainty.log_var.numpy()
        class_log_var = model.classification_uncertainty.log_var.numpy()
        
        np.savez(
            f'{filepath}_uncertainties.npz',
            forecast_log_var=forecast_log_var,
            classification_log_var=class_log_var
        )
        
        print(f"✓ Model saved to {filepath}.keras")
        print(f"✓ Uncertainties saved to {filepath}_uncertainties.npz")
        print(f"  Forecast σ: {model.forecast_uncertainty.get_uncertainty().numpy():.6f}")
        print(f"  Classification σ: {model.classification_uncertainty.get_uncertainty().numpy():.6f}")


def load_model_with_uncertainties(filepath):
    """
    Load model and restore its learnable uncertainty weights.
    
    Args:
        filepath: Base path (without extension) to load model from
    
    Returns:
        Keras model with restored uncertainties
    """
    from deepsequence_hierarchical_attention.components_lightweight import (
        LearnableUncertaintyWeight
    )
    
    # Load base model
    model = keras.models.load_model(f'{filepath}.keras')
    
    # Check if this model has uncertainties
    try:
        uncertainties = np.load(f'{filepath}_uncertainties.npz')
        
        # Recreate uncertainty layers
        model.forecast_uncertainty = LearnableUncertaintyWeight(
            initial_log_var=float(uncertainties['forecast_log_var']),
            name='forecast_uncertainty'
        )
        model.classification_uncertainty = LearnableUncertaintyWeight(
            initial_log_var=float(uncertainties['classification_log_var']),
            name='classification_uncertainty'
        )
        
        # Build and set weights
        model.forecast_uncertainty.build(())
        model.classification_uncertainty.build(())
        
        model.forecast_uncertainty.log_var.assign(
            uncertainties['forecast_log_var']
        )
        model.classification_uncertainty.log_var.assign(
            uncertainties['classification_log_var']
        )
        
        print(f"✓ Model loaded from {filepath}.keras")
        print(f"✓ Uncertainties restored from {filepath}_uncertainties.npz")
        print(f"  Forecast σ: {model.forecast_uncertainty.get_uncertainty().numpy():.6f}")
        print(f"  Classification σ: {model.classification_uncertainty.get_uncertainty().numpy():.6f}")
        
    except FileNotFoundError:
        print(f"✓ Model loaded from {filepath}.keras (no uncertainties)")
    
    return model
