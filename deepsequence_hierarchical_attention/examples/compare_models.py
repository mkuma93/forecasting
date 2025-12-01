"""
Full Dataset Training & Model Comparison
==========================================

Trains Hierarchical Attention and LightGBM on the entire synthetic dataset
and compares their performance with comprehensive metrics and visualizations.

Usage:
    python compare_models.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Import from the package (adjust path to parent directory)
sys.path.insert(0, os.path.abspath('..'))
from deepsequence_hierarchical_attention import DeepSequencePWLHierarchical

# Try to import LightGBM, install if not available
try:
    import lightgbm as lgb
    print("✓ LightGBM already installed")
except ImportError:
    print("Installing LightGBM...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm"])
    import lightgbm as lgb
    print("✓ LightGBM installed successfully")


def generate_synthetic_data(n_samples=10000, n_skus=25, n_features=18, zero_rate=0.90, seed=42):
    """
    Generate synthetic time series data with trend, seasonality, and high sparsity.
    
    Args:
        n_samples: Number of samples to generate
        n_skus: Number of unique SKUs
        n_features: Total number of features
        zero_rate: Proportion of zero values (sparsity)
        seed: Random seed for reproducibility
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        sku_ids: SKU identifiers (n_samples,)
    """
    np.random.seed(seed)
    
    print(f"\nGenerating synthetic data...")
    print(f"  Samples: {n_samples:,}")
    print(f"  SKUs: {n_skus}")
    print(f"  Features: {n_features}")
    print(f"  Target zero rate: {zero_rate:.1%}")
    
    time_index = np.arange(n_samples)
    
    # Feature 0: Time (trend)
    X_trend = (time_index / n_samples).reshape(-1, 1)
    
    # Features 1-10: Fourier features (seasonal)
    X_seasonal = np.column_stack([
        np.sin(2 * np.pi * time_index / 7),    # Daily
        np.cos(2 * np.pi * time_index / 7),
        np.sin(2 * np.pi * time_index / 30),   # Monthly
        np.cos(2 * np.pi * time_index / 30),
        np.sin(2 * np.pi * time_index / 91),   # Quarterly
        np.cos(2 * np.pi * time_index / 91),
        np.sin(2 * np.pi * time_index / 182),  # Semi-annual
        np.cos(2 * np.pi * time_index / 182),
        np.sin(2 * np.pi * time_index / 365),  # Annual
        np.cos(2 * np.pi * time_index / 365)
    ])
    
    # Features 11-15: Holiday distance features
    X_holiday = np.random.exponential(scale=30, size=(n_samples, 5))
    
    # Features 16-17: Lag features (regressor)
    X_regressor = np.random.randn(n_samples, 2)
    
    X = np.column_stack([X_trend, X_seasonal, X_holiday, X_regressor])
    
    # Generate SKU IDs
    sku_ids = np.random.randint(0, n_skus, n_samples)
    
    # Generate target with trend and seasonality
    trend = 0.01 * time_index / 100
    seasonality = 5 * np.sin(2 * np.pi * time_index / 365)
    noise = np.random.randn(n_samples) * 2
    
    y_magnitude = np.maximum(0, 10 + trend + seasonality + X[:, :6].sum(axis=1) * 0.5 + noise)
    
    # Apply sparsity (intermittent demand)
    zero_mask = np.random.rand(n_samples) < zero_rate
    y = y_magnitude.copy()
    y[zero_mask] = 0
    
    print(f"\n✓ Dataset created:")
    print(f"  Actual zero rate: {(y == 0).mean():.1%}")
    print(f"  Non-zero mean: {y[y > 0].mean():.2f}")
    print(f"  Non-zero std: {y[y > 0].std():.2f}")
    
    return X, y, sku_ids


def train_hierarchical_attention(X, y, sku_ids, n_skus, n_features, epochs=5, batch_size=256):
    """
    Train Hierarchical Attention model on full dataset.
    
    Args:
        X: Feature matrix
        y: Target values
        sku_ids: SKU identifiers
        n_skus: Number of unique SKUs
        n_features: Number of features
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        model: Trained Keras model
        y_pred: Predictions on full dataset
        training_time: Time taken for training
    """
    import time
    
    print("\n" + "="*80)
    print("TRAINING HIERARCHICAL ATTENTION MODEL")
    print("="*80)
    
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    # Build model
    model_builder = DeepSequencePWLHierarchical(
        num_skus=n_skus,
        n_features=n_features,
        enable_intermittent_handling=True,
        id_embedding_dim=8,
        component_hidden_units=32,
        component_dropout=0.2,
        zero_prob_hidden_units=64,
        zero_prob_hidden_layers=2,
        zero_prob_dropout=0.2,
        activation='mish'
    )
    
    main_model, _, _, _, _ = model_builder.build_model(
        trend_feature_indices=[0],
        seasonal_feature_indices=list(range(1, 11)),
        holiday_feature_indices=list(range(11, 16)),
        regressor_feature_indices=[16, 17]
    )
    
    main_model.compile(
        optimizer='adam',
        loss={'final_forecast': 'mae'},
        metrics={'final_forecast': ['mae']}
    )
    
    print(f"\n✓ Model built")
    print(f"  Total parameters: {main_model.count_params():,}")
    print(f"  Architecture: TabNet + Cross Layers + Intermittent Handling")
    
    # Train model
    print(f"\nTraining on {len(X):,} samples for {epochs} epochs...")
    start_time = time.time()
    
    history = main_model.fit(
        [X, sku_ids],
        {'final_forecast': y},
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time:.2f} seconds")
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = main_model.predict([X, sku_ids], verbose=0)
    y_pred = predictions['final_forecast'].flatten()
    
    return main_model, y_pred, training_time


def train_lightgbm(X, y, sku_ids, num_boost_round=100):
    """
    Train LightGBM model on full dataset.
    
    Args:
        X: Feature matrix
        y: Target values
        sku_ids: SKU identifiers
        num_boost_round: Number of boosting iterations
        
    Returns:
        model: Trained LightGBM model
        y_pred: Predictions on full dataset
        training_time: Time taken for training
    """
    import time
    
    print("\n" + "="*80)
    print("TRAINING LIGHTGBM MODEL")
    print("="*80)
    
    # Prepare data (add SKU ID as feature)
    X_lgb = np.column_stack([X, sku_ids])
    
    # Configure LightGBM
    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    print(f"\n✓ Configuration:")
    print(f"  Features: {X_lgb.shape[1]} (original + SKU ID)")
    print(f"  Boosting rounds: {num_boost_round}")
    print(f"  Learning rate: {lgb_params['learning_rate']}")
    
    # Train model
    print(f"\nTraining on {len(X):,} samples...")
    start_time = time.time()
    
    lgb_train = lgb.Dataset(X_lgb, label=y)
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=num_boost_round,
        callbacks=[lgb.log_evaluation(period=20)]
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time:.2f} seconds")
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred = lgb_model.predict(X_lgb)
    
    return lgb_model, y_pred, training_time


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model for display
        
    Returns:
        metrics: Dictionary of metric values
    """
    mae = np.abs(y_true - y_pred).mean()
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    
    # Additional metrics for sparse data
    zero_mask = y_true == 0
    nonzero_mask = ~zero_mask
    
    mae_nonzero = np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]).mean() if nonzero_mask.any() else 0
    
    metrics = {
        'MAE': mae,
        'MAPE (%)': mape,
        'R²': r2,
        'MAE (non-zero only)': mae_nonzero
    }
    
    print(f"\n{model_name} Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics


def compare_models(metrics_ha, metrics_lgb, time_ha, time_lgb, params_ha):
    """
    Print comparison table between models.
    
    Args:
        metrics_ha: Metrics for Hierarchical Attention
        metrics_lgb: Metrics for LightGBM
        time_ha: Training time for Hierarchical Attention
        time_lgb: Training time for LightGBM
        params_ha: Number of parameters in Hierarchical Attention
    """
    print("\n" + "="*90)
    print("FULL DATASET PERFORMANCE COMPARISON")
    print("="*90)
    
    comparison_df = pd.DataFrame({
        'Model': ['Hierarchical Attention\n(TabNet + Cross Layers)', 'LightGBM\n(Gradient Boosting)'],
        'MAE': [metrics_ha['MAE'], metrics_lgb['MAE']],
        'MAPE (%)': [metrics_ha['MAPE (%)'], metrics_lgb['MAPE (%)']],
        'R²': [metrics_ha['R²'], metrics_lgb['R²']],
        'Training Time (s)': [f"{time_ha:.2f}", f"{time_lgb:.2f}"],
        'Parameters': [f"{params_ha:,}", "N/A (tree-based)"]
    })
    
    print(comparison_df.to_string(index=False))
    print("="*90)
    
    # Calculate improvements
    mae_improvement = ((metrics_lgb['MAE'] - metrics_ha['MAE']) / metrics_lgb['MAE']) * 100
    mape_improvement = ((metrics_lgb['MAPE (%)'] - metrics_ha['MAPE (%)']) / metrics_lgb['MAPE (%)']) * 100
    r2_improvement = ((metrics_ha['R²'] - metrics_lgb['R²']) / abs(metrics_lgb['R²'])) * 100 if metrics_lgb['R²'] != 0 else 0
    
    print(f"\nHierarchical Attention vs LightGBM:")
    print(f"  MAE:  {mae_improvement:+.1f}% {'(better)' if mae_improvement > 0 else '(worse)'}")
    print(f"  MAPE: {mape_improvement:+.1f}% {'(better)' if mape_improvement > 0 else '(worse)'}")
    print(f"  R²:   {r2_improvement:+.1f}% {'(better)' if r2_improvement > 0 else '(worse)'}")
    print(f"  Training Speed: {time_lgb/time_ha:.1f}x faster" if time_lgb < time_ha else f"  Training Speed: {time_ha/time_lgb:.1f}x slower")


def visualize_comparison(y_true, y_pred_ha, y_pred_lgb, metrics_ha, metrics_lgb, save_path=None):
    """
    Create comparison visualizations.
    
    Args:
        y_true: True values
        y_pred_ha: Hierarchical Attention predictions
        y_pred_lgb: LightGBM predictions
        metrics_ha: Metrics for Hierarchical Attention
        metrics_lgb: Metrics for LightGBM
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Sample 500 points for visualization
    sample_indices = np.random.choice(len(y_true), min(500, len(y_true)), replace=False)
    y_sample = y_true[sample_indices]
    y_pred_ha_sample = y_pred_ha[sample_indices]
    y_pred_lgb_sample = y_pred_lgb[sample_indices]
    
    # 1. Scatter plot: Actual vs Predicted (Hierarchical Attention)
    axes[0].scatter(y_sample, y_pred_ha_sample, alpha=0.5, s=20, color='#2E86AB')
    axes[0].plot([y_sample.min(), y_sample.max()], [y_sample.min(), y_sample.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual', fontweight='bold')
    axes[0].set_ylabel('Predicted (Hierarchical Attention)', fontweight='bold')
    axes[0].set_title(f'Hierarchical Attention\nMAE: {metrics_ha["MAE"]:.4f}', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Scatter plot: Actual vs Predicted (LightGBM)
    axes[1].scatter(y_sample, y_pred_lgb_sample, alpha=0.5, s=20, color='#A23B72')
    axes[1].plot([y_sample.min(), y_sample.max()], [y_sample.min(), y_sample.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual', fontweight='bold')
    axes[1].set_ylabel('Predicted (LightGBM)', fontweight='bold')
    axes[1].set_title(f'LightGBM\nMAE: {metrics_lgb["MAE"]:.4f}', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Error distribution comparison
    errors_ha = y_sample - y_pred_ha_sample
    errors_lgb = y_sample - y_pred_lgb_sample
    
    axes[2].hist(errors_ha, bins=50, alpha=0.5, label='Hierarchical Attention', color='#2E86AB')
    axes[2].hist(errors_lgb, bins=50, alpha=0.5, label='LightGBM', color='#A23B72')
    axes[2].axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
    axes[2].set_xlabel('Prediction Error (Actual - Predicted)', fontweight='bold')
    axes[2].set_ylabel('Frequency', fontweight='bold')
    axes[2].set_title('Error Distribution Comparison', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {save_path}")
    
    plt.show()
    print("\n✓ Comparison visualization complete!")


def main():
    """Main execution function."""
    print("="*80)
    print("HIERARCHICAL ATTENTION vs LIGHTGBM: FULL DATASET COMPARISON")
    print("="*80)
    print("\nThis script compares two forecasting models on synthetic data:")
    print("  1. Hierarchical Attention (TabNet + Cross Layers + Intermittent Handling)")
    print("  2. LightGBM (Gradient Boosting Trees)")
    print("\nNote: Using randomly generated synthetic data for demonstration.")
    
    # Generate data
    X, y, sku_ids = generate_synthetic_data(
        n_samples=10000,
        n_skus=25,
        n_features=18,
        zero_rate=0.90,
        seed=42
    )
    
    # Train Hierarchical Attention
    model_ha, y_pred_ha, time_ha = train_hierarchical_attention(
        X, y, sku_ids,
        n_skus=25,
        n_features=18,
        epochs=5,
        batch_size=256
    )
    
    # Train LightGBM
    model_lgb, y_pred_lgb, time_lgb = train_lightgbm(
        X, y, sku_ids,
        num_boost_round=100
    )
    
    # Calculate metrics
    metrics_ha = calculate_metrics(y, y_pred_ha, "Hierarchical Attention")
    metrics_lgb = calculate_metrics(y, y_pred_lgb, "LightGBM")
    
    # Compare models
    compare_models(
        metrics_ha, metrics_lgb,
        time_ha, time_lgb,
        params_ha=model_ha.count_params()
    )
    
    # Visualize
    visualize_comparison(
        y, y_pred_ha, y_pred_lgb,
        metrics_ha, metrics_lgb,
        save_path='model_comparison.png'
    )
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
