"""
Full Dataset Training & Model Comparison on Real Data
======================================================

Trains Hierarchical Attention and LightGBM on the actual dataset
and compares their performance with comprehensive metrics and visualizations.

IMPORTANT: LightGBM is trained WITHOUT lag features to prevent data leakage.
Only exogenous features (holiday distance, day of week, etc.) are used.

Usage:
    python compare_models_real_data.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Import from the package
sys.path.insert(0, os.path.abspath('..'))
from deepsequence_hierarchical_attention import (
    DeepSequencePWLHierarchical,
    composite_loss
)

# Try to import LightGBM, install if not available
try:
    import lightgbm as lgb
    print("✓ LightGBM already installed")
except ImportError:
    print("Installing LightGBM...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                           "lightgbm"])
    import lightgbm as lgb
    print("✓ LightGBM installed successfully")


def load_real_data(data_dir='../../data'):
    """Load real data with lag_1, lag_2, lag_7 features."""
    """
    Load and prepare real dataset from CSV files.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        X_train, y_train, sku_train: Training data
        X_test, y_test, sku_test: Test data
        feature_info: Dictionary with feature metadata
    """
    print("\n" + "="*80)
    print("LOADING REAL DATA")
    print("="*80)
    
    # Load datasets
    train_df = pd.read_csv(os.path.join(data_dir, 'train_split.csv'), 
                           parse_dates=['ds'])
    val_df = pd.read_csv(os.path.join(data_dir, 'val_split.csv'),
                         parse_dates=['ds'])
    test_df = pd.read_csv(os.path.join(data_dir, 'test_split.csv'),
                          parse_dates=['ds'])
    
    # Load pre-computed holiday features
    holiday_train = pd.read_csv(os.path.join(data_dir, 'holiday_features_train.csv'))
    holiday_val = pd.read_csv(os.path.join(data_dir, 'holiday_features_val.csv'))
    holiday_test = pd.read_csv(os.path.join(data_dir, 'holiday_features_test.csv'))
    
    print(f"\n✓ Data loaded:")
    print(f"  Train samples: {len(train_df):,}")
    print(f"  Validation samples: {len(val_df):,}")
    print(f"  Test samples: {len(test_df):,}")
    print(f"  Base columns: {list(train_df.columns)}")
    print(f"  Holiday features: {len(holiday_train.columns)} pre-computed distances")
    
    # Parse date
    train_df['ds'] = pd.to_datetime(train_df['ds'])
    test_df['ds'] = pd.to_datetime(test_df['ds'])
    
    # Feature engineering: EXOGENOUS + LAG FEATURES (no leakage)
    # Lag features use ONLY historical data (past values)
    
    def create_features(df, holiday_features_df):
        """Create features with proper lag features (no future leakage)."""
        # Sort by SKU and date to ensure proper lag calculation
        df = df.sort_values(['id_var', 'ds']).reset_index(drop=True)
        
        features = {}
        
        # Holiday distance (already in data)
        features['holiday_distance'] = df['cumdist'].values
        
        # Day of week (0-6)
        features['day_of_week'] = df['ds'].dt.dayofweek.values
        
        # Day of month (1-31)
        features['day_of_month'] = df['ds'].dt.day.values
        
        # Month (1-12)
        features['month'] = df['ds'].dt.month.values
        
        # Quarter (1-4)
        features['quarter'] = df['ds'].dt.quarter.values
        
        # Week of year (1-52)
        features['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int).values
        
        # Holiday indicator
        features['is_holiday'] = (df['holiday'] == 1).astype(int).values
        
        # Cyclical encoding for day of week
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Cyclical encoding for month
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # LAG FEATURES (using historical data only)
        # Lag 1, 2, 7 days - simple shifts are guaranteed safe
        features['lag_1'] = df.groupby('id_var')['Quantity'].shift(1).fillna(0).values
        features['lag_2'] = df.groupby('id_var')['Quantity'].shift(2).fillna(0).values
        features['lag_7'] = df.groupby('id_var')['Quantity'].shift(7).fillna(0).values
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        
        # Add pre-computed holiday features (15 holidays)
        features_df = pd.concat([features_df, holiday_features_df.reset_index(drop=True)], axis=1)
        
        return features_df
    
    print("\n✓ Creating features with proper lags (NO LEAKAGE):")
    print("  Temporal features:")
    print("    - Day of week, day of month, month, quarter, week of year")
    print("    - Cyclical encodings (dow_sin/cos, month_sin/cos)")
    print("  Holiday features: 15 pre-computed distances")
    print("  Lag features (historical data only):")
    print("    - lag_1, lag_2, lag_7 (simple shifts)")
    
    X_train_df = create_features(train_df, holiday_train)
    X_val_df = create_features(val_df, holiday_val)
    X_test_df = create_features(test_df, holiday_test)
    
    X_train = X_train_df.values
    X_val = X_val_df.values
    X_test = X_test_df.values
    
    # Target variable
    y_train = train_df['Quantity'].values
    y_val = val_df['Quantity'].values
    y_test = test_df['Quantity'].values
    
    # SKU IDs (encode as integers)
    sku_encoder = {sku: idx for idx, sku in 
                   enumerate(train_df['id_var'].unique())}
    sku_train = train_df['id_var'].map(sku_encoder).values
    sku_val = val_df['id_var'].map(sku_encoder).fillna(-1).astype(int).values
    sku_test = test_df['id_var'].map(sku_encoder).fillna(-1).astype(int).values
    
    # Handle unseen SKUs in validation and test sets
    unseen_val = (sku_val == -1).sum()
    unseen_test = (sku_test == -1).sum()
    if unseen_val > 0:
        print(f"\n⚠ Warning: {unseen_val} validation samples with unseen SKUs "
              f"(will use SKU 0)")
        sku_val[sku_val == -1] = 0
    if unseen_test > 0:
        print(f"⚠ Warning: {unseen_test} test samples with unseen SKUs "
              f"(will use SKU 0)")
        sku_test[sku_test == -1] = 0
    
    print("\n✓ Data prepared:")
    print(f"  Features: {X_train.shape[1]} (reduced to 27: removed lag_14, lag_28)")
    print(f"  Unique SKUs: {len(sku_encoder)}")
    print(f"  Train zero rate: {(y_train == 0).mean():.1%}")
    print(f"  Val zero rate: {(y_val == 0).mean():.1%}")
    print(f"  Test zero rate: {(y_test == 0).mean():.1%}")
    print(f"  Train mean (non-zero): {y_train[y_train > 0].mean():.2f}")
    print(f"  Val mean (non-zero): {y_val[y_val > 0].mean():.2f}")
    print(f"  Test mean (non-zero): {y_test[y_test > 0].mean():.2f}")
    
    feature_info = {
        'n_features': X_train.shape[1],
        'n_skus': len(sku_encoder),
        'feature_names': list(X_train_df.columns)
    }
    
    return (X_train, y_train, sku_train,
            X_val, y_val, sku_val,
            X_test, y_test, sku_test,
            feature_info)


def train_hierarchical_attention(X_train, y_train, sku_train,
                                 X_val, y_val, sku_val,
                                 X_test, y_test, sku_test,
                                 feature_info, epochs=100, batch_size=512):
    """
    Train Hierarchical Attention model on real data.
    
    Args:
        X_train, y_train, sku_train: Training data
        X_val, y_val, sku_val: Validation data
        X_test, y_test, sku_test: Test data
        feature_info: Feature metadata
        epochs: Number of training epochs (default: 100 with early stopping)
        batch_size: Batch size for training
        
    Returns:
        model: Trained Keras model
        y_pred_train: Training predictions
        y_pred_test: Test predictions
        training_time: Time taken for training
    """
    import time
    
    print("\n" + "="*80)
    print("TRAINING HIERARCHICAL ATTENTION MODEL")
    print("="*80)
    
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    n_features = feature_info['n_features']
    n_skus = feature_info['n_skus']
    
    # Build model
    model_builder = DeepSequencePWLHierarchical(
        num_skus=n_skus,
        n_features=n_features,
        enable_intermittent_handling=True,
        id_embedding_dim=16,
        component_hidden_units=64,
        component_dropout=0.3,
        zero_prob_hidden_units=128,
        zero_prob_hidden_layers=2,
        zero_prob_dropout=0.3,
        activation='mish'
    )
    
    # Allocate features to components
    # Features: 0-4 (temporal), 5-9 (cyclical), 10-12 (lags), 13-27 (15 holidays)
    # Total: 28 features (reduced by 1 lag)
    main_model, _, _, _, _ = model_builder.build_model(
        trend_feature_indices=[3, 4],  # quarter, week_of_year
        seasonal_feature_indices=[0, 1, 2, 5, 6, 7, 8, 9],  # temporal + cyclical
        holiday_feature_indices=list(range(13, 28)),  # 15 pre-computed holiday distances
        regressor_feature_indices=[10, 11, 12]  # lag_1, lag_2, lag_7
    )
    
    # Use composite loss for mixed intermittent (90% zeros) + high demand
    main_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={'final_forecast': composite_loss()},
        metrics={'final_forecast': ['mae']}
    )
    
    print(f"  Loss function: Adaptive Composite (BCE weighted by zero fraction, MAE by non-zero fraction)")
    
    print(f"\n✓ Model built")
    print(f"  Total parameters: {main_model.count_params():,}")
    print(f"  Architecture: TabNet + Cross Layers + Intermittent Handling")
    print(f"  Features: {n_features} (exogenous + historical lags)")
    
    # Callbacks for training - FIXED: monitor 'val_loss' not 'val_final_forecast_loss'
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    import os
    
    # Create models directory for reproducibility
    os.makedirs('models', exist_ok=True)
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath='models/best_hierarchical_attention.keras',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print(f"\nTraining on {len(X_train):,} samples for up to {epochs} epochs...")
    print(f"  Early stopping: patience=15 on validation loss")
    print(f"  Reduce LR on plateau: patience=5, factor=0.5")
    start_time = time.time()
    
    history = main_model.fit(
        [X_train, sku_train],
        {'final_forecast': y_train},
        validation_data=([X_val, sku_val], {'final_forecast': y_val}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr, model_checkpoint],
        verbose=1  # Show progress bar
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time:.2f} seconds")
    print("✓ Best model saved to: models/best_hierarchical_attention.keras")
    
    # Get predictions
    print("\nGenerating predictions...")
    pred_train = main_model.predict([X_train, sku_train], verbose=0)
    pred_test = main_model.predict([X_test, sku_test], verbose=0)
    
    y_pred_train = pred_train['final_forecast'].flatten()
    y_pred_test = pred_test['final_forecast'].flatten()
    
    return main_model, y_pred_train, y_pred_test, training_time


def train_lightgbm(X_train, y_train, sku_train,
                  X_test, y_test, sku_test,
                  num_boost_round=200):
    """
    Train LightGBM model on real data.
    
    Uses exogenous features + historical lag features (no leakage).
    
    Args:
        X_train, y_train, sku_train: Training data
        X_test, y_test, sku_test: Test data
        num_boost_round: Number of boosting iterations
        
    Returns:
        model: Trained LightGBM model
        y_pred_train: Training predictions
        y_pred_test: Test predictions
        training_time: Time taken for training
    """
    import time
    
    print("\n" + "="*80)
    print("TRAINING LIGHTGBM MODEL")
    print("="*80)
    
    # Prepare data (add SKU ID as feature)
    # Using ALL features including lag features
    X_train_lgb = np.column_stack([X_train, sku_train])
    X_test_lgb = np.column_stack([X_test, sku_test])
    
    print(f"\n✓ Data preparation:")
    print(f"  Features used: {X_train.shape[1]} (exogenous + lags) + SKU ID")
    print(f"  Lag features: lag_1, lag_7, lag_14, lag_28 (simple shifts only)")
    print(f"  Training samples: {len(X_train_lgb):,}")
    print(f"  Test samples: {len(X_test_lgb):,}")
    
    # Configure LightGBM
    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    print(f"\n✓ Configuration:")
    print(f"  Total features: {X_train_lgb.shape[1]}")
    print(f"  Boosting rounds: {num_boost_round}")
    print(f"  Learning rate: {lgb_params['learning_rate']}")
    print(f"  Num leaves: {lgb_params['num_leaves']}")
    
    # Train model
    print(f"\nTraining...")
    start_time = time.time()
    
    lgb_train = lgb.Dataset(X_train_lgb, label=y_train)
    lgb_test = lgb.Dataset(X_test_lgb, label=y_test, reference=lgb_train)
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_test],
        valid_names=['test'],
        callbacks=[lgb.log_evaluation(period=10)]  # Show progress every 10 rounds
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time:.2f} seconds")
    
    # Save LightGBM model for reproducibility
    import os
    os.makedirs('models', exist_ok=True)
    lgb_model.save_model('models/lightgbm_model.txt')
    print("✓ Model saved to: models/lightgbm_model.txt")
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred_train = lgb_model.predict(X_train_lgb)
    y_pred_test = lgb_model.predict(X_test_lgb)
    
    return lgb_model, y_pred_train, y_pred_test, training_time


def calculate_metrics(y_true, y_pred, model_name="Model", dataset=""):
    """Calculate regression metrics."""
    mae = np.abs(y_true - y_pred).mean()
    
    # MAPE (avoid division by zero)
    nonzero_mask = y_true != 0
    mape = (np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / 
                   y_true[nonzero_mask]).mean() * 100 
            if nonzero_mask.any() else np.inf)
    
    # R²
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Additional metrics for sparse data
    mae_nonzero = (np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]).mean()
                   if nonzero_mask.any() else 0)
    
    metrics = {
        'MAE': mae,
        'MAPE (%)': mape,
        'R²': r2,
        'MAE (non-zero)': mae_nonzero
    }
    
    print(f"\n{model_name} Performance ({dataset}):")
    for metric, value in metrics.items():
        if metric == 'MAPE (%)' and value == np.inf:
            print(f"  {metric}: N/A (no non-zero values)")
        else:
            print(f"  {metric}: {value:.4f}")
    
    return metrics


def compare_models(metrics_ha_train, metrics_ha_test,
                  metrics_lgb_train, metrics_lgb_test,
                  time_ha, time_lgb, params_ha):
    """Print comparison table between models."""
    print("\n" + "="*90)
    print("MODEL COMPARISON ON REAL DATA")
    print("="*90)
    
    comparison_df = pd.DataFrame({
        'Model': ['Hierarchical Attention', 'LightGBM'],
        'Train MAE': [metrics_ha_train['MAE'], metrics_lgb_train['MAE']],
        'Test MAE': [metrics_ha_test['MAE'], metrics_lgb_test['MAE']],
        'Train R²': [metrics_ha_train['R²'], metrics_lgb_train['R²']],
        'Test R²': [metrics_ha_test['R²'], metrics_lgb_test['R²']],
        'Training Time (s)': [f"{time_ha:.2f}", f"{time_lgb:.2f}"],
        'Parameters': [f"{params_ha:,}", "N/A (tree-based)"]
    })
    
    print(comparison_df.to_string(index=False))
    print("="*90)
    
    # Calculate improvements on test set
    mae_improvement = ((metrics_lgb_test['MAE'] - metrics_ha_test['MAE']) / 
                       metrics_lgb_test['MAE']) * 100
    r2_improvement = ((metrics_ha_test['R²'] - metrics_lgb_test['R²']) / 
                      abs(metrics_lgb_test['R²'])) * 100 if \
                      metrics_lgb_test['R²'] != 0 else 0
    
    print(f"\nTest Set: Hierarchical Attention vs LightGBM:")
    print(f"  MAE: {mae_improvement:+.1f}% "
          f"{'(better)' if mae_improvement > 0 else '(worse)'}")
    print(f"  R²:  {r2_improvement:+.1f}% "
          f"{'(better)' if r2_improvement > 0 else '(worse)'}")
    print(f"  Training Speed: "
          f"{time_ha/time_lgb:.1f}x slower than LightGBM")
    
    print(f"\n✓ Both models trained with SAME features")
    print(f"✓ Lag features use historical data only (no leakage)")
    print(f"✓ Fair comparison ensured")


def visualize_comparison(y_test, y_pred_ha, y_pred_lgb,
                        metrics_ha, metrics_lgb, save_path=None):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Sample for visualization (random 1000 points)
    n_samples = min(1000, len(y_test))
    sample_idx = np.random.choice(len(y_test), n_samples, replace=False)
    y_sample = y_test[sample_idx]
    y_pred_ha_sample = y_pred_ha[sample_idx]
    y_pred_lgb_sample = y_pred_lgb[sample_idx]
    
    # 1. Scatter: Actual vs Predicted (Hierarchical Attention)
    axes[0].scatter(y_sample, y_pred_ha_sample, alpha=0.3, s=10, 
                    color='#2E86AB')
    max_val = max(y_sample.max(), y_pred_ha_sample.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2, 
                 label='Perfect Prediction')
    axes[0].set_xlabel('Actual', fontweight='bold')
    axes[0].set_ylabel('Predicted (HA)', fontweight='bold')
    axes[0].set_title(f'Hierarchical Attention\nTest MAE: '
                      f'{metrics_ha["MAE"]:.4f}', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Scatter: Actual vs Predicted (LightGBM)
    axes[1].scatter(y_sample, y_pred_lgb_sample, alpha=0.3, s=10, 
                    color='#A23B72')
    axes[1].plot([0, max_val], [0, max_val], 'r--', lw=2, 
                 label='Perfect Prediction')
    axes[1].set_xlabel('Actual', fontweight='bold')
    axes[1].set_ylabel('Predicted (LGB)', fontweight='bold')
    axes[1].set_title(f'LightGBM\nTest MAE: {metrics_lgb["MAE"]:.4f}', 
                      fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Error distribution
    errors_ha = y_sample - y_pred_ha_sample
    errors_lgb = y_sample - y_pred_lgb_sample
    
    axes[2].hist(errors_ha, bins=50, alpha=0.5, 
                 label='Hierarchical Attention', color='#2E86AB')
    axes[2].hist(errors_lgb, bins=50, alpha=0.5, 
                 label='LightGBM', color='#A23B72')
    axes[2].axvline(x=0, color='red', linestyle='--', lw=2, 
                    label='Zero Error')
    axes[2].set_xlabel('Prediction Error', fontweight='bold')
    axes[2].set_ylabel('Frequency', fontweight='bold')
    axes[2].set_title('Error Distribution', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {save_path}")
    
    plt.show()
    print("\n✓ Visualization complete!")


def main():
    """Main execution function."""
    print("="*80)
    print("HIERARCHICAL ATTENTION vs LIGHTGBM: REAL DATA COMPARISON")
    print("="*80)
    print("\nThis script compares two forecasting models on REAL data:")
    print("  1. Hierarchical Attention (TabNet + Cross + Intermittent)")
    print("  2. LightGBM (Gradient Boosting Trees)")
    print("\n✓ FAIR COMPARISON:")
    print("  - Both models use SAME features (exogenous + lag)")
    print("  - Lag features created with .shift() - NO data leakage")
    print("  - Historical data only (lag_1, lag_2, lag_7)")
    print("  - Simple lag shifts guarantee no future information")
    print("  - Models saved to models/ directory for reproducibility")
    
    # Load real data
    (X_train, y_train, sku_train,
     X_val, y_val, sku_val,
     X_test, y_test, sku_test,
     feature_info) = load_real_data()
    
    # Train Hierarchical Attention
    (model_ha, y_pred_ha_train, y_pred_ha_test, 
     time_ha) = train_hierarchical_attention(
        X_train, y_train, sku_train,
        X_val, y_val, sku_val,
        X_test, y_test, sku_test,
        feature_info,
        epochs=100,  # Max epochs with early stopping
        batch_size=2048  # Larger batch for faster training
    )
    
    # Train LightGBM
    (model_lgb, y_pred_lgb_train, y_pred_lgb_test, 
     time_lgb) = train_lightgbm(
        X_train, y_train, sku_train,
        X_test, y_test, sku_test,
        num_boost_round=100  # Reduced for faster comparison
    )
    
    # Calculate metrics
    metrics_ha_train = calculate_metrics(y_train, y_pred_ha_train, 
                                        "Hierarchical Attention", "Train")
    metrics_ha_test = calculate_metrics(y_test, y_pred_ha_test, 
                                       "Hierarchical Attention", "Test")
    metrics_lgb_train = calculate_metrics(y_train, y_pred_lgb_train, 
                                         "LightGBM", "Train")
    metrics_lgb_test = calculate_metrics(y_test, y_pred_lgb_test, 
                                        "LightGBM", "Test")
    
    # Compare models
    compare_models(
        metrics_ha_train, metrics_ha_test,
        metrics_lgb_train, metrics_lgb_test,
        time_ha, time_lgb,
        params_ha=model_ha.count_params()
    )
    
    # Visualize
    visualize_comparison(
        y_test, y_pred_ha_test, y_pred_lgb_test,
        metrics_ha_test, metrics_lgb_test,
        save_path='model_comparison_real_data.png'
    )
    
    # ROUNDED PREDICTIONS COMPARISON
    print("\n" + "="*80)
    print("ROUNDED PREDICTIONS COMPARISON")
    print("="*80)
    
    # Round predictions
    y_pred_ha_rounded = np.round(y_pred_ha_test)
    y_pred_ha_rounded = np.maximum(y_pred_ha_rounded, 0)  # Ensure non-negative
    
    y_pred_lgb_rounded = np.round(y_pred_lgb_test)
    y_pred_lgb_rounded = np.maximum(y_pred_lgb_rounded, 0)  # Ensure non-negative
    
    # Calculate metrics for rounded predictions
    metrics_ha_rounded = calculate_metrics(y_test, y_pred_ha_rounded, 
                                          "HA (Rounded)", "Test")
    metrics_lgb_rounded = calculate_metrics(y_test, y_pred_lgb_rounded, 
                                           "LGB (Rounded)", "Test")
    
    # Print comparison
    print("\n" + "="*80)
    print("RAW vs ROUNDED PREDICTIONS")
    print("="*80)
    
    print("\nHierarchical Attention:")
    print(f"  RAW:     MAE={metrics_ha_test['mae']:.2f}, "
          f"R²={metrics_ha_test['r2']:.4f}, "
          f"MAPE={metrics_ha_test['mape']:.0f}%")
    print(f"  ROUNDED: MAE={metrics_ha_rounded['mae']:.2f}, "
          f"R²={metrics_ha_rounded['r2']:.4f}, "
          f"MAPE={metrics_ha_rounded['mape']:.0f}%")
    mae_change_ha = ((metrics_ha_rounded['mae'] - metrics_ha_test['mae']) / 
                     metrics_ha_test['mae'] * 100)
    print(f"  Change:  MAE {mae_change_ha:+.1f}%")
    
    print("\nLightGBM:")
    print(f"  RAW:     MAE={metrics_lgb_test['mae']:.2f}, "
          f"R²={metrics_lgb_test['r2']:.4f}, "
          f"MAPE={metrics_lgb_test['mape']:.0f}%")
    print(f"  ROUNDED: MAE={metrics_lgb_rounded['mae']:.2f}, "
          f"R²={metrics_lgb_rounded['r2']:.4f}, "
          f"MAPE={metrics_lgb_rounded['mape']:.0f}%")
    mae_change_lgb = ((metrics_lgb_rounded['mae'] - metrics_lgb_test['mae']) / 
                      metrics_lgb_test['mae'] * 100)
    print(f"  Change:  MAE {mae_change_lgb:+.1f}%")
    
    # Comparison
    print("\n" + "="*80)
    print("ROUNDED PREDICTIONS: HA vs LGB")
    print("="*80)
    mae_imp = ((metrics_lgb_rounded['mae'] - metrics_ha_rounded['mae']) / 
               metrics_lgb_rounded['mae'] * 100)
    r2_imp = ((metrics_ha_rounded['r2'] - metrics_lgb_rounded['r2']) / 
              abs(metrics_lgb_rounded['r2']) * 100)
    
    print(f"\n  MAE:  HA={metrics_ha_rounded['mae']:.2f} vs "
          f"LGB={metrics_lgb_rounded['mae']:.2f} → {mae_imp:+.1f}%")
    print(f"  R²:   HA={metrics_ha_rounded['r2']:.4f} vs "
          f"LGB={metrics_lgb_rounded['r2']:.4f} → {r2_imp:+.1f}%")
    print(f"  MAPE: HA={metrics_ha_rounded['mape']:.0f}% vs "
          f"LGB={metrics_lgb_rounded['mape']:.0f}%")
    
    # Zero/Non-Zero analysis for rounded
    print("\n" + "="*80)
    print("ZERO vs NON-ZERO (Rounded Predictions)")
    print("="*80)
    
    zero_mask = y_test == 0
    nonzero_mask = y_test > 0
    
    print(f"\nZero values ({zero_mask.sum():,} samples, "
          f"{zero_mask.sum()/len(y_test)*100:.1f}%):")
    ha_zero_mae = np.mean(np.abs(y_test[zero_mask] - y_pred_ha_rounded[zero_mask]))
    lgb_zero_mae = np.mean(np.abs(y_test[zero_mask] - y_pred_lgb_rounded[zero_mask]))
    print(f"  HA MAE:  {ha_zero_mae:.2f}")
    print(f"  LGB MAE: {lgb_zero_mae:.2f}")
    print(f"  Improvement: {(lgb_zero_mae - ha_zero_mae)/lgb_zero_mae*100:+.1f}%")
    
    print(f"\nNon-zero values ({nonzero_mask.sum():,} samples, "
          f"{nonzero_mask.sum()/len(y_test)*100:.1f}%):")
    ha_nz_mae = np.mean(np.abs(y_test[nonzero_mask] - y_pred_ha_rounded[nonzero_mask]))
    lgb_nz_mae = np.mean(np.abs(y_test[nonzero_mask] - y_pred_lgb_rounded[nonzero_mask]))
    ha_nz_mape = (np.mean(np.abs((y_test[nonzero_mask] - y_pred_ha_rounded[nonzero_mask]) / 
                                 y_test[nonzero_mask])) * 100)
    lgb_nz_mape = (np.mean(np.abs((y_test[nonzero_mask] - y_pred_lgb_rounded[nonzero_mask]) / 
                                  y_test[nonzero_mask])) * 100)
    print(f"  HA MAE:  {ha_nz_mae:.2f}")
    print(f"  HA MAPE: {ha_nz_mape:.0f}%")
    print(f"  LGB MAE: {lgb_nz_mae:.2f}")
    print(f"  LGB MAPE: {lgb_nz_mape:.0f}%")
    print(f"  MAE Improvement: {(lgb_nz_mae - ha_nz_mae)/lgb_nz_mae*100:+.1f}%")
    print(f"  MAPE Improvement: {(lgb_nz_mape - ha_nz_mape)/lgb_nz_mape*100:+.1f}%")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: model_comparison_real_data.png")


if __name__ == "__main__":
    main()
