"""
Compare models with rounded predictions and analyze by zero/non-zero categories
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deepsequence_hierarchical_attention import (
    DeepSequencePWLHierarchical,
    composite_loss
)

def calculate_metrics(y_true, y_pred, label=""):
    """Calculate comprehensive metrics"""
    # Overall metrics
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf
    
    # Zero vs Non-Zero analysis
    zero_mask = y_true == 0
    nonzero_mask = y_true > 0
    
    zero_count = zero_mask.sum()
    nonzero_count = nonzero_mask.sum()
    
    # Metrics for zero values
    zero_mae = np.mean(np.abs(y_true[zero_mask] - y_pred[zero_mask])) if zero_count > 0 else 0
    zero_rmse = np.sqrt(np.mean((y_true[zero_mask] - y_pred[zero_mask]) ** 2)) if zero_count > 0 else 0
    
    # Metrics for non-zero values
    nonzero_mae = np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask])) if nonzero_count > 0 else 0
    nonzero_rmse = np.sqrt(np.mean((y_true[nonzero_mask] - y_pred[nonzero_mask]) ** 2)) if nonzero_count > 0 else 0
    nonzero_mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100 if nonzero_count > 0 else np.inf
    
    # Confusion matrix for zero/non-zero classification
    pred_zero = y_pred == 0
    pred_nonzero = y_pred > 0
    
    true_negative = (zero_mask & pred_zero).sum()  # Correctly predicted zero
    false_positive = (zero_mask & pred_nonzero).sum()  # Predicted non-zero but was zero
    false_negative = (nonzero_mask & pred_zero).sum()  # Predicted zero but was non-zero
    true_positive = (nonzero_mask & pred_nonzero).sum()  # Correctly predicted non-zero
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    zero_precision = true_negative / (true_negative + false_negative) if (true_negative + false_negative) > 0 else 0
    zero_recall = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
    
    results = {
        'label': label,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'zero_count': zero_count,
        'nonzero_count': nonzero_count,
        'zero_pct': zero_count / len(y_true) * 100,
        'nonzero_pct': nonzero_count / len(y_true) * 100,
        'zero_mae': zero_mae,
        'zero_rmse': zero_rmse,
        'nonzero_mae': nonzero_mae,
        'nonzero_rmse': nonzero_rmse,
        'nonzero_mape': nonzero_mape,
        'true_negative': true_negative,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'true_positive': true_positive,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'zero_precision': zero_precision,
        'zero_recall': zero_recall,
    }
    
    return results

def print_metrics(metrics, title):
    """Print metrics in a formatted way"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    print(f"\nOverall Performance:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    print(f"\nDistribution:")
    print(f"  Zero values:     {metrics['zero_count']:,} ({metrics['zero_pct']:.1f}%)")
    print(f"  Non-zero values: {metrics['nonzero_count']:,} ({metrics['nonzero_pct']:.1f}%)")
    
    print(f"\nZero Values Performance:")
    print(f"  MAE:  {metrics['zero_mae']:.4f}")
    print(f"  RMSE: {metrics['zero_rmse']:.4f}")
    
    print(f"\nNon-Zero Values Performance:")
    print(f"  MAE:  {metrics['nonzero_mae']:.4f}")
    print(f"  RMSE: {metrics['nonzero_rmse']:.4f}")
    print(f"  MAPE: {metrics['nonzero_mape']:.2f}%")
    
    print(f"\nZero/Non-Zero Classification:")
    print(f"  True Negatives (correct zeros):  {metrics['true_negative']:,}")
    print(f"  False Positives (wrong non-zero): {metrics['false_positive']:,}")
    print(f"  False Negatives (wrong zero):     {metrics['false_negative']:,}")
    print(f"  True Positives (correct non-zero): {metrics['true_positive']:,}")
    print(f"  Precision (non-zero): {metrics['precision']:.4f}")
    print(f"  Recall (non-zero):    {metrics['recall']:.4f}")
    print(f"  F1-Score (non-zero):  {metrics['f1']:.4f}")
    print(f"  Precision (zero):     {metrics['zero_precision']:.4f}")
    print(f"  Recall (zero):        {metrics['zero_recall']:.4f}")

def compare_metrics(ha_metrics, lgb_metrics):
    """Compare and print improvement percentages"""
    print(f"\n{'='*80}")
    print("COMPARISON: Hierarchical Attention vs LightGBM (Rounded Predictions)")
    print(f"{'='*80}")
    
    print(f"\nOverall Performance:")
    mae_imp = (lgb_metrics['mae'] - ha_metrics['mae']) / lgb_metrics['mae'] * 100
    rmse_imp = (lgb_metrics['rmse'] - ha_metrics['rmse']) / lgb_metrics['rmse'] * 100
    r2_imp = (ha_metrics['r2'] - lgb_metrics['r2']) / abs(lgb_metrics['r2']) * 100 if lgb_metrics['r2'] != 0 else 0
    
    print(f"  MAE:  HA={ha_metrics['mae']:.2f} vs LGB={lgb_metrics['mae']:.2f} → {mae_imp:+.1f}%")
    print(f"  RMSE: HA={ha_metrics['rmse']:.2f} vs LGB={lgb_metrics['rmse']:.2f} → {rmse_imp:+.1f}%")
    print(f"  R²:   HA={ha_metrics['r2']:.4f} vs LGB={lgb_metrics['r2']:.4f} → {r2_imp:+.1f}%")
    print(f"  MAPE: HA={ha_metrics['mape']:.0f}% vs LGB={lgb_metrics['mape']:.0f}%")
    
    print(f"\nZero Values Performance:")
    zero_mae_imp = (lgb_metrics['zero_mae'] - ha_metrics['zero_mae']) / lgb_metrics['zero_mae'] * 100 if lgb_metrics['zero_mae'] > 0 else 0
    print(f"  MAE:  HA={ha_metrics['zero_mae']:.2f} vs LGB={lgb_metrics['zero_mae']:.2f} → {zero_mae_imp:+.1f}%")
    
    print(f"\nNon-Zero Values Performance:")
    nonzero_mae_imp = (lgb_metrics['nonzero_mae'] - ha_metrics['nonzero_mae']) / lgb_metrics['nonzero_mae'] * 100 if lgb_metrics['nonzero_mae'] > 0 else 0
    nonzero_mape_imp = (lgb_metrics['nonzero_mape'] - ha_metrics['nonzero_mape']) / lgb_metrics['nonzero_mape'] * 100 if lgb_metrics['nonzero_mape'] > 0 else 0
    print(f"  MAE:  HA={ha_metrics['nonzero_mae']:.2f} vs LGB={lgb_metrics['nonzero_mae']:.2f} → {nonzero_mae_imp:+.1f}%")
    print(f"  MAPE: HA={ha_metrics['nonzero_mape']:.0f}% vs LGB={lgb_metrics['nonzero_mape']:.0f}% → {nonzero_mape_imp:+.1f}%")
    
    print(f"\nZero/Non-Zero Classification:")
    f1_imp = (ha_metrics['f1'] - lgb_metrics['f1']) / lgb_metrics['f1'] * 100 if lgb_metrics['f1'] > 0 else 0
    prec_imp = (ha_metrics['precision'] - lgb_metrics['precision']) / lgb_metrics['precision'] * 100 if lgb_metrics['precision'] > 0 else 0
    rec_imp = (ha_metrics['recall'] - lgb_metrics['recall']) / lgb_metrics['recall'] * 100 if lgb_metrics['recall'] > 0 else 0
    
    print(f"  F1-Score:  HA={ha_metrics['f1']:.4f} vs LGB={lgb_metrics['f1']:.4f} → {f1_imp:+.1f}%")
    print(f"  Precision: HA={ha_metrics['precision']:.4f} vs LGB={lgb_metrics['precision']:.4f} → {prec_imp:+.1f}%")
    print(f"  Recall:    HA={ha_metrics['recall']:.4f} vs LGB={lgb_metrics['recall']:.4f} → {rec_imp:+.1f}%")

def main():
    print("Loading data...")
    
    # Load data
    data_dir = '../../data'
    
    train_df = pd.read_csv(os.path.join(data_dir, 'train_split.csv'), 
                           parse_dates=['ds'])
    val_df = pd.read_csv(os.path.join(data_dir, 'val_split.csv'),
                         parse_dates=['ds'])
    test_df = pd.read_csv(os.path.join(data_dir, 'test_split.csv'),
                          parse_dates=['ds'])
    
    # Load and merge holiday features
    holiday_test = pd.read_csv(os.path.join(data_dir, 'holiday_features_test.csv'))
    test_df = pd.concat([test_df.reset_index(drop=True), 
                         holiday_test.reset_index(drop=True)], axis=1)
    
    # Feature columns
    feature_cols = [
        'day_of_week', 'day_of_month', 'month', 'quarter',
        'week_of_year', 'day_of_year', 'is_weekend',
        'sin_day_of_week', 'cos_day_of_week',
        'sin_day_of_month', 'cos_day_of_month',
        'sin_month', 'cos_month',
        'sin_day_of_year', 'cos_day_of_year',
        'days_to_holi_1', 'days_to_holi_2', 'days_to_holi_3',
        'days_to_holi_4', 'days_to_holi_5', 'days_to_holi_6',
        'days_to_holi_7', 'days_to_holi_8', 'days_to_holi_9',
        'days_to_holi_10', 'days_to_holi_11', 'days_to_holi_12',
        'days_to_holi_13', 'days_to_holi_14', 'days_to_holi_15',
        'lag_1', 'lag_7', 'lag_14', 'lag_30'
    ]
    
    X_test = test_df[feature_cols].values
    y_test = test_df['demand'].values
    
    print(f"Test samples: {len(y_test):,}")
    print(f"Test zero rate: {(y_test == 0).sum() / len(y_test) * 100:.1f}%")
    
    # Load Hierarchical Attention model
    print("\nLoading Hierarchical Attention model...")
    n_features = len(feature_cols)
    ha_model = DeepSequencePWLHierarchical(
        input_dim=n_features,
        output_dim=1,
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        virtual_batch_size=256,
        momentum=0.98
    )
    
    ha_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={'final_forecast': composite_loss()},
        metrics={'final_forecast': ['mae']}
    )
    
    ha_model.load_weights('best_model.h5')
    
    # Make predictions
    print("\nMaking predictions...")
    ha_pred_raw = ha_model.predict(X_test, batch_size=2048, verbose=0)
    ha_pred_raw = ha_pred_raw.flatten()
    ha_pred_rounded = np.round(ha_pred_raw)
    ha_pred_rounded = np.maximum(ha_pred_rounded, 0)  # Ensure non-negative
    
    # Load LightGBM model
    print("Loading LightGBM model...")
    lgb_model = lgb.Booster(model_file='lightgbm_model.txt')
    lgb_pred_raw = lgb_model.predict(X_test)
    lgb_pred_rounded = np.round(lgb_pred_raw)
    lgb_pred_rounded = np.maximum(lgb_pred_rounded, 0)  # Ensure non-negative
    
    # Calculate metrics for raw predictions
    print("\n" + "="*80)
    print("RAW PREDICTIONS (Without Rounding)")
    print("="*80)
    
    ha_raw_metrics = calculate_metrics(y_test, ha_pred_raw, "Hierarchical Attention (Raw)")
    print_metrics(ha_raw_metrics, "Hierarchical Attention - Raw Predictions")
    
    lgb_raw_metrics = calculate_metrics(y_test, lgb_pred_raw, "LightGBM (Raw)")
    print_metrics(lgb_raw_metrics, "LightGBM - Raw Predictions")
    
    # Calculate metrics for rounded predictions
    print("\n" + "="*80)
    print("ROUNDED PREDICTIONS")
    print("="*80)
    
    ha_rounded_metrics = calculate_metrics(y_test, ha_pred_rounded, "Hierarchical Attention (Rounded)")
    print_metrics(ha_rounded_metrics, "Hierarchical Attention - Rounded Predictions")
    
    lgb_rounded_metrics = calculate_metrics(y_test, lgb_pred_rounded, "LightGBM (Rounded)")
    print_metrics(lgb_rounded_metrics, "LightGBM - Rounded Predictions")
    
    # Compare raw predictions
    print("\n" + "="*80)
    print("RAW PREDICTIONS COMPARISON")
    print("="*80)
    compare_metrics(ha_raw_metrics, lgb_raw_metrics)
    
    # Compare rounded predictions
    compare_metrics(ha_rounded_metrics, lgb_rounded_metrics)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nImpact of Rounding:")
    print(f"  Hierarchical Attention:")
    print(f"    MAE: {ha_raw_metrics['mae']:.4f} → {ha_rounded_metrics['mae']:.4f} (Δ={ha_rounded_metrics['mae']-ha_raw_metrics['mae']:.4f})")
    print(f"    Non-zero MAE: {ha_raw_metrics['nonzero_mae']:.2f} → {ha_rounded_metrics['nonzero_mae']:.2f} (Δ={ha_rounded_metrics['nonzero_mae']-ha_raw_metrics['nonzero_mae']:.2f})")
    print(f"    F1-Score: {ha_raw_metrics['f1']:.4f} → {ha_rounded_metrics['f1']:.4f} (Δ={ha_rounded_metrics['f1']-ha_raw_metrics['f1']:.4f})")
    
    print(f"\n  LightGBM:")
    print(f"    MAE: {lgb_raw_metrics['mae']:.4f} → {lgb_rounded_metrics['mae']:.4f} (Δ={lgb_rounded_metrics['mae']-lgb_raw_metrics['mae']:.4f})")
    print(f"    Non-zero MAE: {lgb_raw_metrics['nonzero_mae']:.2f} → {lgb_rounded_metrics['nonzero_mae']:.2f} (Δ={lgb_rounded_metrics['nonzero_mae']-lgb_raw_metrics['nonzero_mae']:.2f})")
    print(f"    F1-Score: {lgb_raw_metrics['f1']:.4f} → {lgb_rounded_metrics['f1']:.4f} (Δ={lgb_rounded_metrics['f1']-lgb_raw_metrics['f1']:.4f})")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
