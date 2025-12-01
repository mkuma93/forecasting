"""
Load trained models and compare rounded vs raw predictions
Uses saved models from compare_models_real_data.py
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deepsequence_hierarchical_attention import DeepSequencePWLHierarchical, composite_loss

print("="*80)
print("ROUNDED PREDICTIONS COMPARISON (Using Saved Models)")
print("="*80)

# Import the data loading function from the main script
from compare_models_real_data import load_real_data

# Load data
print("\nLoading data...")
(X_train, y_train, sku_train,
 X_val, y_val, sku_val,
 X_test, y_test, sku_test,
 feature_info) = load_real_data()

print(f"✓ Test samples: {len(y_test):,}")
print(f"  Zero rate: {(y_test == 0).sum() / len(y_test) * 100:.1f}%")

# Load Hierarchical Attention model
print("\nLoading Hierarchical Attention model...")
n_features = X_test.shape[1]
n_skus = len(np.unique(sku_test))

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

# Build model with feature allocation
model_ha, _, _, _, _ = model_builder.build_model(
    trend_feature_indices=[3, 4],
    seasonal_feature_indices=[0, 1, 2, 5, 6, 7, 8, 9],
    holiday_feature_indices=list(range(14, 29)),
    regressor_feature_indices=[10, 11, 12, 13]
)

model_ha.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss={'final_forecast': composite_loss()},
    metrics={'final_forecast': ['mae']}
)

model_ha.load_weights('best_model.h5')
print("✓ Model loaded")

# Load LightGBM model
print("\nLoading LightGBM model...")
model_lgb = lgb.Booster(model_file='lightgbm_model.txt')
print("✓ Model loaded")

# Make predictions
print("\nGenerating predictions...")
y_pred_ha = model_ha.predict(X_test, batch_size=2048, verbose=0)
y_pred_ha = y_pred_ha.flatten()

y_pred_lgb = model_lgb.predict(X_test)

print("✓ Predictions generated")

# Round predictions
y_pred_ha_rounded = np.round(y_pred_ha)
y_pred_ha_rounded = np.maximum(y_pred_ha_rounded, 0)

y_pred_lgb_rounded = np.round(y_pred_lgb)
y_pred_lgb_rounded = np.maximum(y_pred_lgb_rounded, 0)

# Calculate metrics
def calc_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

print("\n" + "="*80)
print("OVERALL PERFORMANCE")
print("="*80)

# Raw predictions
ha_raw = calc_metrics(y_test, y_pred_ha)
lgb_raw = calc_metrics(y_test, y_pred_lgb)

print("\nRAW Predictions:")
print(f"  HA:  MAE={ha_raw['mae']:.2f}, R²={ha_raw['r2']:.4f}, MAPE={ha_raw['mape']:.0f}%")
print(f"  LGB: MAE={lgb_raw['mae']:.2f}, R²={lgb_raw['r2']:.4f}, MAPE={lgb_raw['mape']:.0f}%")
print(f"  Improvement: MAE {(lgb_raw['mae']-ha_raw['mae'])/lgb_raw['mae']*100:+.1f}%")

# Rounded predictions
ha_rnd = calc_metrics(y_test, y_pred_ha_rounded)
lgb_rnd = calc_metrics(y_test, y_pred_lgb_rounded)

print("\nROUNDED Predictions:")
print(f"  HA:  MAE={ha_rnd['mae']:.2f}, R²={ha_rnd['r2']:.4f}, MAPE={ha_rnd['mape']:.0f}%")
print(f"  LGB: MAE={lgb_rnd['mae']:.2f}, R²={lgb_rnd['r2']:.4f}, MAPE={lgb_rnd['mape']:.0f}%")
print(f"  Improvement: MAE {(lgb_rnd['mae']-ha_rnd['mae'])/lgb_rnd['mae']*100:+.1f}%")

print("\nImpact of Rounding:")
print(f"  HA:  MAE {ha_raw['mae']:.2f} → {ha_rnd['mae']:.2f} ({(ha_rnd['mae']-ha_raw['mae'])/ha_raw['mae']*100:+.1f}%)")
print(f"  LGB: MAE {lgb_raw['mae']:.2f} → {lgb_rnd['mae']:.2f} ({(lgb_rnd['mae']-lgb_raw['mae'])/lgb_raw['mae']*100:+.1f}%)")

# Zero/Non-zero analysis
print("\n" + "="*80)
print("ZERO vs NON-ZERO PERFORMANCE")
print("="*80)

zero_mask = y_test == 0
nonzero_mask = y_test > 0

print(f"\nData distribution:")
print(f"  Zero: {zero_mask.sum():,} ({zero_mask.sum()/len(y_test)*100:.1f}%)")
print(f"  Non-zero: {nonzero_mask.sum():,} ({nonzero_mask.sum()/len(y_test)*100:.1f}%)")

# Zero values
print("\n--- ZERO VALUES ---")
print("\nRAW predictions:")
ha_zero_raw = np.mean(np.abs(y_test[zero_mask] - y_pred_ha[zero_mask]))
lgb_zero_raw = np.mean(np.abs(y_test[zero_mask] - y_pred_lgb[zero_mask]))
print(f"  HA:  MAE={ha_zero_raw:.2f}")
print(f"  LGB: MAE={lgb_zero_raw:.2f}")
print(f"  Improvement: {(lgb_zero_raw-ha_zero_raw)/lgb_zero_raw*100:+.1f}%")

print("\nROUNDED predictions:")
ha_zero_rnd = np.mean(np.abs(y_test[zero_mask] - y_pred_ha_rounded[zero_mask]))
lgb_zero_rnd = np.mean(np.abs(y_test[zero_mask] - y_pred_lgb_rounded[zero_mask]))
print(f"  HA:  MAE={ha_zero_rnd:.2f}")
print(f"  LGB: MAE={lgb_zero_rnd:.2f}")
print(f"  Improvement: {(lgb_zero_rnd-ha_zero_rnd)/lgb_zero_rnd*100:+.1f}%")

# Non-zero values
print("\n--- NON-ZERO VALUES ---")
print("\nRAW predictions:")
ha_nz_raw_mae = np.mean(np.abs(y_test[nonzero_mask] - y_pred_ha[nonzero_mask]))
lgb_nz_raw_mae = np.mean(np.abs(y_test[nonzero_mask] - y_pred_lgb[nonzero_mask]))
ha_nz_raw_mape = np.mean(np.abs((y_test[nonzero_mask] - y_pred_ha[nonzero_mask]) / y_test[nonzero_mask])) * 100
lgb_nz_raw_mape = np.mean(np.abs((y_test[nonzero_mask] - y_pred_lgb[nonzero_mask]) / y_test[nonzero_mask])) * 100
print(f"  HA:  MAE={ha_nz_raw_mae:.2f}, MAPE={ha_nz_raw_mape:.0f}%")
print(f"  LGB: MAE={lgb_nz_raw_mae:.2f}, MAPE={lgb_nz_raw_mape:.0f}%")
print(f"  Improvement: MAE {(lgb_nz_raw_mae-ha_nz_raw_mae)/lgb_nz_raw_mae*100:+.1f}%, MAPE {(lgb_nz_raw_mape-ha_nz_raw_mape)/lgb_nz_raw_mape*100:+.1f}%")

print("\nROUNDED predictions:")
ha_nz_rnd_mae = np.mean(np.abs(y_test[nonzero_mask] - y_pred_ha_rounded[nonzero_mask]))
lgb_nz_rnd_mae = np.mean(np.abs(y_test[nonzero_mask] - y_pred_lgb_rounded[nonzero_mask]))
ha_nz_rnd_mape = np.mean(np.abs((y_test[nonzero_mask] - y_pred_ha_rounded[nonzero_mask]) / y_test[nonzero_mask])) * 100
lgb_nz_rnd_mape = np.mean(np.abs((y_test[nonzero_mask] - y_pred_lgb_rounded[nonzero_mask]) / y_test[nonzero_mask])) * 100
print(f"  HA:  MAE={ha_nz_rnd_mae:.2f}, MAPE={ha_nz_rnd_mape:.0f}%")
print(f"  LGB: MAE={lgb_nz_rnd_mae:.2f}, MAPE={lgb_nz_rnd_mape:.0f}%")
print(f"  Improvement: MAE {(lgb_nz_rnd_mae-ha_nz_rnd_mae)/lgb_nz_rnd_mae*100:+.1f}%, MAPE {(lgb_nz_rnd_mape-ha_nz_rnd_mape)/lgb_nz_rnd_mape*100:+.1f}%")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nRounded predictions show:")
print(f"  • Overall MAE: HA {ha_rnd['mae']:.2f} vs LGB {lgb_rnd['mae']:.2f}")
print(f"  • Zero MAE: HA {ha_zero_rnd:.2f} vs LGB {lgb_zero_rnd:.2f}")
print(f"  • Non-zero MAE: HA {ha_nz_rnd_mae:.2f} vs LGB {lgb_nz_rnd_mae:.2f}")
print(f"  • Non-zero MAPE: HA {ha_nz_rnd_mape:.0f}% vs LGB {lgb_nz_rnd_mape:.0f}%")
print("\nRounding impact:")
print(f"  • HA: MAE {ha_raw['mae']:.2f} → {ha_rnd['mae']:.2f}")
print(f"  • LGB: MAE {lgb_raw['mae']:.2f} → {lgb_rnd['mae']:.2f}")
print("="*80)
