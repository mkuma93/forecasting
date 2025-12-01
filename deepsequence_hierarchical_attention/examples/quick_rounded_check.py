"""
Simplified comparison: round predictions from trained models and compare metrics
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

print("="*80)
print("LOADING TRAINED MODELS AND GENERATING ROUNDED PREDICTIONS")
print("="*80)

# Load test data
data_dir = '../../data'
test_df = pd.read_csv(os.path.join(data_dir, 'test_split.csv'))
y_test = test_df['demand'].values

print(f"\nTest dataset:")
print(f"  Samples: {len(y_test):,}")
print(f"  Zero rate: {(y_test == 0).sum() / len(y_test) * 100:.1f}%")
print(f"  Non-zero rate: {(y_test > 0).sum() / len(y_test) * 100:.1f}%")
print(f"  Mean (all): {y_test.mean():.2f}")
print(f"  Mean (non-zero): {y_test[y_test > 0].mean():.2f}")

# Load feature data (will be recreated properly from compare_models_real_data.py logic)
# For now, we'll extract predictions from the training log
print("\nExtracting predictions from training outputs...")

# Check if we have model files
if os.path.exists('best_model.h5') and os.path.exists('lightgbm_model.txt'):
    print("✓ Found trained model files")
    
    # Load the feature-engineered test data
    print("\nRecreating test features from raw data...")
    
    # We'll need to run the actual prediction code from compare_models_real_data.py
    # Let's check if there are saved predictions instead
    
    print("\nSearching for saved prediction files...")
    for f in os.listdir('.'):
        if 'pred' in f.lower() or 'forecast' in f.lower():
            print(f"  Found: {f}")
    
    # For now, let's use the values from training_full.log
    print("\nExtracting metric values from training log...")
    
    with open('training_full.log', 'r') as f:
        log_content = f.read()
    
    # Extract the test performance section
    if "=== Test Performance ===" in log_content:
        test_section = log_content.split("=== Test Performance ===")[1].split("===")[0]
        print("\n" + "="*80)
        print("RAW PREDICTIONS (from training log)")
        print("="*80)
        print(test_section)
    
    # Now let's calculate rounded predictions
    # We need to actually load models and make predictions
    print("\n" + "="*80)
    print("Loading models to generate rounded predictions...")
    print("="*80)
    
    # This requires loading the full data pipeline - let's create a simpler approach
    print("\nFor rounded predictions, we need the actual prediction arrays.")
    print("The training script should save these. Let me check the script...")
    
    # Read the comparison script to see if it saves predictions
    with open('compare_models_real_data.py', 'r') as f:
        script_content = f.read()
    
    if 'np.save' in script_content or 'to_csv' in script_content:
        print("✓ Script saves predictions")
    else:
        print("✗ Script doesn't save predictions - need to modify it")
        print("\nRecommendation: Modify compare_models_real_data.py to save:")
        print("  1. np.save('ha_predictions.npy', ha_pred)")
        print("  2. np.save('lgb_predictions.npy', lgb_pred)")
        print("  3. np.save('y_test.npy', y_test)")
        print("\nThen we can load and round them for comparison.")

else:
    print("✗ Model files not found")
    print("  Expected: best_model.h5, lightgbm_model.txt")
    print("  Please ensure the training completed successfully")

# Manual calculation based on log values
print("\n" + "="*80)
print("MANUAL ROUNDED PREDICTION COMPARISON")
print("="*80)

print("\nBased on the training log, let's estimate rounded prediction performance:")
print("\nRaw Predictions (from log):")
print("  HA:  MAE=1.19, RMSE=4.25, R²=-0.01, MAPE=100%")
print("  LGB: MAE=1.30, RMSE=4.55, R²=-0.12, MAPE=324%")

print("\nRounded Predictions (estimated):")
print("  Rounding typically improves:")
print("    - Zero/non-zero classification (F1-score)")
print("    - Reduces MAE for intermittent demand (fewer fractional predictions)")
print("    - May slightly increase MAE for continuous values")

print("\nFor 90% zero rate data:")
print("  - Rounding helps identify zeros more accurately")
print("  - Improves precision/recall for zero classification")
print("  - HA likely benefits more due to better probability calibration")

print("\nExpected impact:")
print("  HA rounded:  MAE ~1.15-1.20 (slight improvement)")
print("  LGB rounded: MAE ~1.28-1.33 (slight improvement)")
print("  Both will have better zero classification metrics")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("To get exact rounded prediction metrics:")
print("1. Modify compare_models_real_data.py to save predictions:")
print("   - Add at end of script before plotting:")
print("     np.save('ha_test_pred.npy', ha_test_pred)")
print("     np.save('lgb_test_pred.npy', lgb_pred)")
print("     np.save('y_test_values.npy', y_test)")
print("")
print("2. Re-run training or just the prediction part")
print("")
print("3. Then run this script to compare rounded vs raw predictions")
print("="*80)
