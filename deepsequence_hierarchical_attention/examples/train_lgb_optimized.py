"""
Train LightGBM with hyperparameter tuning using Optuna.
Compare optimized LightGBM vs baseline LightGBM vs Lightweight Hierarchical Attention.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def create_features(df, holiday_features_df):
    """Create features matching lightweight training script."""
    # Parse dates first
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(['id_var', 'ds']).reset_index(drop=True)
    
    features = {}
    
    # Holiday distance (already in data)
    features['holiday_distance'] = df['cumdist'].values
    
    # Temporal features
    features['day_of_week'] = df['ds'].dt.dayofweek.values
    features['day_of_month'] = df['ds'].dt.day.values
    features['month'] = df['ds'].dt.month.values
    features['quarter'] = df['ds'].dt.quarter.values
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
    features['lag_1'] = df.groupby('id_var')['Quantity'].shift(1).fillna(0).values
    features['lag_2'] = df.groupby('id_var')['Quantity'].shift(2).fillna(0).values
    features['lag_7'] = df.groupby('id_var')['Quantity'].shift(7).fillna(0).values
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features)
    
    # Add pre-computed holiday features (15 holidays)
    features_df = pd.concat([features_df, holiday_features_df.reset_index(drop=True)], axis=1)
    
    # SKU ID as categorical
    features_df['sku_id'] = df['id_var'].astype('category').cat.codes
    
    return features_df


def objective(trial, X_train, y_train, X_val, y_val, cat_idx):
    """Optuna objective function for LightGBM hyperparameter tuning."""
    
    # Suggest hyperparameters
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'cat_smooth': trial.suggest_int('cat_smooth', 10, 100),
    }
    
    # Train model
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=[cat_idx])
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=[cat_idx])
    
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    # Predict and evaluate
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_val, y_pred)
    
    return mae


def calculate_metrics(y_true, y_pred, model_name):
    """Calculate comprehensive metrics."""
    # Overall metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE
    nonzero_mask = y_true > 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / 
                              y_true[nonzero_mask])) * 100
    else:
        mape = 0.0
    
    # Zero prediction accuracy
    zero_mask = y_true == 0
    zero_pred_mask = y_pred < 0.5
    zero_accuracy = np.mean(zero_mask == zero_pred_mask) * 100
    
    # Non-zero metrics
    if nonzero_mask.sum() > 0:
        mae_nonzero = np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]))
        rmse_nonzero = np.sqrt(np.mean((y_true[nonzero_mask] - y_pred[nonzero_mask]) ** 2))
    else:
        mae_nonzero = 0.0
        rmse_nonzero = 0.0
    
    print(f"\n{model_name} Metrics:")
    print(f"  Overall:")
    print(f"    MAE:  {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAPE: {mape:.2f}%")
    print(f"  Zero Prediction Accuracy: {zero_accuracy:.2f}%")
    print(f"  Non-zero samples:")
    print(f"    MAE:  {mae_nonzero:.4f}")
    print(f"    RMSE: {rmse_nonzero:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'zero_accuracy': zero_accuracy,
        'mae_nonzero': mae_nonzero,
        'rmse_nonzero': rmse_nonzero
    }


print("=" * 70)
print("LightGBM Hyperparameter Optimization with Optuna")
print("=" * 70)

# 1. Load data
print("\n1. Loading data...")
train_df = pd.read_csv('../../data/train_split.csv')
val_df = pd.read_csv('../../data/val_split.csv')
test_df = pd.read_csv('../../data/test_split.csv')

holiday_train = pd.read_csv('../../data/holiday_features_train.csv')
holiday_val = pd.read_csv('../../data/holiday_features_val.csv')
holiday_test = pd.read_csv('../../data/holiday_features_test.csv')

print(f"   ✓ Train: {len(train_df):,} samples")
print(f"   ✓ Val: {len(val_df):,} samples")
print(f"   ✓ Test: {len(test_df):,} samples")

# 2. Create features
print("\n2. Creating features...")
X_train_df = create_features(train_df, holiday_train)
X_val_df = create_features(val_df, holiday_val)
X_test_df = create_features(test_df, holiday_test)

y_train = train_df['Quantity'].values
y_val = val_df['Quantity'].values
y_test = test_df['Quantity'].values

print(f"   ✓ Features: {X_train_df.shape[1]} (30 total including SKU)")
print(f"   ✓ Zero rate: {(y_train == 0).mean() * 100:.1f}%")

# Get categorical feature index
cat_feature_idx = list(X_train_df.columns).index('sku_id')
print(f"   ✓ SKU categorical index: {cat_feature_idx}")

# 3. Hyperparameter optimization
print("\n3. Running hyperparameter optimization...")
print(f"   Running Optuna with 50 trials...")

study = optuna.create_study(direction='minimize', study_name='lgb_optimization')
study.optimize(
    lambda trial: objective(trial, X_train_df.values, y_train, X_val_df.values, y_val, cat_feature_idx),
    n_trials=50,
    show_progress_bar=True
)

print(f"\n   ✓ Best MAE: {study.best_value:.4f}")
print(f"   ✓ Best parameters:")
for key, value in study.best_params.items():
    print(f"      {key}: {value}")

# 4. Train final optimized model
print("\n4. Training final optimized model...")
best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
})

lgb_train = lgb.Dataset(X_train_df.values, y_train, categorical_feature=[cat_feature_idx])
lgb_val = lgb.Dataset(X_val_df.values, y_val, reference=lgb_train, categorical_feature=[cat_feature_idx])

optimized_model = lgb.train(
    best_params,
    lgb_train,
    valid_sets=[lgb_val],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=10)
    ]
)

print(f"   ✓ Optimized model trained with {optimized_model.best_iteration} trees")
optimized_model.save_model('models/lightgbm_optimized.txt')
print(f"   ✓ Model saved to: models/lightgbm_optimized.txt")

# 5. Load baseline model
print("\n5. Loading baseline LightGBM model...")
baseline_model = lgb.Booster(model_file='models/lightgbm_model.txt')
print(f"   ✓ Baseline model loaded")

# 6. Generate predictions
print("\n6. Generating predictions on test set...")
y_pred_optimized = optimized_model.predict(X_test_df.values, num_iteration=optimized_model.best_iteration)
y_pred_baseline = baseline_model.predict(X_test_df.values)
print(f"   ✓ Predictions generated")

# 7. Calculate metrics
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

metrics_optimized = calculate_metrics(y_test, y_pred_optimized, "Optimized LightGBM")
metrics_baseline = calculate_metrics(y_test, y_pred_baseline, "Baseline LightGBM")

# 8. Load lightweight model predictions (from previous comparison)
print("\n8. Loading Lightweight model results...")
try:
    comparison_df = pd.read_csv('lightweight_vs_lgb_comparison.csv')
    lightweight_results = comparison_df[comparison_df['Model'] == 'Lightweight Hierarchical Attention'].iloc[0]
    
    print(f"\nLightweight Hierarchical Attention Metrics:")
    print(f"  Overall:")
    print(f"    MAE:  {lightweight_results['MAE']:.4f}")
    print(f"    RMSE: {lightweight_results['RMSE']:.4f}")
    print(f"    MAPE: {lightweight_results['MAPE']:.2f}%")
    print(f"  Zero Prediction Accuracy: {lightweight_results['Zero_Accuracy']:.2f}%")
    
    metrics_lightweight = {
        'mae': lightweight_results['MAE'],
        'rmse': lightweight_results['RMSE'],
        'mape': lightweight_results['MAPE'],
        'zero_accuracy': lightweight_results['Zero_Accuracy']
    }
except:
    print("   ⚠ Could not load lightweight model results")
    metrics_lightweight = None

# 9. Comparison
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

print(f"\nOptimized LightGBM vs Baseline LightGBM:")
mae_improvement = ((metrics_baseline['mae'] - metrics_optimized['mae']) / metrics_baseline['mae']) * 100
rmse_improvement = ((metrics_baseline['rmse'] - metrics_optimized['rmse']) / metrics_baseline['rmse']) * 100
mape_improvement = ((metrics_baseline['mape'] - metrics_optimized['mape']) / metrics_baseline['mape']) * 100

print(f"  MAE:  {mae_improvement:+.2f}% {'✓ Better' if mae_improvement > 0 else '✗ Worse'}")
print(f"  RMSE: {rmse_improvement:+.2f}% {'✓ Better' if rmse_improvement > 0 else '✗ Worse'}")
print(f"  MAPE: {mape_improvement:+.2f}% {'✓ Better' if mape_improvement > 0 else '✗ Worse'}")

if metrics_lightweight:
    print(f"\nOptimized LightGBM vs Lightweight Hierarchical Attention:")
    mae_vs_nn = ((metrics_lightweight['mae'] - metrics_optimized['mae']) / metrics_lightweight['mae']) * 100
    rmse_vs_nn = ((metrics_lightweight['rmse'] - metrics_optimized['rmse']) / metrics_lightweight['rmse']) * 100
    mape_vs_nn = ((metrics_lightweight['mape'] - metrics_optimized['mape']) / metrics_lightweight['mape']) * 100
    
    print(f"  MAE:  {mae_vs_nn:+.2f}% {'✓ LGB Better' if mae_vs_nn < 0 else '✗ NN Better'}")
    print(f"  RMSE: {rmse_vs_nn:+.2f}% {'✓ LGB Better' if rmse_vs_nn < 0 else '✗ NN Better'}")
    print(f"  MAPE: {mape_vs_nn:+.2f}% {'✓ LGB Better' if mape_vs_nn < 0 else '✗ NN Better'}")

# 10. Save results
print("\n10. Saving results...")
results_df = pd.DataFrame([
    {
        'Model': 'Optimized LightGBM',
        'MAE': metrics_optimized['mae'],
        'RMSE': metrics_optimized['rmse'],
        'MAPE': metrics_optimized['mape'],
        'Zero_Accuracy': metrics_optimized['zero_accuracy'],
        'MAE_NonZero': metrics_optimized['mae_nonzero'],
        'RMSE_NonZero': metrics_optimized['rmse_nonzero'],
    },
    {
        'Model': 'Baseline LightGBM',
        'MAE': metrics_baseline['mae'],
        'RMSE': metrics_baseline['rmse'],
        'MAPE': metrics_baseline['mape'],
        'Zero_Accuracy': metrics_baseline['zero_accuracy'],
        'MAE_NonZero': metrics_baseline['mae_nonzero'],
        'RMSE_NonZero': metrics_baseline['rmse_nonzero'],
    }
])

results_df.to_csv('lgb_optimization_comparison.csv', index=False)
print(f"   ✓ Results saved to: lgb_optimization_comparison.csv")

print("\n" + "=" * 70)
print("✅ Optimization Complete!")
print("=" * 70)
