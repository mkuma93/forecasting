"""
Compare Lightweight Hierarchical Attention vs LightGBM Performance
"""
import sys
sys.path.insert(0, '../')

import numpy as np
import pandas as pd
import keras
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# Import custom layers to register them for model loading
from deepsequence_hierarchical_attention.components_lightweight import (
    GatherLayer, SqueezeLayer, ReduceSumLayer, OneMinusLayer,
    ChangepointReLU, TrendComponentLightweight, HolidayComponentLightweight,
    SeasonalComponentLightweight, RegressorComponentLightweight,
    CrossLayerLightweight, IntermittentHandlerLightweight
)

print("=" * 70)
print("Lightweight Hierarchical Attention vs LightGBM Comparison")
print("=" * 70)

# Load test data
print("\n1. Loading test data...")
test_df = pd.read_csv('../../data/test_split.csv')
holiday_test = pd.read_csv('../../data/holiday_features_test.csv')

def create_features(df, holiday_df):
    """Create 27 features matching training."""
    features_df = pd.DataFrame()
    
    # Parse dates
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Temporal features (5)
    features_df['day_of_week'] = df['ds'].dt.dayofweek
    features_df['day_of_month'] = df['ds'].dt.day
    features_df['month'] = df['ds'].dt.month
    features_df['quarter'] = df['ds'].dt.quarter
    features_df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
    
    # Cyclical encodings (4)
    features_df['dow_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
    features_df['dow_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
    features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
    features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
    
    # Lag features (3)
    df_sorted = df.sort_values(['id_var', 'ds'])
    features_df['lag_1'] = df_sorted.groupby('id_var')['Quantity'].shift(1).fillna(0).values
    features_df['lag_2'] = df_sorted.groupby('id_var')['Quantity'].shift(2).fillna(0).values
    features_df['lag_7'] = df_sorted.groupby('id_var')['Quantity'].shift(7).fillna(0).values
    
    # Add original holiday features from dataset
    features_df['holiday_distance'] = df['cumdist']
    features_df['is_holiday'] = df['holiday']
    
    # Holiday features (15 pre-computed distances)
    features_df = pd.concat([features_df, holiday_df], axis=1)
    
    # Add SKU ID for LightGBM (needs it as categorical feature)
    features_df['sku_id'] = df['id_var'].astype('category').cat.codes
    
    return features_df

X_test_df = create_features(test_df, holiday_test)
# For neural network: exclude SKU (it's fed separately)
X_test_nn = X_test_df.drop(columns=['sku_id']).values
# For LightGBM: include SKU
X_test = X_test_df.values
y_test = test_df['Quantity'].values

# Encode SKUs
train_df = pd.read_csv('../../data/train_split.csv')
sku_categories = train_df['id_var'].astype('category').cat.categories
sku_test = pd.Categorical(test_df['id_var'], categories=sku_categories).codes

print(f"   ✓ Test samples: {len(X_test):,}")
print(f"   ✓ Features: {X_test.shape[1]}")
print(f"   ✓ Zero rate: {(y_test == 0).mean():.1%}")

# Scale features (using same scaler as training)
print("\n2. Scaling features...")
scaler = StandardScaler()
# Fit on train data
train_df = pd.read_csv('../../data/train_split.csv')
holiday_train = pd.read_csv('../../data/holiday_features_train.csv')
X_train_df = create_features(train_df, holiday_train)
# Fit scaler on features without SKU
scaler.fit(X_train_df.drop(columns=['sku_id']).values)
X_test_scaled = scaler.transform(X_test_nn)
print("   ✓ Features scaled")

# Load Lightweight model
print("\n3. Loading Lightweight Hierarchical Attention model...")
lightweight_model = keras.models.load_model('models/best_lightweight.keras', compile=False)
print(f"   ✓ Model loaded: {lightweight_model.count_params():,} parameters")

# Load LightGBM model
print("\n4. Loading LightGBM model...")
lgb_model = lgb.Booster(model_file='models/lightgbm_model.txt')
print("   ✓ LightGBM model loaded")

# Get predictions
print("\n5. Generating predictions...")
pred_lightweight = lightweight_model.predict([X_test_scaled, sku_test], verbose=0)
y_pred_lightweight = pred_lightweight['final_forecast'].flatten()

y_pred_lgb = lgb_model.predict(X_test)

print("   ✓ Predictions generated")

# Calculate metrics
def calculate_metrics(y_true, y_pred, model_name):
    """Calculate comprehensive metrics."""
    # Overall metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAPE (avoid division by zero)
    nonzero_mask = y_true > 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / 
                              y_true[nonzero_mask])) * 100
    else:
        mape = np.inf
    
    # Zero prediction accuracy
    zero_accuracy = np.mean((y_true == 0) == (y_pred < 0.5))
    
    # Non-zero metrics
    if nonzero_mask.sum() > 0:
        mae_nonzero = np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]))
        rmse_nonzero = np.sqrt(np.mean((y_true[nonzero_mask] - y_pred[nonzero_mask]) ** 2))
    else:
        mae_nonzero = 0
        rmse_nonzero = 0
    
    print(f"\n{model_name} Metrics:")
    print(f"  Overall:")
    print(f"    MAE:  {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAPE: {mape:.2f}%")
    print(f"  Zero Prediction Accuracy: {zero_accuracy:.2%}")
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

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

metrics_lightweight = calculate_metrics(y_test, y_pred_lightweight, "Lightweight Hierarchical Attention")
metrics_lgb = calculate_metrics(y_test, y_pred_lgb, "LightGBM")

# Comparison
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

mae_improvement = ((metrics_lgb['mae'] - metrics_lightweight['mae']) / metrics_lgb['mae']) * 100
rmse_improvement = ((metrics_lgb['rmse'] - metrics_lightweight['rmse']) / metrics_lgb['rmse']) * 100
mape_improvement = ((metrics_lgb['mape'] - metrics_lightweight['mape']) / metrics_lgb['mape']) * 100

print(f"\nLightweight vs LightGBM:")
print(f"  MAE:  {mae_improvement:+.2f}% {'✓ Better' if mae_improvement > 0 else '✗ Worse'}")
print(f"  RMSE: {rmse_improvement:+.2f}% {'✓ Better' if rmse_improvement > 0 else '✗ Worse'}")
print(f"  MAPE: {mape_improvement:+.2f}% {'✓ Better' if mape_improvement > 0 else '✗ Worse'}")

print(f"\nModel Size:")
print(f"  Lightweight: 54,805 parameters")
print(f"  LightGBM: {lgb_model.num_trees()} trees")

# Save results
results_df = pd.DataFrame({
    'Model': ['Lightweight Hierarchical Attention', 'LightGBM'],
    'MAE': [metrics_lightweight['mae'], metrics_lgb['mae']],
    'RMSE': [metrics_lightweight['rmse'], metrics_lgb['rmse']],
    'MAPE (%)': [metrics_lightweight['mape'], metrics_lgb['mape']],
    'Zero Accuracy': [metrics_lightweight['zero_accuracy'], metrics_lgb['zero_accuracy']],
    'MAE (non-zero)': [metrics_lightweight['mae_nonzero'], metrics_lgb['mae_nonzero']],
    'RMSE (non-zero)': [metrics_lightweight['rmse_nonzero'], metrics_lgb['rmse_nonzero']]
})

results_df.to_csv('lightweight_vs_lgb_comparison.csv', index=False)
print(f"\n✓ Results saved to: lightweight_vs_lgb_comparison.csv")

print("\n" + "=" * 70)
print("✅ Comparison Complete!")
print("=" * 70)
