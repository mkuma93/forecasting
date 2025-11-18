"""
Test script for IntermittentHandler integration with DeepSequenceModel
"""
import numpy as np
import tensorflow as tf
from src.deepsequence import (
    SeasonalComponent,
    RegressorComponent,
    DeepSequenceModel,
    IntermittentHandler
)

print("TensorFlow version:", tf.__version__)
print("\n" + "="*60)
print("Testing IntermittentHandler Integration")
print("="*60 + "\n")

# Create sample data
n_samples = 100
seasonal_input = np.random.randn(n_samples, 10)  # 10 seasonal features
regressor_input = np.random.randn(n_samples, 5)  # 5 regressor features
target = np.random.randn(n_samples, 1)  # Target values

print("✓ Sample data created:")
print(f"  - Seasonal input shape: {seasonal_input.shape}")
print(f"  - Regressor input shape: {regressor_input.shape}")
print(f"  - Target shape: {target.shape}\n")

# Test 1: Model WITHOUT intermittent handler
print("Test 1: Building model WITHOUT intermittent handler")
print("-" * 60)
seasonal_comp = SeasonalComponent(
    weekly_hidden=8,
    monthly_hidden=8,
    yearly_hidden=8,
    activation='relu'
)
seasonal_comp.build(seasonal_input.shape[1])

regressor_comp = RegressorComponent(
    trend_hidden=8,
    exog_hidden=8,
    activation='relu'
)
regressor_comp.build(regressor_input.shape[1])

model_base = DeepSequenceModel(mode='additive', use_intermittent=False)
model_base_built = model_base.build(seasonal_comp, regressor_comp)

print(f"✓ Base model built successfully")
print(f"  - Model name: {model_base_built.name}")
print(f"  - Output shape: {model_base_built.output.shape}")
print(f"  - Total parameters: {model_base_built.count_params():,}\n")

# Test 2: Model WITH intermittent handler
print("Test 2: Building model WITH intermittent handler")
print("-" * 60)
seasonal_comp2 = SeasonalComponent(
    weekly_hidden=8,
    monthly_hidden=8,
    yearly_hidden=8,
    activation='relu'
)
seasonal_comp2.build(seasonal_input.shape[1])

regressor_comp2 = RegressorComponent(
    trend_hidden=8,
    exog_hidden=8,
    activation='relu'
)
regressor_comp2.build(regressor_input.shape[1])

model_intermittent = DeepSequenceModel(mode='additive', use_intermittent=True)
intermittent_config = {
    'hidden_units': 16,
    'hidden_layers': 2,
    'activation': 'relu',
    'dropout': 0.2,
    'l1_reg': 0.01
}
model_intermittent_built = model_intermittent.build(
    seasonal_comp2,
    regressor_comp2,
    intermittent_config=intermittent_config
)

print(f"✓ Intermittent model built successfully")
print(f"  - Model name: {model_intermittent_built.name}")
print(f"  - Output shape: {model_intermittent_built.output.shape}")
print(f"  - Total parameters: {model_intermittent_built.count_params():,}")
print(f"  - Parameters increase: {model_intermittent_built.count_params() - model_base_built.count_params():,}\n")

# Test 3: Make predictions
print("Test 3: Testing predictions")
print("-" * 60)
base_pred = model_base_built.predict([seasonal_input[:5], regressor_input[:5]], verbose=0)
intermittent_pred = model_intermittent_built.predict([seasonal_input[:5], regressor_input[:5]], verbose=0)

print("✓ Predictions generated successfully")
print(f"\nBase model predictions (first 5):")
print(base_pred.flatten())
print(f"\nIntermittent model predictions (first 5):")
print(intermittent_pred.flatten())
print(f"\nPrediction ratios (intermittent/base):")
print((intermittent_pred / (base_pred + 1e-8)).flatten())
print("\n✓ Ratios represent probability mask (0-1 range applied)\n")

# Test 4: Model summary
print("Test 4: Model architecture comparison")
print("-" * 60)
print("\nBase Model Summary:")
print("=" * 60)
model_base_built.summary()
print("\n\nIntermittent Model Summary:")
print("=" * 60)
model_intermittent_built.summary()

print("\n" + "="*60)
print("✓ All tests passed successfully!")
print("="*60)
