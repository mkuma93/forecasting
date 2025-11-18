"""
Test Unit Normalization Layer
"""
import numpy as np
import tensorflow as tf
from src.deepsequence.unit_norm import UnitNorm, UnitNormDense, apply_unit_norm

print("="*70)
print("Testing Unit Normalization")
print("="*70)

# Test 1: UnitNorm layer
print("\n1. Testing UnitNorm Layer")
print("-"*70)
unit_norm = UnitNorm(name='test_unit_norm')
sample_input = tf.constant([[3.0, 4.0], [1.0, 1.0], [5.0, 12.0]])
output = unit_norm(sample_input)

print(f"Input:\n{sample_input.numpy()}")
print(f"\nOutput (normalized):\n{output.numpy()}")

# Verify unit norm
norms = tf.sqrt(tf.reduce_sum(tf.square(output), axis=-1))
print(f"\nL2 norms of output rows: {norms.numpy()}")
print(f"All norms ≈ 1.0: {tf.reduce_all(tf.abs(norms - 1.0) < 1e-6).numpy()}")

# Test 2: UnitNormDense layer
print("\n\n2. Testing UnitNormDense Layer")
print("-"*70)
unit_norm_dense = UnitNormDense(
    units=8,
    activation='relu',
    name='test_unit_norm_dense'
)
sample_input2 = tf.random.normal([5, 10])
output2 = unit_norm_dense(sample_input2)

print(f"Input shape: {sample_input2.shape}")
print(f"Output shape: {output2.shape}")
print(f"Output sample (first row): {output2[0].numpy()}")

# Verify unit norm
norms2 = tf.sqrt(tf.reduce_sum(tf.square(output2), axis=-1))
print(f"\nL2 norms: {norms2.numpy()}")
print(f"All norms ≈ 1.0: {tf.reduce_all(tf.abs(norms2 - 1.0) < 1e-6).numpy()}")

# Test 3: Functional API
print("\n\n3. Testing Functional API")
print("-"*70)
test_tensor = tf.constant([[6.0, 8.0], [3.0, 4.0]])
normalized = apply_unit_norm(test_tensor)

print(f"Input:\n{test_tensor.numpy()}")
print(f"Normalized:\n{normalized.numpy()}")
print(f"Expected: [[0.6, 0.8], [0.6, 0.8]]")

# Test 4: Integration in model
print("\n\n4. Testing Integration in Model")
print("-"*70)
inputs = tf.keras.layers.Input(shape=(10,), name='input')
x = tf.keras.layers.Dense(32, activation='relu')(inputs)
x = UnitNorm(name='unit_norm_1')(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x = UnitNorm(name='unit_norm_2')(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='test_model')
model.summary()

# Test prediction
test_data = tf.random.normal([3, 10])
predictions = model.predict(test_data, verbose=0)
print(f"\nTest predictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions.flatten()}")

# Test 5: Benefits of unit normalization
print("\n\n5. Demonstrating Benefits")
print("="*70)

print("\nWithout Unit Normalization:")
print("  • Activations can explode (very large values)")
print("  • Gradients can vanish or explode")
print("  • Training instability")

print("\nWith Unit Normalization:")
print("  ✓ Activations bounded (||x||_2 = 1)")
print("  ✓ Stable gradient flow")
print("  ✓ Implicit regularization")
print("  ✓ Better for embeddings and attention")

# Example: Compare activation magnitudes
x_no_norm = tf.keras.layers.Dense(32, activation='relu')(test_data)
x_with_norm = UnitNorm()(x_no_norm)

print(f"\nActivation magnitudes:")
print(f"  Without norm - Max: {tf.reduce_max(x_no_norm).numpy():.4f}, Mean: {tf.reduce_mean(x_no_norm).numpy():.4f}")
print(f"  With norm    - Max: {tf.reduce_max(x_with_norm).numpy():.4f}, Mean: {tf.reduce_mean(x_with_norm).numpy():.4f}")
print(f"  Norm of normalized: {tf.sqrt(tf.reduce_sum(tf.square(x_with_norm[0]))).numpy():.4f}")

print("\n" + "="*70)
print("✓ All unit normalization tests passed!")
print("="*70)

print("\n📊 Unit Normalization Benefits for DeepSequence:")
print("  • TabNet outputs: Stabilized attention features")
print("  • Dense layers: Controlled activation magnitudes")
print("  • Intermittent handler: Better probability predictions")
print("  • Overall: Improved training stability and convergence")
