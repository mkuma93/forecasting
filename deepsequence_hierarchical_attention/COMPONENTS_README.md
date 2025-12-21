# Hierarchical Attention Components

This package provides **two separate implementations** of hierarchical attention for intermittent demand forecasting:

---

## 1. **Lightweight Hierarchical Attention** (Production)

**Files:**
- `components_lightweight.py` (91KB, 2,677 lines)

**Function:**
```python
from deepsequence_hierarchical_attention import build_hierarchical_model_lightweight

model = build_hierarchical_model_lightweight(
    num_features=29,
    num_skus=6099,
    embedding_dim=32,
    hidden_units=64
)
```

**Characteristics:**
- ✅ Optimized for production use
- ✅ Fast training and inference
- ✅ Lower memory footprint
- ✅ Simple dense layers with attention
- ✅ Multi-output: classification + forecasting
- 📊 **Model size:** ~1MB
- ⚡ **Training speed:** ~18ms/step

**Best for:**
- Production deployments
- Real-time inference
- Resource-constrained environments
- Quick iterations

---

## 2. **TabNet-Based Hierarchical Attention** (Research)

**Files:**
- `components.py` (1,570 lines) - Component builders with PWL calibration
- `tabnet.py` (429 lines) - TabNet encoder implementation  
- `model.py` (211 lines) - Model creation utilities

**Function:**
```python
from deepsequence_hierarchical_attention import (
    create_hierarchical_model,
    DeepSequencePWLHierarchical
)

model = create_hierarchical_model(
    num_features=29,
    num_skus=6099,
    embedding_dim=64,
    hidden_units=128,
    use_tabnet=True
)
```

**Characteristics:**
- 🔬 Advanced feature selection
- 📈 Piecewise linear calibration
- 🎯 Attentive feature transformer
- 🌲 Ghost batch normalization
- 🔍 Interpretable attention weights
- 📊 **Model size:** ~4MB
- ⏱️ **Training speed:** ~45ms/step

**Best for:**
- Research and experimentation
- Maximum accuracy
- Feature importance analysis
- Complex non-linear patterns

---

## Comparison

| Feature | Lightweight | TabNet-Based |
|---------|------------|--------------|
| **Model Size** | ~1MB | ~4MB |
| **Training Speed** | 18ms/step | 45ms/step |
| **Memory Usage** | Low | Medium-High |
| **Accuracy** | High | Higher |
| **Interpretability** | Moderate | High |
| **Production Ready** | ✅ Yes | ⚠️ Requires tuning |

---

## Usage Examples

### Lightweight (Current Production Model)

```python
# Train lightweight model
from deepsequence_hierarchical_attention.examples import train_lightweight_adaptive_loss

# Uses: components_lightweight.py
# Training script: examples/train_lightweight_adaptive_loss.py
```

### TabNet-Based (Research Model)

```python
# Train TabNet-based model
from deepsequence_hierarchical_attention import (
    create_hierarchical_model,
    compile_hierarchical_model,
    get_training_callbacks
)

model = create_hierarchical_model(
    num_features=29,
    num_skus=6099,
    use_tabnet=True
)

model = compile_hierarchical_model(model)
```

---

## Component Independence

Both implementations are **completely independent**:
- No shared dependencies between implementations
- Can be used separately without importing the other
- Different hyperparameters and architectures
- Separate training scripts and configurations

---

## Recommendation

- **Start with:** Lightweight implementation
- **Upgrade to:** TabNet-based if you need:
  - Better feature importance analysis
  - Higher accuracy (at cost of speed)
  - Advanced interpretability

---

## Files Structure

```
deepsequence_hierarchical_attention/
├── components_lightweight.py  ← Lightweight implementation
├── components.py             ← TabNet hierarchical components
├── tabnet.py                 ← TabNet encoder
├── model.py                  ← Model creation utilities
├── losses.py                 ← Shared loss functions
└── __init__.py              ← Exports both implementations
```
