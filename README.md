# DeepFuture Net - SKU Level Forecasting

A machine learning project for forecasting at SKU (Stock Keeping Unit) level using various techniques including LightGBM models and deep learning approaches.

## Project Overview

This project aims to forecast stock demand at the SKU level using historical data. Multiple modeling approaches have been explored and compared:

- **DeepFuture Net** ⭐: A custom deep learning architecture inspired by Prophet, designed specifically for SKU-level forecasting with seasonal patterns
- **LightGBM Models**: Cluster-based and distance-based forecasting approaches
- **Baseline Models**: Naive forecasting methods for comparison and benchmarking

### Key Innovation: DeepFuture Net

DeepFuture Net is an original deep learning architecture that combines:
- **Seasonal Components**: Inspired by Prophet's additive model for capturing weekly, monthly, and yearly seasonality
- **Recurrent Components**: LSTM/GRU layers for temporal dependencies
- **Contextual Features**: Cluster-based and exogenous variables integration
- **Multi-horizon Forecasting**: Direct prediction of multiple future time steps

This custom architecture was developed to handle the unique challenges of retail SKU forecasting, including intermittent demand and multiple seasonal patterns.

## Project Structure

```
forecasting/
├── src/deepfuture/              # Core DeepFuture Net package
│   ├── __init__.py
│   ├── model.py                 # Main model class
│   ├── seasonal_component.py   # Seasonal decomposition
│   ├── regressor_component.py  # Regression component
│   ├── utils.py                 # Utility functions
│   ├── activations.py          # Custom activations
│   └── config.py               # Configuration
│
├── notebooks/
│   └── DeepFuture_Demo.ipynb   # End-to-end demo
│
├── scripts/
│   └── fix_notebooks.py        # Notebook utility scripts
│
├── README.md
├── ARCHITECTURE.md             # Technical architecture docs
├── PERFORMANCE_COMPARISON.md   # Model benchmarks
├── TEST_REPORT.md             # Validation report
├── LICENSE                    # MIT License
└── requirements.txt           # Dependencies
```

## Requirements

```bash
# Core dependencies
pandas
numpy
scikit-learn
lightgbm
tensorflow  # or pytorch
matplotlib
seaborn
jupyter
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mkuma93/forecasting.git
cd forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with Demo Notebook

```bash
jupyter notebook notebooks/DeepFuture_Demo.ipynb
```

### Using DeepFuture Net in Your Code

```python
import sys
sys.path.insert(0, 'src')
from deepfuture import DeepFutureModel, SeasonalComponent, RegressorComponent

# Prepare your time series data
# ts: DataFrame with columns ['ds', 'id_cat', 'target_variable']
# exog: DataFrame with exogenous variables

# Build seasonal component
seasonal = SeasonalComponent(
    data=ts, target=['target'], id_var='id_cat',
    horizon=8, weekly=True, monthly=True, yearly=True
)
seasonal.seasonal_feature()
seasonal.seasonal_model(hidden=2, hidden_unit=32)

# Build regressor component
regressor = RegressorComponent(
    ts=ts, exog=exog, target=['target'], id_var='id_cat',
    categorical_var=['cluster'], context_variable=['price', 'lag1']
)
regressor.reg_model(id_input=seasonal.s_model.input[-1])

# Combine and train
model = DeepFutureModel(mode='additive')
model.build(seasonal, regressor)
model.compile(loss='mape', learning_rate=0.001)
history = model.fit(train_input, train_target, epochs=50)

# Predict
predictions = model.predict(test_input)
```

## Models

### DeepFuture Net (Custom Architecture) ⭐
**Original contribution** - A Prophet-inspired deep learning architecture featuring:
- Seasonal decomposition modules (weekly, monthly, yearly)
- Recurrent regression components with LSTM/GRU
- Embedding layers for categorical features (StockCode, clusters)
- Constraint handling for business logic
- Multi-horizon forecasting capability

**Architecture Highlights**:
- Modular design with separate seasonal and regression components
- Configurable hidden layers and activation functions
- L1 regularization for feature selection
- Early stopping and model checkpointing
- Support for exogenous variables (price, holidays, clusters)

### LightGBM Models
- **Cluster-based**: Groups similar SKUs and forecasts by cluster
- **Zero/Non-zero distance**: Handles intermittent demand patterns
- Lag features and distance-to-zero variables

### Baseline
- Naive shift-7 method for comparison
- Benchmark for model performance evaluation

## Results

### Performance Summary

An **ensemble approach** combining all three models achieves the best overall performance by selecting the optimal model for each SKU based on validation MAPE.

| Model | Typical Use Case | Validation MAPE Range |
|-------|-----------------|---------------------|
| **DeepFuture Net** | High-volume, complex seasonality | 145-310% |
| **LightGBM Cluster** | Medium-volume, stable patterns | 195-275% |
| **LightGBM Distance** | Low-volume, intermittent demand | 240-280% |
| **Ensemble** | All SKUs (best per-SKU selection) | **~180-220%** |

*Note: MAPE values are high due to intermittent demand with many zero/near-zero values - expected for retail SKU forecasting.*

**📊 Detailed Comparison**: See [PERFORMANCE_COMPARISON.md](PERFORMANCE_COMPARISON.md) for comprehensive analysis.

### Output Files
- Final forecasts: `final_forecast.csv` (ensemble predictions)
- Per-model MAPE: `lgb_cluster_mape.csv`, `lgbnon-zerointerval_mape.csv`, `non-zero-mean_df.csv`
- Individual forecasts: `deep_future_forecast.csv`, `lgb_clusterdistanceforecast.csv`, `lgb_zerodistanceforecast.csv`

## Data

**Note**: Data files are not included in this repository. 

To use this project with your own data:
1. Prepare your time series data with columns: `ds` (date), `StockCode` (SKU ID), `Quantity` (target)
2. Add exogenous variables (optional): price, clusters, holidays, etc.
3. Follow the demo notebook for complete workflow
4. See `ARCHITECTURE.md` for detailed data requirements

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contact

Mritunjay Kumar - [GitHub](https://github.com/mkuma93)

## Citation

If you use DeepFuture Net or find this work helpful, please cite:

```
@misc{deepfuture_net,
  author = {Mritunjay Kumar},
  title = {DeepFuture Net: A Prophet-Inspired Deep Learning Architecture for SKU-Level Forecasting},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/mkuma93/forecasting}
}
```

## Acknowledgments

- **DeepFuture Net**: Original architecture designed by Mritunjay Kumar, inspired by Facebook's Prophet
- Built for retail SKU-level forecasting with intermittent demand patterns
- Combines deep learning with seasonal decomposition methodology
