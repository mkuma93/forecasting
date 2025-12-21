"""
Feature Configuration Loader and Validator
Ensures all models use the exact same feature specification.
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path


class FeatureConfig:
    """Load and validate feature configuration from YAML."""
    
    def __init__(self, config_path='../feature_config.yaml'):
        config_path = Path(config_path)
        if not config_path.exists():
            # Try alternative path
            config_path = Path(__file__).parent.parent / 'feature_config.yaml'
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration is complete and consistent."""
        # Check all sections exist
        required_sections = ['cyclical_features', 'lag_features', 
                           'holiday_features', 'model_architecture', 
                           'feature_order', 'metadata']
        # trend_features is optional for backward compatibility
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
        
        # Verify total count
        expected_total = self.config['metadata']['total_features']
        actual_total = len(self.config['feature_order'])
        if expected_total != actual_total:
            raise ValueError(
                f"Feature count mismatch: expected {expected_total}, "
                f"got {actual_total}"
            )
        
        # Verify indices are sequential
        all_features = []
        if 'trend_features' in self.config:
            all_features.extend(self.config['trend_features'])
        all_features.extend(self.config['cyclical_features'])
        all_features.extend(self.config['lag_features'])
        all_features.extend(self.config['holiday_features'])
        if 'binary_holiday_features' in self.config:
            all_features.extend(self.config['binary_holiday_features'])
        
        indices = [f['index'] for f in all_features]
        expected_indices = list(range(len(all_features)))
        if indices != expected_indices:
            raise ValueError("Feature indices are not sequential")
    
    @property
    def total_features(self):
        """Total number of features."""
        return self.config['metadata']['total_features']
    
    @property
    def cyclical_names(self):
        """List of cyclical feature names in order."""
        return [f['name'] for f in self.config['cyclical_features']]
    
    @property
    def lag_names(self):
        """List of lag feature names in order."""
        return [f['name'] for f in self.config['lag_features']]
    
    @property
    def holiday_names(self):
        """List of holiday distance feature names in order."""
        return [f['name'] for f in self.config['holiday_features']]
    
    @property
    def binary_holiday_names(self):
        """List of binary holiday feature names in order."""
        if 'binary_holiday_features' in self.config:
            return [f['name'] for f in self.config['binary_holiday_features']]
        return []
    
    @property
    def feature_names(self):
        """All feature names in correct order."""
        return self.config['feature_order']
    
    @property
    def trend_indices(self):
        """Indices for trend component."""
        if 'trend_component' in self.config['model_architecture']:
            return self.config['model_architecture']['trend_component']['feature_indices']
        return []
    
    @property
    def seasonal_indices(self):
        """Indices for seasonal component."""
        return self.config['model_architecture']['seasonal_component']['feature_indices']
    
    @property
    def regressor_indices(self):
        """Indices for regressor component."""
        return self.config['model_architecture']['regressor_component']['feature_indices']
    
    @property
    def holiday_indices(self):
        """Indices for holiday component."""
        return self.config['model_architecture']['holiday_component']['feature_indices']
    
    def create_features(self, df, holiday_features_df):
        """
        Create features according to config specification.
        
        Args:
            df: DataFrame with columns ['ds', 'id_var', 'Quantity']
            holiday_features_df: DataFrame with holiday distance features
            
        Returns:
            DataFrame with features in correct order
        """
        df = df.sort_values(['id_var', 'ds']).reset_index(drop=True)
        features = {}
        
        # Create trend features from config
        for trend_feature in self.config['trend_features']:
            name = trend_feature['name']
            source_col = trend_feature['source_column']
            transformation = trend_feature['transformation']
            
            if transformation == 'days_since_epoch':
                # Convert date column to days since Unix epoch 1970-01-01
                epoch = pd.Timestamp('1970-01-01')
                features[name] = (df[source_col] - epoch).dt.days.values
            else:
                raise ValueError(f"Unknown transformation: {transformation}")
        
        # Extract time components
        day_of_week = df['ds'].dt.dayofweek.values
        month = df['ds'].dt.month.values
        day_of_year = df['ds'].dt.dayofyear.values
        
        # Create cyclical features
        for feat_config in self.config['cyclical_features']:
            name = feat_config['name']
            if 'dow' in name:
                period = 7
                value = day_of_week
            elif 'month' in name:
                period = 12
                value = month
            elif 'year' in name:
                period = 365.25
                value = day_of_year
            
            if 'sin' in name:
                features[name] = np.sin(2 * np.pi * value / period)
            else:  # cos
                features[name] = np.cos(2 * np.pi * value / period)
        
        # Create lag features
        for feat_config in self.config['lag_features']:
            name = feat_config['name']
            lag = feat_config['lag']
            features[name] = (df.groupby('id_var')['Quantity']
                            .shift(lag)
                            .fillna(0)
                            .values)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        
        # Add holiday distance features (must be in correct order)
        expected_holidays = self.holiday_names
        actual_holidays = [col for col in holiday_features_df.columns 
                          if col.startswith('days_from_')]
        
        # Verify all holidays present
        missing = set(expected_holidays) - set(actual_holidays)
        if missing:
            raise ValueError(f"Missing holiday features: {missing}")
        
        # Add distance features in correct order
        holiday_subset = holiday_features_df[expected_holidays].reset_index(drop=True)
        features_df = pd.concat([features_df, holiday_subset], axis=1)
        
        # Add binary holiday features
        binary_holiday_names = self.binary_holiday_names
        if binary_holiday_names:
            # Create binary indicators: 1 if distance is 0, else 0
            for dist_name, binary_name in zip(expected_holidays, binary_holiday_names):
                features_df[binary_name] = (holiday_features_df[dist_name] == 0).astype(int).values
        
        # Final validation
        if list(features_df.columns) != self.feature_names:
            raise ValueError(
                f"Feature order mismatch!\n"
                f"Expected: {self.feature_names}\n"
                f"Got: {list(features_df.columns)}"
            )
        
        return features_df
    
    def validate_features(self, features_df):
        """
        Validate that a DataFrame has correct features.
        
        Args:
            features_df: DataFrame to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If validation fails
        """
        # Check column names
        expected_cols = self.feature_names
        actual_cols = list(features_df.columns)
        
        if actual_cols != expected_cols:
            raise ValueError(
                f"Feature validation failed!\n"
                f"Expected columns: {expected_cols}\n"
                f"Actual columns: {actual_cols}\n"
                f"Missing: {set(expected_cols) - set(actual_cols)}\n"
                f"Extra: {set(actual_cols) - set(expected_cols)}"
            )
        
        # Check number of features
        if len(actual_cols) != self.total_features:
            raise ValueError(
                f"Expected {self.total_features} features, "
                f"got {len(actual_cols)}"
            )
        
        return True
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 80)
        print(f"FEATURE CONFIGURATION v{self.config['metadata']['version']}")
        print("=" * 80)
        print(f"\nTotal Features: {self.total_features}")
        print(f"Last Updated: {self.config['metadata']['last_updated']}")
        print(f"\nContext: {self.config['metadata']['dataset_context']}")
        
        print("\n" + "-" * 80)
        print(f"CYCLICAL FEATURES ({len(self.cyclical_names)})")
        print("-" * 80)
        for i, name in enumerate(self.cyclical_names):
            print(f"  [{i}] {name}")
        
        print("\n" + "-" * 80)
        print(f"LAG FEATURES ({len(self.lag_names)})")
        print("-" * 80)
        offset = len(self.cyclical_names)
        for i, name in enumerate(self.lag_names):
            print(f"  [{offset + i}] {name}")
        
        print("\n" + "-" * 80)
        print(f"HOLIDAY FEATURES ({len(self.holiday_names)})")
        print("-" * 80)
        offset = len(self.cyclical_names) + len(self.lag_names)
        for i, name in enumerate(self.holiday_names):
            print(f"  [{offset + i}] {name}")
        
        print("\n" + "-" * 80)
        print("MODEL ARCHITECTURE MAPPING")
        print("-" * 80)
        print(f"  Seasonal Component: indices {self.seasonal_indices}")
        print(f"  Regressor Component: indices {self.regressor_indices}")
        print(f"  Holiday Component: indices {self.holiday_indices}")
        print("=" * 80)


# Convenience function
def load_feature_config(config_path='../feature_config.yaml'):
    """Load feature configuration."""
    return FeatureConfig(config_path)


if __name__ == '__main__':
    # Test loading config
    config = load_feature_config()
    config.print_summary()
    
    print("\n✅ Feature configuration loaded and validated successfully!")
