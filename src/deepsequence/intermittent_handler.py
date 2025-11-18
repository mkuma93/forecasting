"""
Intermittent Handler for DeepSequence.

Handles sparse/intermittent demand by predicting probability of non-zero demand.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l1


class IntermittentHandler:
    """
    Intermittent demand handler that predicts probability of non-zero demand.
    
    Takes inputs from both seasonal and regressor components and outputs
    a probability (0-1) that gets multiplied with the final forecast.
    """
    
    def __init__(self,
                 hidden_units: int = 32,
                 hidden_layers: int = 2,
                 activation: str = 'relu',
                 dropout: float = 0.2,
                 l1_reg: float = 0.01):
        """
        Initialize intermittent handler.
        
        Args:
            hidden_units: Number of units in hidden layers
            hidden_layers: Number of hidden layers
            activation: Activation function for hidden layers
            dropout: Dropout rate
            l1_reg: L1 regularization factor
        """
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.l1_reg = l1_reg
        self.model = None
    
    def build_model(self, seasonal_inputs, regressor_inputs):
        """
        Build the intermittent handler model.
        
        Args:
            seasonal_inputs: List of seasonal input layers
            regressor_inputs: List of regressor input layers
            
        Returns:
            Keras Model that outputs probability (0-1)
        """
        # Concatenate all inputs
        if isinstance(seasonal_inputs, list):
            concat_seasonal = layers.Concatenate()(seasonal_inputs) if len(seasonal_inputs) > 1 else seasonal_inputs[0]
        else:
            concat_seasonal = seasonal_inputs
            
        if isinstance(regressor_inputs, list):
            concat_regressor = layers.Concatenate()(regressor_inputs) if len(regressor_inputs) > 1 else regressor_inputs[0]
        else:
            concat_regressor = regressor_inputs
        
        # Combine seasonal and regressor features
        combined = layers.Concatenate(name='intermittent_concat')([
            concat_seasonal, concat_regressor
        ])
        
        # Hidden layers
        x = combined
        for i in range(self.hidden_layers):
            x = layers.Dense(
                self.hidden_units,
                activation=self.activation,
                kernel_regularizer=l1(self.l1_reg),
                name=f'intermittent_hidden_{i+1}'
            )(x)
            
            if self.dropout > 0:
                x = layers.Dropout(self.dropout, name=f'intermittent_dropout_{i+1}')(x)
        
        # Output layer with sigmoid to get probability (0-1)
        probability = layers.Dense(
            1,
            activation='sigmoid',
            name='intermittent_probability'
        )(x)
        
        # Create model
        all_inputs = seasonal_inputs + regressor_inputs if isinstance(seasonal_inputs, list) else [seasonal_inputs, regressor_inputs]
        self.model = Model(
            inputs=all_inputs,
            outputs=probability,
            name='intermittent_handler'
        )
        
        return self.model
    
    def get_probability_layer(self):
        """
        Get the probability output layer for integration with main model.
        
        Returns:
            Output layer (probability)
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        return self.model.output


def apply_intermittent_mask(forecast_output, probability_output):
    """
    Apply intermittent mask to forecast by multiplying with probability.
    
    Args:
        forecast_output: Base forecast from DeepSequence
        probability_output: Probability of non-zero demand (0-1)
        
    Returns:
        Masked forecast (forecast * probability)
    """
    return layers.Multiply(name='intermittent_masked_forecast')([
        forecast_output, probability_output
    ])
