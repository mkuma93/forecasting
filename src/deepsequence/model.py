"""
Main DeepSequence model combining seasonal and regression components.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import List
import numpy as np

from .seasonal_component import SeasonalComponent
from .regressor_component import RegressorComponent
from .intermittent_handler import IntermittentHandler, apply_intermittent_mask
from .config import DEFAULT_LEARNING_RATE, TRAINING_PARAMS


class DeepSequenceModel:
    """
    DeepSequence: Complete forecasting model combining seasonal and
    regression components.
    """

    def __init__(self, mode: str = 'additive', use_intermittent: bool = False):
        self.mode = mode
        self.use_intermittent = use_intermittent
        self.seasonal_model = None
        self.regressor_model = None
        self.intermittent_handler = None
        self.full_model = None

    def build(self, seasonal_component: SeasonalComponent, regressor_component: RegressorComponent,
              intermittent_config: dict = None):
        self.seasonal_model = seasonal_component.s_model
        self.regressor_model = regressor_component.combined_reg_model

        if self.mode == 'additive':
            combined_output = layers.Add(name='additive_forecast')([
                self.seasonal_model.output, self.regressor_model.output
            ])
        else:
            combined_output = layers.Multiply(name='multiplicative_forecast')([
                self.seasonal_model.output, self.regressor_model.output
            ])

        # Apply intermittent handler if enabled
        if self.use_intermittent:
            if intermittent_config is None:
                intermittent_config = {}
            
            # Create intermittent handler
            self.intermittent_handler = IntermittentHandler(**intermittent_config)
            
            # Build handler model using same inputs
            probability = self.intermittent_handler.build_model(
                self.seasonal_model.input,
                self.regressor_model.input
            )
            
            # Apply mask: multiply forecast with probability
            combined_output = apply_intermittent_mask(combined_output, probability.output)

        self.full_model = Model(
            inputs=[self.seasonal_model.input, self.regressor_model.input],
            outputs=combined_output,
            name='deepsequence_net'
        )

        return self.full_model

    def compile(self, loss='mape', learning_rate: float = DEFAULT_LEARNING_RATE):
        if loss == 'mape':
            loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
        elif loss == 'mae':
            loss_fn = tf.keras.losses.MeanAbsoluteError()
        elif loss == 'mse':
            loss_fn = tf.keras.losses.MeanSquaredError()
        else:
            loss_fn = loss

        self.full_model.compile(
            loss=loss_fn,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )

    def fit(self, train_input: List, train_target: np.ndarray, val_input: List = None, val_target: np.ndarray = None,
            epochs: int = 500, batch_size: int = 512, checkpoint_path: str = None, patience: int = 10, verbose: int = 1):
        callbacks = []
        early_stop = EarlyStopping(
            monitor='val_loss' if val_input else 'loss',
            mode=TRAINING_PARAMS['mode'],
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stop)

        if checkpoint_path:
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss' if val_input else 'loss',
                save_best_only=TRAINING_PARAMS['save_best_only'],
                mode=TRAINING_PARAMS['mode']
            )
            callbacks.append(checkpoint)

        validation_data = None
        if val_input is not None and val_target is not None:
            validation_data = (val_input, val_target)

        history = self.full_model.fit(
            train_input,
            train_target,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, inputs: List) -> np.ndarray:
        return self.full_model.predict(inputs)

    def save(self, path: str):
        self.full_model.save(path)

    @staticmethod
    def load(path: str, custom_objects: dict = None):
        model = DeepSequenceModel()
        model.full_model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        return model

    def summary(self):
        if self.full_model:
            self.full_model.summary()
        else:
            print("Model not built yet. Call build() first.")
