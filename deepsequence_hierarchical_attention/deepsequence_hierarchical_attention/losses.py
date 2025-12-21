"""
Custom loss functions for DeepSequence Hierarchical Attention.

Includes composite loss optimized for intermittent demand forecasting.
"""

import tensorflow as tf


def composite_loss(alpha=0.5, bce_weight=None, zero_rate=None, average_nonzero_demand=None, 
                   false_negative_weight=5.0):
    """
    Composite loss with data-driven dynamic weighting and costly false negatives.
    
    Optimized for intermittent demand with automatic weight balancing:
    - BCE weight: zero_rate (e.g., 0.90 for 90% zeros)
    - MAE weight: (1 - zero_rate) * log1p(avg_nonzero) - scales by demand magnitude
    - False negative penalty: Make predicting zero for non-zero costly
    
    The log1p scaling ensures MAE weight increases with demand magnitude:
    - Low avg demand (e.g., 2): log1p(2) = 1.10 → modest MAE weight
    - Medium avg demand (e.g., 6.53): log1p(6.53) = 2.02 → ~2x MAE weight
    - High avg demand (e.g., 50): log1p(50) = 3.93 → ~4x MAE weight
    
    This automatically balances zero classification vs magnitude prediction
    based on both sparsity AND magnitude of the data.
    
    Args:
        alpha: Not used, kept for backward compatibility
        bce_weight: If provided, overrides zero_rate for BCE weight
        zero_rate: Fraction of zeros in training data (e.g., 0.90)
        average_nonzero_demand: Average of non-zero demand values (e.g., 6.53)
            Used to scale MAE weight by demand magnitude via log1p
        false_negative_weight: Weight for false negatives (default 5.0)
            Higher = more penalty for predicting zero when actual is non-zero
    
    Returns:
        Loss function compatible with Keras model.compile()
    
    Example:
        >>> zero_rate = (train_df['Quantity'] == 0).mean()  # e.g., 0.90
        >>> avg_nonzero = train_df[train_df['Quantity'] > 0]['Quantity'].mean()  # e.g., 6.53
        >>> model.compile(
        ...     optimizer='adam',
        ...     loss={'final_forecast': composite_loss(
        ...         zero_rate=zero_rate,
        ...         average_nonzero_demand=avg_nonzero,
        ...         false_negative_weight=5.0  # Penalize false negatives
        ...     )},
        ...     metrics=['mae']
        ... )
    """
    # Compute weights from data statistics (cast to float32 for TensorFlow)
    if bce_weight is not None:
        # Use provided BCE weight
        final_bce_weight = tf.cast(bce_weight, tf.float32)
        final_mae_weight = tf.cast(1.0 - bce_weight, tf.float32)
        # Apply log1p scaling if avg_nonzero provided
        if average_nonzero_demand is not None:
            log_scale = tf.math.log1p(tf.cast(average_nonzero_demand, tf.float32))
            final_mae_weight = final_mae_weight * log_scale
    elif zero_rate is not None:
        # Data-driven weighting based on zero rate
        # BCE weight = % zeros, MAE weight = % non-zeros * log1p(avg_nonzero)
        final_bce_weight = tf.cast(zero_rate, tf.float32)
        final_mae_weight = tf.cast(1.0 - zero_rate, tf.float32)
        # Apply log1p scaling if avg_nonzero provided
        if average_nonzero_demand is not None:
            log_scale = tf.math.log1p(tf.cast(average_nonzero_demand, tf.float32))
            final_mae_weight = final_mae_weight * log_scale
    else:
        # Fallback to defaults
        final_bce_weight = tf.constant(0.90, dtype=tf.float32)
        final_mae_weight = tf.constant(0.10, dtype=tf.float32)
    
    def loss_fn(y_true, y_pred):
        """
        Compute composite loss with static weighting.
        
        BCE and MAE are weighted by predefined constants.
        MAE is computed only on non-zero demand samples.
        
        Args:
            y_true: True demand values (shape: [batch_size,] or [batch_size, 1])
            y_pred: Predicted demand values (shape: [batch_size,] or [batch_size, 1])
        
        Returns:
            Scalar loss value
        """
        # Flatten inputs to ensure consistent shapes
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Binary target: 1 if demand > 0, else 0
        y_binary = tf.cast(y_true > 0, tf.float32)
        
        # Count non-zero samples
        n_nonzero = tf.reduce_sum(y_binary)
        
        # y_pred is already a probability (0-1) from model's sigmoid output
        # Clip to avoid log(0) errors
        epsilon = 1e-7
        y_pred_binary = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Weighted binary cross-entropy loss
        # Penalize false negatives (predicting zero when actual is non-zero)
        # pos_weight > 1: higher penalty for missing non-zero samples
        fn_weight = tf.constant(false_negative_weight, dtype=tf.float32)
        
        # Compute weighted BCE manually
        # Loss = -(y*log(p)*fn_weight + (1-y)*log(1-p))
        # epsilon already applied via clip_by_value above
        bce_loss = -(
            y_binary * tf.math.log(y_pred_binary) * fn_weight +
            (1 - y_binary) * tf.math.log(1 - y_pred_binary)
        )
        
        # Mean absolute error loss (magnitude prediction)
        # MASKED: Only compute MAE on non-zero samples
        non_zero_mask = y_binary
        
        # Extract non-zero predictions and targets for MAE
        # y_pred should represent base_forecast (before zero probability adjustment)
        mae_loss = tf.abs(y_true - y_pred) * non_zero_mask
        
        # Average MAE only over non-zero samples
        mae_loss_avg = tf.reduce_sum(mae_loss) / (n_nonzero + 1e-7)
        
        # Combined loss: bce_weight * BCE + mae_weight * MAE_nonzero
        combined = final_bce_weight * tf.reduce_mean(bce_loss) + final_mae_weight * mae_loss_avg
        
        return combined
    
    return loss_fn


def base_forecast_mse(weight=0.4):
    """
    MSE loss for base_forecast (auxiliary task).
    
    Provides direct supervision for base_forecast to learn proper magnitude,
    independent of zero_probability. Only computed on non-zero samples.
    
    Theory: Multi-task learning to decouple magnitude prediction from classification.
    - Base forecast learns: "What is the demand when non-zero?"
    - Zero probability learns: "Will there be demand?"
    
    Args:
        weight: Weight for the MSE loss (default 0.4)
    
    Returns:
        Loss function compatible with Keras model.compile()
    """
    final_weight = tf.constant(weight, dtype=tf.float32)
    
    def loss_fn(y_true, y_pred):
        """
        Compute MSE on base_forecast for non-zero samples only.
        
        Args:
            y_true: True demand values
            y_pred: Base forecast predictions (before zero probability adjustment)
        
        Returns:
            Scalar MSE loss
        """
        # Flatten inputs
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Mask to non-zero samples only
        non_zero_mask = tf.cast(y_true > 0, tf.float32)
        n_nonzero = tf.reduce_sum(non_zero_mask) + 1e-7
        
        # MSE on non-zero samples
        squared_error = tf.square(y_true - y_pred) * non_zero_mask
        mse = tf.reduce_sum(squared_error) / n_nonzero
        
        return final_weight * mse
    
    return loss_fn


def base_forecast_mae(weight=0.4):
    """
    MAE loss for base_forecast (auxiliary task).
    
    Provides direct supervision for base_forecast to learn proper magnitude,
    independent of zero_probability. Only computed on non-zero samples.
    
    Theory: Multi-task learning to decouple magnitude prediction from classification.
    - Base forecast learns: "What is the demand when non-zero?"
    - Zero probability learns: "Will there be demand?"
    
    Args:
        weight: Weight for the MAE loss (default 0.4)
    
    Returns:
        Loss function compatible with Keras model.compile()
    """
    final_weight = tf.constant(weight, dtype=tf.float32)
    
    def loss_fn(y_true, y_pred):
        """
        Compute MAE on base_forecast for non-zero samples only.
        
        Args:
            y_true: True demand values
            y_pred: Base forecast predictions (before zero probability adjustment)
        
        Returns:
            Scalar MAE loss
        """
        # Flatten inputs
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Mask to non-zero samples only
        non_zero_mask = tf.cast(y_true > 0, tf.float32)
        n_nonzero = tf.reduce_sum(non_zero_mask) + 1e-7
        
        # MAE on non-zero samples
        absolute_error = tf.abs(y_true - y_pred) * non_zero_mask
        mae = tf.reduce_sum(absolute_error) / n_nonzero
        
        return final_weight * mae
    
    return loss_fn


def sku_aware_composite_loss(alpha=0.9):
    """
    SKU-aware composite loss with volume-based MAE weighting.
    
    Formula: alpha * BCE + log1p(sku_mean_demand) * MAE
    
    This variant weights MAE by log1p of SKU mean demand, giving more
    importance to forecast accuracy on high-volume SKUs while avoiding
    extreme weights for very high-demand items.
    
    Args:
        alpha: Weight for BCE component (default: 0.9 for 90% zero rate)
    
    Returns:
        Loss function compatible with Keras model.compile()
        
    Note:
        This requires SKU mean demand to be computed externally and passed
        as a constant. For dynamic computation, use a custom training loop.
    """
    def loss_fn(y_true, y_pred):
        """
        Compute SKU-aware composite loss.
        
        Args:
            y_true: True demand values (shape: [batch_size,])
            y_pred: Predicted demand values (shape: [batch_size,])
        
        Returns:
            Scalar loss value
        """
        # Flatten inputs
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Binary target: 1 if demand > 0, else 0
        y_binary = tf.cast(y_true > 0, tf.float32)
        
        # Convert predictions to binary probabilities
        y_pred_binary = tf.nn.sigmoid(y_pred / 10.0)
        
        # Binary cross-entropy loss (zero detection)
        bce_loss = tf.keras.losses.binary_crossentropy(y_binary, y_pred_binary)
        
        # Mean absolute error loss (magnitude prediction)
        mae_loss = tf.abs(y_true - y_pred)
        
        # For this batch, use batch-level NON-ZERO mean as proxy for SKU mean
        # In practice, this should be replaced with actual SKU mean demand
        non_zero_mask = tf.cast(y_true > 0, tf.float32)
        n_nonzero = tf.reduce_sum(non_zero_mask) + 1e-7
        batch_mean_nonzero = tf.reduce_sum(y_true * non_zero_mask) / n_nonzero
        mae_weight = tf.math.log1p(batch_mean_nonzero)
        
        # Combined loss: alpha * BCE + log1p(mean) * MAE
        combined = alpha * bce_loss + mae_weight * mae_loss
        
        return tf.reduce_mean(combined)
    
    return loss_fn


def weighted_composite_loss(alpha=0.5):
    """
    SKU-weighted composite loss for prioritizing high-volume SKUs.
    
    Similar to composite_loss but supports sample weights to emphasize
    important SKUs (e.g., high-revenue or high-volume products).
    
    Args:
        alpha: Weight for BCE component (default: 0.5)
    
    Returns:
        Loss function compatible with Keras model.compile()
    
    Example:
        >>> # Calculate SKU weights
        >>> sku_mean_demand = df.groupby('sku')['demand'].mean()
        >>> sample_weights = np.log1p(sku_mean_demand[sku_ids].values)
        >>> 
        >>> model.compile(
        ...     optimizer='adam',
        ...     loss={'final_forecast': weighted_composite_loss(alpha=0.5)}
        ... )
        >>> model.fit(X, y, sample_weight=sample_weights)
    """
    def loss_fn(y_true, y_pred, sample_weight=None):
        """
        Compute weighted composite loss.
        
        Args:
            y_true: True demand values
            y_pred: Predicted demand values
            sample_weight: Optional weights per sample (e.g., log1p(mean_demand))
        
        Returns:
            Scalar loss value
        """
        # Binary target
        y_binary = tf.cast(y_true > 0, tf.float32)
        y_pred_binary = tf.nn.sigmoid(y_pred / 10.0)
        
        # Binary cross-entropy (no weighting for zero detection)
        bce_loss = tf.keras.losses.binary_crossentropy(y_binary, y_pred_binary)
        
        # MAE with optional weighting
        mae_per_sample = tf.abs(y_true - y_pred)
        
        if sample_weight is not None:
            weighted_mae = mae_per_sample * sample_weight
        else:
            weighted_mae = mae_per_sample
        
        # Combined loss
        combined = alpha * bce_loss + weighted_mae
        
        return tf.reduce_mean(combined)
    
    return loss_fn


def mae_loss():
    """
    Simple MAE loss wrapper for consistency.
    
    Returns:
        MAE loss function
    """
    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    return loss_fn
