"""
Custom loss functions for DeepSequence Hierarchical Attention.

Includes composite loss optimized for intermittent demand forecasting.
"""

import tensorflow as tf


def composite_loss(alpha=0.5):
    """
    Composite loss with adaptive weighting by zero/non-zero counts.
    
    Optimized for intermittent demand with high zero rates (e.g., 90%).
    - BCE component: Weighted by zero count (for zero detection)
    - MAE component: Weighted by non-zero count (for magnitude prediction)
    
    The weights are dynamically calculated from each batch:
    - bce_weight = proportion of zeros in batch
    - mae_weight = proportion of non-zeros in batch
    
    Args:
        alpha: Base weight multiplier (default: 0.5, not used in adaptive version)
               Kept for backward compatibility
    
    Returns:
        Loss function compatible with Keras model.compile()
    
    Example:
        >>> model.compile(
        ...     optimizer='adam',
        ...     loss={'final_forecast': composite_loss()},
        ...     metrics=['mae']
        ... )
    """
    def loss_fn(y_true, y_pred):
        """
        Compute composite loss with adaptive zero/non-zero weighting.
        
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
        
        # Calculate zero and non-zero fractions in batch
        total_count = tf.cast(tf.shape(y_true)[0], tf.float32)
        zero_count = tf.reduce_sum(1.0 - y_binary)
        nonzero_count = tf.reduce_sum(y_binary)
        
        # Convert to fractions (proportions)
        zero_fraction = zero_count / (total_count + 1e-7)
        nonzero_fraction = nonzero_count / (total_count + 1e-7)
        
        # Adaptive weights based on batch composition (fractions sum to 1.0)
        # More zeros -> higher BCE weight
        # More non-zeros -> higher MAE weight
        bce_weight = zero_fraction
        mae_weight = nonzero_fraction
        
        # Convert predictions to binary probabilities
        # Scale down predictions before sigmoid for numerical stability
        y_pred_binary = tf.nn.sigmoid(y_pred / 10.0)
        
        # Binary cross-entropy loss (zero detection)
        bce_loss = tf.keras.losses.binary_crossentropy(y_binary, y_pred_binary)
        
        # Mean absolute error loss (magnitude prediction)
        mae_loss = tf.abs(y_true - y_pred)
        
        # Combined loss with adaptive weights
        combined = bce_weight * bce_loss + mae_weight * mae_loss
        
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
