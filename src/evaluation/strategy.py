"""
Strategy generation and position sizing functionality.
"""
import numpy as np
from typing import Dict

def alpha_to_strategy(
    alpha_values: np.ndarray,
    target_volatility: float = 0.10,
    max_leverage: float = 1.0
) -> np.ndarray:
    """
    Convert alpha signals into market-neutral strategy weights.
    
    Args:
        alpha_values: Raw alpha signals (time x symbols)
        target_volatility: Desired annual portfolio volatility
        max_leverage: Maximum gross leverage allowed
        
    Returns:
        Position weights array (time x symbols)
    """
    weights = np.zeros_like(alpha_values)
    
    # Convert to market neutral weights
    for t in range(len(alpha_values)):
        if np.any(np.isfinite(alpha_values[t])):
            # Make market neutral
            signal_t = alpha_values[t] - np.mean(alpha_values[t])
            
            # Standardize bet sizes
            signal_std = np.std(signal_t[np.isfinite(signal_t)])
            if signal_std > 0:
                weights[t] = signal_t / signal_std
    
    # Scale to target volatility
    daily_vol_target = target_volatility / np.sqrt(252)
    weights = weights * daily_vol_target
    
    # Apply leverage constraint
    for t in range(len(weights)):
        if np.sum(np.abs(weights[t])) > 0:
            weights[t] = weights[t] * max_leverage / np.sum(np.abs(weights[t]))
    
    return weights 