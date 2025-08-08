"""
Fitness evaluation functions for alpha mining.
"""
import numpy as np
from scipy.stats import spearmanr
from typing import Callable
from alpha_tree import AlphaTree


def compute_fitness(
    alpha_tree: AlphaTree,
    market_data: dict,
    forward_returns: np.ndarray,
    method: str = 'rank_ic'
) -> float:
    """
    Compute the fitness of an alpha expression.
    
    Args:
        alpha_tree: The alpha expression to evaluate
        market_data: Dictionary of market data arrays
        forward_returns: Array of forward returns
        method: Fitness method ('rank_ic' or 'ic')
        
    Returns:
        Fitness score (higher is better)
    """
    # Evaluate alpha on data
    alpha_values = alpha_tree.evaluate(market_data)
    alpha_values = np.nan_to_num(alpha_values, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate IC
    if method == 'rank_ic':
        return compute_rank_ic(alpha_values, forward_returns)
    elif method == 'ic':
        return compute_ic(alpha_values, forward_returns)
    else:
        raise ValueError(f"Unknown fitness method: {method}")


def compute_rank_ic(
    alpha_values: np.ndarray,
    forward_returns: np.ndarray
) -> float:
    """
    Compute the Spearman rank correlation (Rank IC) between alpha values
    and forward returns.
    """
    # Compute rank correlation for each time period
    rank_ics = []
    for t in range(min(len(alpha_values), len(forward_returns))):
        if np.all(np.isfinite(alpha_values[t])) and np.all(np.isfinite(forward_returns[t])):
            # Check for constant arrays
            if np.all(alpha_values[t] == alpha_values[t][0]) or np.all(forward_returns[t] == forward_returns[t][0]):
                # If either array is constant, correlation is undefined
                # We'll assign a correlation of 0 as it provides no predictive value
                rank_ics.append(0.0)
                continue
                
            try:
                rho, _ = spearmanr(alpha_values[t], forward_returns[t])
                if np.isfinite(rho):
                    rank_ics.append(rho)
            except (ValueError, RuntimeWarning):
                # Handle any other numerical issues by assigning 0 correlation
                rank_ics.append(0.0)
    
    # Return mean Rank IC
    return np.mean(rank_ics) if rank_ics else 0.0


def compute_ic(
    alpha_values: np.ndarray,
    forward_returns: np.ndarray
) -> float:
    """
    Compute the Pearson correlation (IC) between alpha values and forward returns.
    
    Follows the academic definition:
    IC = (1/T) * sum(PearsonCorr(a_t, r_t)) from t=1 to T
    where a_t represents alpha values and r_t represents forward returns at time t.
    
    Args:
        alpha_values: Array of alpha values (time x symbols)
        forward_returns: Array of forward returns (time x symbols)
        
    Returns:
        Mean Information Coefficient (Pearson correlation)
    """
    # Compute correlation for each time period
    ics = []
    for t in range(min(len(alpha_values), len(forward_returns))):
        if np.all(np.isfinite(alpha_values[t])) and np.all(np.isfinite(forward_returns[t])):
            # Check for constant arrays
            if np.all(alpha_values[t] == alpha_values[t][0]) or np.all(forward_returns[t] == forward_returns[t][0]):
                # If either array is constant, correlation is undefined
                # We'll assign a correlation of 0 as it provides no predictive value
                ics.append(0.0)
                continue
                
            try:
                # Use np.corrcoef for Pearson correlation
                corr = np.corrcoef(alpha_values[t], forward_returns[t])[0, 1]
                if np.isfinite(corr):
                    ics.append(corr)
            except (ValueError, RuntimeWarning):
                # Handle any numerical issues by assigning 0 correlation
                ics.append(0.0)
    
    # Return mean IC
    return np.mean(ics) if ics else 0.0 