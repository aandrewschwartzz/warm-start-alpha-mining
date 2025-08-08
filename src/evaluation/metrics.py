"""
Core metrics calculation functionality for alpha evaluation.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import spearmanr

@dataclass
class AlphaMetrics:
    """Detailed metrics for alpha evaluation."""
    strategy_name: str
    alpha_structure: str
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    timestamp: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def calculate_ic_metrics(alpha_values: np.ndarray, forward_returns: np.ndarray, 
                        forward_window: int = 5) -> Dict[str, float]:
    """
    Calculate Information Coefficient (IC) based metrics.
    
    Args:
        alpha_values: Alpha signals (time x symbols)
        forward_returns: Forward returns (time x symbols)
        forward_window: Number of days for forward returns calculation
        
    Returns:
        Dictionary of IC metrics
    """
    # Calculate forward returns
    forward_rets = np.zeros_like(forward_returns)
    for t in range(len(forward_returns) - forward_window):
        forward_rets[t] = np.prod(1 + forward_returns[t:t+forward_window], axis=0) - 1
    forward_rets[-forward_window:] = np.nan
    
    # Calculate IC for each time period
    ics = []
    for t in range(len(alpha_values) - forward_window):
        if (np.all(np.isfinite(alpha_values[t])) and 
            np.all(np.isfinite(forward_rets[t])) and
            not np.all(alpha_values[t] == alpha_values[t][0]) and 
            not np.all(forward_rets[t] == forward_rets[t][0])):
            
            rho, _ = spearmanr(alpha_values[t], forward_rets[t])
            if np.isfinite(rho):
                ics.append(rho)
    
    ics = np.array(ics)
    
    if len(ics) == 0:
        return {
            'mean_ic': 0.0,
            'information_ratio': 0.0,
            'ic_std': 0.0,
            'percent_positive_ic': 0.0
        }
    
    return {
        'mean_ic': np.mean(ics),
        'information_ratio': np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0,
        'ic_std': np.std(ics),
        'percent_positive_ic': np.mean(ics > 0) * 100
    }

def calculate_strategy_metrics(positions: np.ndarray, returns: np.ndarray, 
                             transaction_cost: float = 0.0010,
                             market_impact: float = 0.0005) -> Dict[str, float]:
    """
    Calculate performance metrics for a trading strategy.
    
    Args:
        positions: Position weights (time x symbols)
        returns: Forward returns (time x symbols)
        transaction_cost: One-way transaction cost
        market_impact: Estimated slippage/market impact
    """
    # Calculate costs
    position_changes = np.abs(positions[1:] - positions[:-1])
    total_cost = (position_changes * (transaction_cost + market_impact)).sum(axis=1)
    
    # Calculate strategy returns
    strategy_returns = np.nansum(positions[1:] * returns[1:], axis=1) - total_cost
    valid_returns = strategy_returns[np.isfinite(strategy_returns)]
    
    if len(valid_returns) == 0:
        return {
            'mean_return': np.nan,
            'volatility': np.nan,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan,
            'hit_ratio': np.nan,
            'turnover': np.nan
        }
    
    # Calculate metrics
    mean_ret = np.mean(valid_returns) * 252
    vol = np.std(valid_returns) * np.sqrt(252)
    
    # Calculate drawdown
    cum_returns = np.cumprod(1 + valid_returns)
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / rolling_max - 1
    
    return {
        'mean_return': mean_ret,
        'volatility': vol,
        'sharpe_ratio': mean_ret / vol if vol > 0 else 0,
        'max_drawdown': np.min(drawdowns),
        'hit_ratio': np.mean(valid_returns > 0) * 100,
        'turnover': np.mean(position_changes.sum(axis=1)) * 252
    } 