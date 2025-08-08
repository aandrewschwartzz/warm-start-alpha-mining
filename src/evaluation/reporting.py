"""
Reporting and results saving functionality.
"""
import os
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np

def save_weights(positions: np.ndarray, 
                symbols: List[str], 
                dates: List[str], 
                output_path: str) -> None:
    """
    Save portfolio weights to CSV file.
    """
    df = pd.DataFrame(positions, columns=symbols, index=dates)
    df.index.name = 'date'
    df.to_csv(output_path)

def save_metrics(train_metrics: Dict[str, float],
                val_metrics: Dict[str, float],
                train_strategy_metrics: Dict[str, float],
                val_strategy_metrics: Dict[str, float],
                alpha_str: str,
                tree_structure: List[str],
                output_dir: str) -> None:
    """
    Save evaluation metrics to CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/metrics.csv", "w") as f:
        # Write header information
        f.write(f"Alpha Expression,,,{alpha_str}\n")
        f.write("Tree Structure,,,\n")
        for line in tree_structure:
            f.write(f",,,{line}\n")
        f.write("\n")
        
        # Write alpha metrics
        f.write("Alpha Metrics\n")
        f.write("Metric,Training,Validation\n")
        for metric in ['mean_ic', 'information_ratio', 'ic_std', 'percent_positive_ic']:
            train_val = train_metrics.get(metric, '')
            val_val = val_metrics.get(metric, '')
            f.write(f"{metric},{train_val},{val_val}\n")
        
        # Write strategy metrics
        f.write("\nStrategy Metrics\n")
        f.write("Metric,Training,Validation\n")
        for metric in ['mean_return', 'volatility', 'sharpe_ratio', 
                      'max_drawdown', 'hit_ratio', 'turnover']:
            train_val = train_strategy_metrics.get(metric, '')
            val_val = val_strategy_metrics.get(metric, '')
            f.write(f"{metric},{train_val},{val_val}\n")

def print_evaluation_summary(train_metrics: Dict[str, float],
                           val_metrics: Dict[str, float],
                           train_strategy_metrics: Dict[str, float],
                           val_strategy_metrics: Dict[str, float],
                           start_date: str,
                           end_date: str,
                           symbols: List[str]) -> None:
    """
    Print a summary of evaluation results.
    """
    print("\nEvaluation Results Summary:")
    print("=" * 50)
    print(f"Training Period: {start_date} to {end_date}")
    print(f"Number of Symbols: {len(symbols)}")
    
    print("\nAlpha Metrics:")
    print("-" * 30)
    print("Training Period:")
    print(f"Mean IC: {train_metrics['mean_ic']:.4f}")
    print(f"Information Ratio: {train_metrics['information_ratio']:.4f}")
    print(f"IC Std Dev: {train_metrics['ic_std']:.4f}")
    print(f"% Positive IC: {train_metrics['percent_positive_ic']:.1f}%")
    
    print("\nValidation Period:")
    print(f"Mean IC: {val_metrics['mean_ic']:.4f}")
    print(f"Information Ratio: {val_metrics['information_ratio']:.4f}")
    print(f"IC Std Dev: {val_metrics['ic_std']:.4f}")
    print(f"% Positive IC: {val_metrics['percent_positive_ic']:.1f}%")
    
    print("\nStrategy Metrics:")
    print("-" * 30)
    print("Training Period:")
    print(f"Annualized Return: {train_strategy_metrics['mean_return']:.2%}")
    print(f"Annualized Volatility: {train_strategy_metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {train_strategy_metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {train_strategy_metrics['max_drawdown']:.2%}")
    print(f"Hit Ratio: {train_strategy_metrics['hit_ratio']:.1f}%")
    
    print("\nValidation Period:")
    print(f"Annualized Return: {val_strategy_metrics['mean_return']:.2%}")
    print(f"Annualized Volatility: {val_strategy_metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {val_strategy_metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {val_strategy_metrics['max_drawdown']:.2%}")
    print(f"Hit Ratio: {val_strategy_metrics['hit_ratio']:.1f}%") 