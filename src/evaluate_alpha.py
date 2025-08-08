"""
Script to evaluate and analyze alpha tree performance.
Configure the parameters below and run with: python -m src.evaluate_alpha
"""
from datetime import datetime, timedelta
import logging
import os
from typing import List
import pandas as pd
import numpy as np

from alpha_tree import AlphaTree, AlphaNode, NodeType, DataField
from data_loader import DataLoader
from evaluation.metrics import calculate_ic_metrics, calculate_strategy_metrics
from evaluation.strategy import alpha_to_strategy
from evaluation.reporting import save_weights, save_metrics, print_evaluation_summary
from evaluation.weights_conversion import convert_weights_to_dict, save_dict_to_file

# Configure logging
logger = logging.getLogger(__name__)

# ============ EVALUATION CONFIGURATION ============
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'NVDA', 'TSLA', 'JPM', 'V', 'WMT'
]

END_DATE = datetime.now().strftime('%Y-%m-%d')
START_DATE = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')

TRAIN_RATIO = 0.6
VAL_RATIO = 0.4

def tree_to_str(node: AlphaNode) -> str:
    """Convert alpha tree to string representation."""
    if node.node_type == NodeType.DATA:
        return str(node.value.value)
    else:
        children_str = [tree_to_str(child) for child in node.children]
        return f"{node.value.value}({','.join(children_str)})"

def format_alpha_tree(node: AlphaNode, indent: str = "") -> List[str]:
    """Format alpha tree as readable structure."""
    lines = []
    if node.node_type == NodeType.DATA:
        lines.append(f"{indent}└── Data: {node.value.value}")
    else:
        lines.append(f"{indent}└── {'Op' if node.node_type == NodeType.OPERATOR else 'Subset'}: {node.value.value}")
        for i, child in enumerate(node.children):
            new_indent = indent + ("    ├── " if i < len(node.children) - 1 else "    └── ")
            lines.extend(format_alpha_tree(child, new_indent))
    return lines

def evaluate_alpha(alpha_tree: AlphaTree) -> None:
    """Evaluate alpha tree performance and generate detailed metrics."""
    # Convert tree to string for logging and directory naming
    alpha_str = tree_to_str(alpha_tree.root)
    logger.info(f"Starting evaluation of alpha: {alpha_str}")
    results_dir = os.path.join('evaluation_results', alpha_str)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and split data
    logger.info(f"Loading market data from {START_DATE} to {END_DATE}...")
    loader = DataLoader(SYMBOLS, START_DATE, END_DATE)
    (train_data, train_returns), (val_data, val_returns) = loader.split_data(
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO
    )
    
    # Generate alpha values
    train_alphas = alpha_tree.evaluate(train_data)
    val_alphas = alpha_tree.evaluate(val_data)
    
    # Calculate alpha metrics
    train_metrics = calculate_ic_metrics(train_alphas, train_returns)
    val_metrics = calculate_ic_metrics(val_alphas, val_returns)
    
    # Generate and evaluate strategy
    train_positions = alpha_to_strategy(train_alphas)
    val_positions = alpha_to_strategy(val_alphas)
    
    train_strategy_metrics = calculate_strategy_metrics(train_positions, train_returns)
    val_strategy_metrics = calculate_strategy_metrics(val_positions, val_returns)
    
    # Get actual dates from the data
    train_dates, val_dates = loader.get_split_dates(train_ratio=TRAIN_RATIO)
    
    # Save results
    train_weights_path = os.path.join(results_dir, 'train_weights.csv')
    val_weights_path = os.path.join(results_dir, 'val_weights.csv')
    
    save_weights(
        positions=train_positions,
        symbols=SYMBOLS,
        dates=train_dates,
        output_path=train_weights_path
    )
    
    save_weights(
        positions=val_positions,
        symbols=SYMBOLS,
        dates=val_dates,
        output_path=val_weights_path
    )
    
    # Convert weights to dictionary format
    train_dict = convert_weights_to_dict(train_weights_path)
    val_dict = convert_weights_to_dict(val_weights_path)
    
    # Save dictionary format
    save_dict_to_file(train_dict, os.path.join(results_dir, 'QC_train_weights_dict.txt'))
    save_dict_to_file(val_dict, os.path.join(results_dir, 'QC_val_weights_dict.txt'))
    
    save_metrics(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        train_strategy_metrics=train_strategy_metrics,
        val_strategy_metrics=val_strategy_metrics,
        alpha_str=alpha_str,
        tree_structure=format_alpha_tree(alpha_tree.root),
        output_dir=results_dir
    )
    
    # Print summary
    print_evaluation_summary(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        train_strategy_metrics=train_strategy_metrics,
        val_strategy_metrics=val_strategy_metrics,
        start_date=START_DATE,
        end_date=END_DATE,
        symbols=SYMBOLS
    )
