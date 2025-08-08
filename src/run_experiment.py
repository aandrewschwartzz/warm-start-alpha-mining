"""
Example script to run Warm Start GP experiment.
"""
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import List
from dataclasses import dataclass
from data_loader import DataLoader
from alpha_tree import AlphaTree, AlphaNode, NodeType, Operator, DataField, SubsetType
from warm_start_gp import WarmStartGP
import multiprocessing as mp

@dataclass
class ExperimentConfig:
    """Configuration for GP experiment."""
    symbols: List[str]
    start_date: str
    end_date: str
    population_size: int = 12
    max_generations: int = 100
    n_processes: int = 8

def create_warm_start_alpha() -> AlphaTree:
    """
    Create Alpha1 from the paper:
    "Rank the daily intraday returns of the past 20 days by the same day's turnover rate, 
    and average the top 5."
    """
    # Create the tree structure
    alpha1 = AlphaTree(
        AlphaNode(
            NodeType.OPERATOR,
            Operator.MA,
            [
                # First select top 5 ranked values
                AlphaNode(
                    NodeType.SUBSET,
                    SubsetType.TOP_N,
                    [
                        # Rank intraday returns BY turnover rate
                        AlphaNode(
                            NodeType.OPERATOR,
                            Operator.ASC_RANK,
                            [
                                AlphaNode(NodeType.DATA, DataField.INTRADAY_RETURNS),  # Values to rank
                                AlphaNode(NodeType.DATA, DataField.TURNOVER_RATE)      # Rank by this
                            ]
                        )
                    ]
                )
            ]
        )
    )
    return alpha1


def run_experiment(config: ExperimentConfig, run_id: int = 0) -> None:
    """
    Run the Warm Start GP experiment.
    
    Args:
        config: Experiment configuration
        run_id: Identifier for this particular run
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - Process %(process)d - %(message)s'
    )
    
    # Load data
    logging.info(f"Run {run_id}: Loading market data...")
    loader = DataLoader(config.symbols, config.start_date, config.end_date)
    
    # Validate returns calculation before proceeding
    logging.info("Validating returns calculations...")
    loader.validate_returns()
    
    # Split data
    (train_data, train_returns), (val_data, val_returns) = loader.split_data()
    
    # Create warm start alpha
    warm_start_alpha = create_warm_start_alpha()
    
    # Initialize and run GP
    logging.info(f"Run {run_id}: Running Warm Start GP...")
    gp = WarmStartGP(
        warm_start_alpha=warm_start_alpha,
        population_size=config.population_size,
        max_generations=config.max_generations
    )
    
    best_alpha, best_fitness, train_history, val_history = gp.evolve(
        train_data,
        train_returns,
        val_data,
        val_returns
    )
    
    # Log results
    logging.info(f"Run {run_id}: Best fitness (train): {best_fitness:.4f}")
    logging.info(f"Run {run_id}: Best validation fitness: {val_history[-1]:.4f}")
    
    # Plot fitness history
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_history, label='Train')
    # plt.plot(val_history, label='Validation')
    # plt.xlabel('Generation')
    # plt.ylabel('Mean Population Fitness')
    # plt.title(f'Fitness History - Run {run_id}')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'fitness_history_run_{run_id}.png')
    # plt.close()


def run_parallel_experiments(config: ExperimentConfig) -> None:
    """
    Run multiple Warm Start GP experiments in parallel.
    
    Args:
        config: Experiment configuration
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - Process %(process)d - %(message)s'
    )
    logging.info(f"Starting {config.n_processes} parallel experiments...")
    
    # Run experiments in parallel
    with mp.Pool(processes=config.n_processes) as pool:
        # Create list of (config, run_id) tuples
        experiment_args = [(config, run_id) for run_id in range(config.n_processes)]
        pool.starmap(run_experiment, experiment_args)
    
    logging.info("All parallel experiments completed.")


if __name__ == '__main__':
    # Example usage
    config = ExperimentConfig(
        symbols=[
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'NVDA', 'TSLA', 'JPM', 'V', 'WMT'
        ],
        end_date=datetime.now().strftime('%Y-%m-%d'),
        start_date=(datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    )
    
    run_parallel_experiments(config) 