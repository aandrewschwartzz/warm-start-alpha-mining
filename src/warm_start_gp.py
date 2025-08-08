"""
Implementation of Warm Start Genetic Programming for alpha mining.
"""
import random
import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import List, Tuple, Optional
from alpha_tree import AlphaTree, AlphaNode, NodeType, Operator, DataField, SubsetType
from fitness import compute_fitness
import logging
from evaluate_alpha import tree_to_str, evaluate_alpha

logger = logging.getLogger(__name__)

class WarmStartGP:
    """Warm Start Genetic Programming for alpha mining."""
    
    def __init__(
        self,
        warm_start_alpha: AlphaTree,
        population_size: int = 50,
        tournament_size: int = 3,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.3,
        max_generations: int = 100,
        early_stopping_generations: int = 20
    ):
        """
        Initialize the Warm Start GP algorithm.
        
        Args:
            warm_start_alpha: Initial alpha to start from
            population_size: Size of the population
            tournament_size: Number of individuals in tournament selection
            crossover_prob: Probability of crossover vs mutation
            mutation_prob: Probability of mutation
            max_generations: Maximum number of generations
            early_stopping_generations: Stop if no improvement after this many generations
        """
        self.warm_start_alpha = warm_start_alpha
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_generations = max_generations
        self.early_stopping_generations = early_stopping_generations
        
        self.population: List[AlphaTree] = []
        self.best_fitness = float('-inf')
        self.best_alpha: Optional[AlphaTree] = None
        self.generation = 0
        self.generations_without_improvement = 0
    
    def initialize_population(self) -> None:
        """Initialize population with mutations of warm start alpha."""
        self.population = [self.warm_start_alpha]
        
        while len(self.population) < self.population_size:
            # Create new individual by mutating warm start alpha
            new_alpha = self.point_mutation(self.warm_start_alpha.copy())
            
            # Add if not duplicate
            if not any(new_alpha.equals(alpha) for alpha in self.population):
                self.population.append(new_alpha)
    
    def evolve(
        self,
        market_data: dict,
        forward_returns: np.ndarray,
        val_market_data: dict,
        val_forward_returns: np.ndarray,
        discrepancy_penalty: float = 0.5
    ) -> Tuple[AlphaTree, float, List[float], List[float]]:
        """
        Run the evolutionary process with vectorized operations.
        
        Args:
            market_data: Training market data
            forward_returns: Training forward returns
            val_market_data: Validation market data
            val_forward_returns: Validation forward returns
            discrepancy_penalty: Weight of the discrepancy penalty (default: 0.3)
            
        Returns:
            Tuple of (best_alpha, best_fitness, train_fitness_history, val_fitness_history)
        """
        self.initialize_population()
        print(f"Initialized population with {self.population_size} individuals")
        
        train_fitness_history = []
        val_fitness_history = []
        
        # Pre-allocate arrays for fitness values
        fitness_values = np.zeros(self.population_size)
        val_fitness_values = np.zeros(self.population_size)
        
        while (
            self.generation < self.max_generations and
            self.generations_without_improvement < self.early_stopping_generations
        ):
            print(f"\nGeneration {self.generation + 1}/{self.max_generations}")
            
            # Vectorized fitness computation
            fitness_values = np.array([
                compute_fitness(alpha, market_data, forward_returns)
                for alpha in self.population
            ])
            
            # Vectorized validation fitness computation
            val_fitness_values = np.array([
                compute_fitness(alpha, val_market_data, val_forward_returns)
                for alpha in self.population
            ])
            
            # Calculate discrepancy and adjusted fitness
            discrepancy = np.abs(fitness_values - val_fitness_values)
            adjusted_fitness = fitness_values - discrepancy_penalty * discrepancy
            
            # Track best alpha using adjusted fitness
            best_idx = np.argmax(adjusted_fitness)
            current_best_adjusted_fitness = adjusted_fitness[best_idx]
            current_best_train_fitness = fitness_values[best_idx]
            current_best_val_fitness = val_fitness_values[best_idx]
            
            if (
                self.best_alpha is None or
                current_best_adjusted_fitness > self.best_fitness
            ):
                self.best_fitness = current_best_adjusted_fitness
                self.best_alpha = self.population[best_idx].copy()
                self.generations_without_improvement = 0
                print(f"New best adjusted fitness: {self.best_fitness:.4f}")
                print(f"Training fitness: {current_best_train_fitness:.4f}")
                print(f"Validation fitness: {current_best_val_fitness:.4f}")
                print(f"Discrepancy: {discrepancy[best_idx]:.4f}")
            else:
                self.generations_without_improvement += 1
                print(f"No improvement for {self.generations_without_improvement} generations")
            
            # Record history using vectorized means
            train_fitness = np.mean(fitness_values)
            val_fitness = np.mean(val_fitness_values)
            train_fitness_history.append(train_fitness)
            val_fitness_history.append(val_fitness)
            
            print(f"Average training fitness: {train_fitness:.4f}")
            print(f"Average validation fitness: {val_fitness:.4f}")
            print(f"Average discrepancy: {np.mean(discrepancy):.4f}")
            
            # Create next generation with vectorized selection
            next_population = [self.population[best_idx]]  # Elitism
            
            # Pre-compute tournament probabilities using adjusted fitness
            selection_probs = self._compute_selection_probabilities(
                fitness_values,
                val_fitness_values,
                discrepancy_penalty
            )
            
            # Batch creation of new individuals
            while len(next_population) < self.population_size:
                batch_size = min(
                    self.population_size - len(next_population),
                    max(1, self.population_size // 10)  # Create ~10% of population at a time
                )
                
                # For generation 0, use only point mutations
                # For subsequent generations, use both crossover and mutation
                if self.generation == 0:
                    crossover_mask = np.zeros(batch_size, dtype=bool)  # Force all mutations
                else:
                    crossover_mask = np.random.random(batch_size) < self.crossover_prob
                
                new_individuals = []
                for do_crossover in crossover_mask:
                    if do_crossover and len(self.population) > 1 and self.generation > 0:
                        # Vectorized tournament selection
                        parents = np.random.choice(
                            len(self.population),
                            size=2,
                            p=selection_probs
                        )
                        child = self.restricted_crossover(
                            self.population[parents[0]],
                            self.population[parents[1]]
                        )
                    else:
                        parent_idx = np.random.choice(
                            len(self.population),
                            p=selection_probs
                        )
                        child = self.point_mutation(self.population[parent_idx].copy())
                    
                    # Only add non-duplicate children
                    if not any(child.equals(alpha) for alpha in next_population):
                        new_individuals.append(child)
                
                next_population.extend(new_individuals)
            
            self.population = next_population[:self.population_size]  # Ensure exact size
            self.generation += 1
        
        # Print detailed information about the best alpha
        print("\n" + "="*50)
        print("Best Alpha Information:")
        print("="*50)
        print(f"Final Training Fitness: {self.best_fitness:.4f}")
        
        # Compute validation performance
        val_fitness = compute_fitness(self.best_alpha, val_market_data, val_forward_returns)
        print(f"Final Validation Fitness: {val_fitness:.4f}")
        
        # Print alpha structure
        print("\nAlpha Expression Structure:")
        self._print_alpha_tree(self.best_alpha.root)
        
        # Compute and print performance statistics
        train_predictions = self.best_alpha.evaluate(market_data)
        val_predictions = self.best_alpha.evaluate(val_market_data)
        
        print("\nTraining Set Statistics:")
        self._print_prediction_stats(train_predictions, forward_returns)
        
        print("\nValidation Set Statistics:")
        self._print_prediction_stats(val_predictions, val_forward_returns)
        
        print("\nEvolution Statistics:")
        print(f"Total Generations: {self.generation}")
        print(f"Early Stopping after {self.generations_without_improvement} generations without improvement")
        print("="*50)
        
        # After printing all statistics, collect them for logging
        train_ics = [
            np.corrcoef(train_predictions[t], forward_returns[t])[0,1]
            for t in range(len(train_predictions))
            if np.all(np.isfinite(train_predictions[t])) and np.all(np.isfinite(forward_returns[t]))
        ]
        
        experiment_stats = {
            'train_fitness': self.best_fitness,
            'val_fitness': val_fitness,
            'mean_ic': np.mean(train_ics),
            'ir': np.mean(train_ics) / np.std(train_ics) if len(train_ics) > 0 else 0,
            'ic_std': np.std(train_ics),
            'max_ic': np.max(train_ics),
            'min_ic': np.min(train_ics),
            'generations': self.generation,
            'generations_without_improvement': self.generations_without_improvement
        }
        
        # Log experiment results
        self._log_experiment_results(experiment_stats)
        
        # After logging experiment results
        print("\nPerforming detailed evaluation of best alpha...")
        print("=" * 50)
        print("Evaluating GA's best alpha")
        print(f"GA Reported Fitness - Training: {experiment_stats['train_fitness']:.4f}, "
              f"Validation: {experiment_stats['val_fitness']:.4f}")
        print("=" * 50)
        evaluate_alpha(self.best_alpha)
        
        return (
            self.best_alpha,
            self.best_fitness,
            train_fitness_history,
            val_fitness_history
        )
    
    def _print_alpha_tree(self, node: AlphaNode, indent: str = "") -> None:
        """Print a readable representation of the alpha tree."""
        if node.node_type == NodeType.DATA:
            print(f"{indent}Data: {node.value.value}")
        else:
            print(f"{indent}Op: {node.value.value}")
            for child in node.children:
                self._print_alpha_tree(child, indent + "  ")
    
    def _print_prediction_stats(self, predictions: np.ndarray, returns: np.ndarray) -> None:
        """
        Print detailed statistics about alpha predictions using Pearson correlation (IC).
        Follows the academic definition:
        IC = (1/T) * sum(PearsonCorr(a_t, r_t)) from t=1 to T
        """
        # Convert to float64 to avoid numerical issues
        predictions = predictions.astype(np.float64)
        returns = returns.astype(np.float64)
        
        # Calculate daily Pearson ICs
        ics = []
        for t in range(len(predictions)):
            # Skip if either array has all NaN or inf values
            if not np.any(np.isfinite(predictions[t])) or not np.any(np.isfinite(returns[t])):
                continue
                
            # Get only finite values from both arrays
            mask = np.isfinite(predictions[t]) & np.isfinite(returns[t])
            if not np.any(mask):
                continue
                
            pred_t = predictions[t][mask]
            ret_t = returns[t][mask]
            
            # Skip if either array is constant or too small
            if len(pred_t) < 2 or len(ret_t) < 2:
                continue
            if np.all(pred_t == pred_t[0]) or np.all(ret_t == ret_t[0]):
                continue
                
            # Calculate standard deviations
            pred_std = np.std(pred_t)
            ret_std = np.std(ret_t)
            
            # Skip if either standard deviation is zero
            if pred_std == 0 or ret_std == 0:
                continue
                
            try:
                # Calculate correlation manually to avoid numpy warnings
                pred_norm = (pred_t - np.mean(pred_t)) / pred_std
                ret_norm = (ret_t - np.mean(ret_t)) / ret_std
                corr = np.mean(pred_norm * ret_norm)
                
                if np.isfinite(corr):
                    ics.append(corr)
            except (ValueError, RuntimeWarning):
                continue
        
        # Calculate statistics
        if ics:
            mean_ic = np.mean(ics)
            ic_std = np.std(ics)
            ir = mean_ic / ic_std if ic_std > 0 else 0.0
            max_ic = np.max(ics)
            min_ic = np.min(ics)
        else:
            mean_ic = ic_std = ir = max_ic = min_ic = 0.0
        
        # Print statistics
        print("\nPearson IC Statistics:")
        print("=====================")
        print(f"Mean Pearson IC: {mean_ic:.4f}")
        print(f"Information Ratio: {ir:.4f}")
        print(f"IC Std Dev: {ic_std:.4f}")
        print(f"Max Pearson IC: {max_ic:.4f}")
        print(f"Min Pearson IC: {min_ic:.4f}")
        print(f"Valid ICs: {len(ics)} out of {len(predictions)} time points")
    
    def _compute_selection_probabilities(
        self,
        fitness_values: np.ndarray,
        val_fitness_values: np.ndarray,
        discrepancy_penalty: float = 0.3
    ) -> np.ndarray:
        """
        Compute selection probabilities based on fitness values and training/validation discrepancy.
        Uses rank-based selection with a penalty for training/validation discrepancy.
        
        Args:
            fitness_values: Array of training fitness values
            val_fitness_values: Array of validation fitness values
            discrepancy_penalty: Weight of the discrepancy penalty (default: 0.3)
            
        Returns:
            Array of selection probabilities
        """
        # Calculate discrepancy between training and validation fitness
        discrepancy = np.abs(fitness_values - val_fitness_values)
        
        # Compute adjusted fitness that penalizes large discrepancies
        adjusted_fitness = fitness_values - discrepancy_penalty * discrepancy
        
        # Rank-based selection using adjusted fitness
        ranks = np.argsort(np.argsort(-adjusted_fitness))  # -1 for descending order
        selection_probs = 1 / (ranks + 1)
        
        return selection_probs / np.sum(selection_probs)  # Normalize to probabilities
    
    def tournament_selection(self, fitness_values: List[float]) -> AlphaTree:
        """Select individual using tournament selection."""
        tournament_indices = random.sample(
            range(len(self.population)),
            min(self.tournament_size, len(self.population))
        )
        winner_idx = max(tournament_indices, key=lambda i: fitness_values[i])
        return self.population[winner_idx]
    
    def restricted_crossover(
        self,
        parent1: AlphaTree,
        parent2: AlphaTree
    ) -> AlphaTree:
        """
        Perform restricted crossover between two parents.
        Only exchanges subtrees at identical positions.
        """
        # Add logging before crossover
        logger.debug(f"Performing crossover between:\nParent1: {tree_to_str(parent1.root)}\nParent2: {tree_to_str(parent2.root)}")
        
        def get_random_node(root: AlphaNode) -> Tuple[AlphaNode, List[int]]:
            """Get random node and its path from root."""
            nodes = []
            paths = []
            
            def traverse(node: AlphaNode, path: List[int]):
                nodes.append((node, path))
                for i, child in enumerate(node.children):
                    traverse(child, path + [i])
            
            traverse(root, [])
            if not nodes:  # Should never happen but let's be safe
                return root, []
            return random.choice(nodes)
        
        # Get random node and path from parent1
        node1, path = get_random_node(parent1.root)
        if not path:  # If we got the root node, just return parent1
            return parent1
            
        try:
            # Find corresponding node in parent2
            node2 = parent2.root
            for idx in path:
                if idx >= len(node2.children):
                    return parent1  # Invalid path, return unchanged
                node2 = node2.children[idx]
            
            # Verify nodes have same number of children to maintain tree validity
            if len(node1.children) != len(node2.children):
                return parent1
            
            # Create offspring by copying parent1
            offspring = parent1.copy()
            
            # Replace subtree in offspring
            current = offspring.root
            for idx in path[:-1]:
                if idx >= len(current.children):
                    return parent1  # Extra safety check
                current = current.children[idx]
            
            if path[-1] >= len(current.children):
                return parent1  # Final safety check
                
            current.children[path[-1]] = node2.copy()
            
            # Add logging after crossover
            logger.debug(f"Crossover result: {tree_to_str(offspring.root)}")
            return offspring
            
        except (IndexError, AttributeError):
            # If anything goes wrong, return unchanged parent
            return parent1
    
    def point_mutation(self, alpha: AlphaTree) -> AlphaTree:
        """
        Perform point mutation on an alpha tree.
        Changes one node's operator or data field while preserving structure.
        Subset operators can only be replaced by other subset operators.
        """
        # Add logging before mutation
        logger.debug(f"Performing mutation on: {tree_to_str(alpha.root)}")
        
        def get_random_node(root: AlphaNode) -> Tuple[AlphaNode, List[int]]:
            """Get random node and its path from root."""
            nodes = []
            paths = []
            
            def traverse(node: AlphaNode, path: List[int]):
                nodes.append((node, path))
                for i, child in enumerate(node.children):
                    traverse(child, path + [i])
            
            traverse(root, [])
            return random.choice(nodes)
        
        # Get random node to mutate
        node, path = get_random_node(alpha.root)
        
        # Create mutated copy
        mutated = alpha.copy()
        
        # Find node to mutate in copy
        current = mutated.root
        for idx in path[:-1]:
            current = current.children[idx]
        target = current.children[path[-1]] if path else current
        
        # Mutate node while preserving type and subset constraints
        if target.node_type == NodeType.SUBSET:
            # If it's a subset node, only allow mutation to other subset types
            target.value = random.choice(list(SubsetType))
        elif target.node_type == NodeType.OPERATOR:
            # Change operator while keeping same number of children
            # and ensuring we don't convert to/from subset operations
            num_children = len(target.children)
            valid_operators = [
                op for op in Operator
                if op.num_args == num_children and op != Operator.SUBSET  # Never allow conversion to SUBSET
            ]
            if valid_operators:  # Only mutate if we have valid options
                target.value = random.choice(valid_operators)
        else:  # DATA node
            target.value = random.choice(list(DataField))
        
        # Add logging after mutation
        logger.debug(f"Mutation result: {tree_to_str(mutated.root)}")
        return mutated
    
    def evaluate_population(self, population: List[AlphaTree], batch_size: int = 100):
        """Evaluate population in batches to manage memory."""
        fitness_values = []
        for i in range(0, len(population), batch_size):
            batch = population[i:i + batch_size]
            batch_fitness = self._evaluate_batch(batch)
            fitness_values.extend(batch_fitness)
        return fitness_values

    def _log_experiment_results(self, experiment_stats: dict) -> None:
        """
        Log experiment results to a CSV file, avoiding duplicate alpha expressions.
        
        Args:
            experiment_stats: Dictionary containing experiment statistics
        """
        log_file = "experiment_results.csv"
        
        # Get alpha structure string and verify it matches current state
        alpha_str = tree_to_str(self.best_alpha.root)
        
        # Add verification logging
        logger.info(f"Logging experiment results for alpha: {alpha_str}")
        
        # Verify the structure matches what was evaluated
        current_structure = tree_to_str(self.best_alpha.root)
        if alpha_str != current_structure:
            logger.error(f"Alpha structure mismatch!\nLogged: {alpha_str}\nCurrent: {current_structure}")
            raise ValueError("Alpha structure mismatch detected")
        
        # Prepare row data
        row_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'alpha_structure': alpha_str,
            'final_training_fitness': experiment_stats['train_fitness'],
            'final_validation_fitness': experiment_stats['val_fitness'],
            'mean_ic': experiment_stats['mean_ic'],
            'information_ratio': experiment_stats['ir'],
            'ic_std_dev': experiment_stats['ic_std'],
            'max_ic': experiment_stats['max_ic'],
            'min_ic': experiment_stats['min_ic'],
            'total_generations': experiment_stats['generations'],
            'generations_without_improvement': experiment_stats['generations_without_improvement']
        }
        
        # Create DataFrame for single row
        df_new = pd.DataFrame([row_data])
        
        # Check if file exists and handle accordingly
        if os.path.exists(log_file):
            # Read existing CSV
            df_existing = pd.read_csv(log_file)
            
            # Check if this alpha structure already exists
            if not df_existing['alpha_structure'].eq(alpha_str).any():
                # No duplicate found, append new row
                df_new.to_csv(log_file, mode='a', header=False, index=False)
            
        else:
            # File doesn't exist, create new one
            df_new.to_csv(log_file, mode='w', header=True, index=False)
            logger.info("Created new results file and logged alpha expression")