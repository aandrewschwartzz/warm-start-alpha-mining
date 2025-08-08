"""
Alpha tree representation for genetic programming.
"""
from enum import Enum
from typing import Optional, List, Union, Tuple
import numpy as np


class NodeType(Enum):
    """Types of nodes in the alpha tree."""
    OPERATOR = "operator"
    DATA = "data"
    SUBSET = "subset" 


class SubsetType(Enum):
    """Types of subset operations available."""
    TOP_N = "top_n"  # Take top N values
    BOTTOM_N = "bottom_n"  # Take bottom N values
    ABOVE_MEAN = "above_mean"  # Take values above mean
    BELOW_MEAN = "below_mean"  # Take values below mean
    
    def __init__(self, value):
        self._value_ = value
        

class Operator(Enum):
    """Available operators for alpha expressions."""
    ADD = ("+", 2)
    SUB = ("-", 2)
    MUL = ("*", 2)
    DIV = ("/", 2)
    ASC_RANK = ("asc_rank", 2)  # Changed to 2 args: (values_to_rank, rank_by)
    MA = ("ma", 1)  # Moving average
    STD = ("std", 1)  # Standard deviation
    DELAY = ("delay", 1)  # Lagged value
    ZSCORE = ("zscore", 1)  # 20-day rolling Z-score
    SUBSET = ("subset", 2)  # New operator for subset operations

    def __init__(self, value, num_args):
        self._value_ = value
        self.num_args = num_args


class DataField(Enum):
    """Available market data fields."""
    CLOSE = "close"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    VOLUME = "volume"
    RETURNS = "returns"
    VWAP = "vwap"  # Volume-Weighted Average Price
    INTRADAY_RETURNS = "intraday_returns"  # (close - open) / open
    TURNOVER_RATE = "turnover_rate"  # daily volume / shares outstanding
    PE_RATIO = "pe_ratio"  # Price to Earnings ratio


class AlphaNode:
    """Node in the alpha expression tree."""
    
    def __init__(
        self,
        node_type: NodeType,
        value: Union[Operator, DataField, SubsetType],
        children: Optional[List['AlphaNode']] = None
    ):
        self.node_type = node_type
        self.value = value
        self.children = children or []
    
    def equals(self, other: 'AlphaNode') -> bool:
        """Check if two nodes are identical."""
        if not isinstance(other, AlphaNode):
            return False
        if self.node_type != other.node_type or self.value != other.value:
            return False
        if len(self.children) != len(other.children):
            return False
        return all(c1.equals(c2) for c1, c2 in zip(self.children, other.children))
    
    def copy(self) -> 'AlphaNode':
        """Create a deep copy of the node."""
        return AlphaNode(
            self.node_type,
            self.value,
            [child.copy() for child in self.children]
        )


class AlphaTree:
    """Complete alpha expression tree."""
    
    def __init__(self, root: AlphaNode):
        self.root = root
    
    def equals(self, other: 'AlphaTree') -> bool:
        """Check if two trees are identical."""
        return self.root.equals(other.root)
    
    def copy(self) -> 'AlphaTree':
        """Create a deep copy of the tree."""
        return AlphaTree(self.root.copy())
    
    def evaluate(self, data: dict) -> np.ndarray:
        """
        Evaluate the alpha expression on the given market data.
        
        Args:
            data: Dictionary mapping DataField values to numpy arrays
            
        Returns:
            numpy array of alpha values
        """
        return self._evaluate_node(self.root, data)
    
    def _evaluate_node(self, node: AlphaNode, data: dict) -> np.ndarray:
        """Recursively evaluate a node in the tree."""
        if node.node_type == NodeType.DATA:
            # Ensure data is float64
            return data[node.value.value].astype(np.float64)
            
        # Evaluate children first
        child_results = [self._evaluate_node(child, data) for child in node.children]
        
        # Handle subset operations
        if node.node_type == NodeType.SUBSET:
            return self._apply_subset(node.value, child_results[0])
        
        # Apply operator
        if node.value == Operator.ADD:
            return child_results[0] + child_results[1]
        elif node.value == Operator.SUB:
            return child_results[0] - child_results[1]
        elif node.value == Operator.MUL:
            # Cast both operands to float64 and handle edge cases consistently
            a = child_results[0].astype(np.float64)
            b = child_results[1].astype(np.float64)
            return np.multiply(a, b, dtype=np.float64)
        elif node.value == Operator.DIV:
            # Safe division (avoid divide by zero)
            denominator = child_results[1].astype(np.float64)
            numerator = child_results[0].astype(np.float64)
            return np.divide(numerator, denominator,
                           out=np.zeros_like(numerator),
                           where=denominator!=0)
        elif node.value == Operator.ASC_RANK:
            # Now takes two arguments: values_to_rank and rank_by
            values_to_rank = child_results[0]
            rank_by = child_results[1]
            if isinstance(values_to_rank, tuple):  # If input is a subset
                data, mask = values_to_rank
                ranked = self._rank_transform(data, rank_by, ascending=True)
                result = np.zeros_like(ranked)
                result[mask] = ranked[mask]
                return result
            return self._rank_transform(values_to_rank, rank_by, ascending=True)
        elif node.value == Operator.MA:
            if isinstance(child_results[0], tuple):  # If input is a subset
                data, mask = child_results[0]
                ma = self._moving_average(data, window=5)  # Default 5-day MA
                result = np.zeros_like(ma)
                result[mask] = ma[mask]
                return result
            return self._moving_average(child_results[0].astype(np.float64), window=5)
        elif node.value == Operator.STD:
            if isinstance(child_results[0], tuple):  # If input is a subset
                data, mask = child_results[0]
                std = self._rolling_std(data, window=20)  # Default 20-day std
                result = np.zeros_like(std)
                result[mask] = std[mask]
                return result
            return self._rolling_std(child_results[0].astype(np.float64), window=20)
        elif node.value == Operator.DELAY:
            if isinstance(child_results[0], tuple):  # If input is a subset
                data, mask = child_results[0]
                delayed = np.roll(data, 1)
                result = np.zeros_like(delayed)
                result[mask] = delayed[mask]
                return result
            return np.roll(child_results[0], 1)
        elif node.value == Operator.ZSCORE:
            if isinstance(child_results[0], tuple):  # If input is a subset
                data, mask = child_results[0]
                zscore = self._rolling_zscore(data, window=20)
                result = np.zeros_like(zscore)
                result[mask] = zscore[mask]
                return result
            return self._rolling_zscore(child_results[0].astype(np.float64), window=20)
            
        raise ValueError(f"Unknown operator: {node.value}")

    def _apply_subset(self, subset_type: SubsetType, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply subset operation to data.
        Returns:
            Tuple of (data, mask) where mask indicates which values are in the subset
        """
        mask = np.zeros_like(data, dtype=bool)
        
        # Process each time point
        for t in range(data.shape[0]):
            current_values = data[t]  # Get all symbols at current time t
            
            # Skip if all values are constant
            if np.all(current_values == current_values[0]):
                mask[t] = True
                continue
            
            if subset_type == SubsetType.TOP_N:
                # Select top 5 symbols
                n = min(5, len(current_values))
                threshold = np.percentile(current_values, 100 - (100 * n/len(current_values)))
                mask[t] = current_values >= threshold
                
            elif subset_type == SubsetType.BOTTOM_N:
                # Select bottom 5 symbols
                n = min(5, len(current_values))
                threshold = np.percentile(current_values, (100 * n/len(current_values)))
                mask[t] = current_values <= threshold
            
            elif subset_type == SubsetType.ABOVE_MEAN:
                mean = np.mean(current_values)
                mask[t] = current_values > mean
            
            elif subset_type == SubsetType.BELOW_MEAN:
                mean = np.mean(current_values)
                mask[t] = current_values < mean
            
            # Ensure at least one value is selected
            if not np.any(mask[t]):
                mask[t] = True

        return data, mask

    @staticmethod
    def _rank_transform(x: np.ndarray, rank_by: np.ndarray, ascending: bool = True) -> np.ndarray:
        """
        Cross-sectional rank transformation.
        Ranks values in x based on corresponding values in rank_by.
        Only ranks using the last 20 values for each time point.
        
        Args:
            x: Input array to be ranked (time x symbols)
            rank_by: Array to rank by (time x symbols)
            ascending: If True, rank from low to high. If False, rank from high to low.
        """
        x = x.astype(np.float64)
        rank_by = rank_by.astype(np.float64)
        result = np.zeros_like(x)
        
        # For each time point
        for t in range(x.shape[0]):
            # Get start index for last 20 values (or 0 if less than 20 values available)
            start_idx = max(0, t - 19)
            
            # Get the subset of values to use for ranking
            rank_by_subset = rank_by[start_idx:t+1]
            rank_by_mean = np.mean(rank_by_subset, axis=0)  # Average over the last 20 periods
            
            if ascending:
                # Get rank based on rank_by values
                ranks = np.argsort(np.argsort(rank_by_mean))
            else:
                ranks = np.argsort(np.argsort(-rank_by_mean))
            
            # Normalize ranks to [0, 1]
            result[t] = ranks.astype(np.float64) / x.shape[1]
        
        return result
    
    @staticmethod
    def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
        """
        Simple moving average with proper padding.
        Operates on time axis (axis 0) for each symbol.
        """
        x = x.astype(np.float64)
        result = np.empty_like(x, dtype=np.float64)
        
        # Adjust window size if larger than data
        window = min(window, x.shape[0])
        if window < 1:
            window = 1
        
        # Process each symbol (column)
        for j in range(x.shape[1]):
            series = x[:, j]  # Get time series for this symbol
            
            # Calculate valid MA values
            cumsum = np.cumsum(np.insert(series, 0, 0))
            ma = (cumsum[window:] - cumsum[:-window]) / window
            
            # Pad the beginning with the first MA value
            result[:, j] = np.concatenate([
                np.full(window-1, ma[0] if len(ma) > 0 else np.mean(series)),
                ma
            ])
        
        return result
    
    @staticmethod
    def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling standard deviation with proper padding.
        Operates on time axis (axis 0) for each symbol.
        """
        x = x.astype(np.float64)
        result = np.empty_like(x, dtype=np.float64)
        
        # Adjust window size if larger than data
        window = min(window, x.shape[0])
        if window < 2:  # Need at least 2 points for std
            window = 2
        
        # Process each symbol (column)
        for j in range(x.shape[1]):
            series = x[:, j]  # Get time series for this symbol
            
            # Calculate valid std values using cumsum method for efficiency
            cumsum = np.cumsum(np.insert(series, 0, 0))
            cumsum_sq = np.cumsum(np.insert(series**2, 0, 0))
            window_sums = cumsum[window:] - cumsum[:-window]
            window_sum_sqs = cumsum_sq[window:] - cumsum_sq[:-window]
            
            # Calculate std using the formula: std = sqrt(E[X^2] - E[X]^2)
            means = window_sums / window
            mean_sqs = window_sum_sqs / window
            stds = np.sqrt(np.maximum(mean_sqs - means**2, 0))  # Use maximum to avoid negative values due to numerical errors
            
            # Pad the beginning with the first std value
            first_std = stds[0] if len(stds) > 0 else np.std(series[:window])
            result[:, j] = np.concatenate([
                np.full(window-1, first_std),
                stds
            ])
        
        return result

    @staticmethod
    def _rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate rolling Z-score with proper padding.
        Z-score = (x - rolling_mean) / rolling_std
        Operates on time axis (axis 0) for each symbol.
        
        Args:
            x: Input array (time x symbols)
            window: Rolling window size (default 20)
        """
        x = x.astype(np.float64)
        result = np.empty_like(x, dtype=np.float64)
        
        # Adjust window size if larger than data
        window = min(window, x.shape[0])
        if window < 2:  # Need at least 2 points for std
            window = 2
        
        # Process each symbol (column)
        for j in range(x.shape[1]):
            series = x[:, j]  # Get time series for this symbol
            
            # Calculate rolling mean and std using cumsum method
            cumsum = np.cumsum(np.insert(series, 0, 0))
            cumsum_sq = np.cumsum(np.insert(series**2, 0, 0))
            
            # Calculate rolling means
            window_sums = cumsum[window:] - cumsum[:-window]
            means = window_sums / window
            
            # Calculate rolling standard deviations
            window_sum_sqs = cumsum_sq[window:] - cumsum_sq[:-window]
            mean_sqs = window_sum_sqs / window
            stds = np.sqrt(np.maximum(mean_sqs - means**2, 0))  # Use maximum to avoid negative values
            
            # Calculate z-scores for the valid range
            valid_zscore = np.zeros_like(means)
            non_zero_std = stds != 0
            valid_zscore[non_zero_std] = (series[window-1:][non_zero_std] - means[non_zero_std]) / stds[non_zero_std]
            
            # Pad the beginning with the first valid z-score
            first_zscore = valid_zscore[0] if len(valid_zscore) > 0 else 0
            result[:, j] = np.concatenate([
                np.full(window-1, first_zscore),
                valid_zscore
            ])
        
        return result 