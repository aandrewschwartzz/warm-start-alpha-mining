"""
Data loader for market data from Yahoo Finance.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from alpha_tree import DataField

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and preprocesses market data for alpha mining."""
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d'),
        end_date: str = datetime.now().strftime('%Y-%m-%d')
    ):
        """
        Initialize the data loader.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in 'YYYY-MM-DD' format (default: 5 years ago)
            end_date: End date in 'YYYY-MM-DD' format (default: today)
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self._data = None
        self._returns = None
    
    def load_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Load and preprocess market data.
        
        Returns:
            Tuple of (market_data, forward_returns) where:
            - market_data: Dict mapping DataField values to numpy arrays
            - forward_returns: numpy array of 5-day forward returns
            
        Note:
            All numerical data is stored as np.float64 for consistent precision
        """
        if self._data is not None:
            return self._data, self._returns
            
        # Download data for each symbol
        dfs = []
        shares_outstanding = {}
        pe_ratios_dict = {}
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            # Explicitly set dtype for numerical columns
            df = ticker.history(start=self.start_date, end=self.end_date)
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[numeric_columns] = df[numeric_columns].astype(np.float64)
            df['Symbol'] = symbol
            dfs.append(df)
            
            # Convert shares outstanding to float64
            try:
                shares = ticker.info.get('sharesOutstanding', None)
                if shares is not None:
                    shares_outstanding[symbol] = np.float64(shares)
                else:
                    logger.warning(f"No shares outstanding data available for {symbol}. Using default value of 100M.")
                    shares_outstanding[symbol] = np.float64(1e8)
                
                # Convert P/E ratio to float64
                pe = ticker.info.get('trailingPE', None)
                if pe is not None and not np.isnan(pe):
                    pe_ratios_dict[symbol] = np.float64(pe)
                else:
                    reason = "not available" if pe is None else "NaN value"
                    logger.warning(f"P/E ratio data for {symbol} is {reason}. Using market average default value of 15.")
                    pe_ratios_dict[symbol] = np.float64(15)
            except Exception as e:
                logger.warning(f"Error fetching data for {symbol}: {str(e)}. Using default values.")
                shares_outstanding[symbol] = np.float64(1e8)
                pe_ratios_dict[symbol] = np.float64(15)
        
        # Combine all data
        combined_df = pd.concat(dfs)
        
        # Pivot to get time x symbol matrices
        market_data = {}
        for field in DataField:
            # Skip fields that are calculated rather than loaded directly
            if field in [DataField.RETURNS, DataField.VWAP, DataField.INTRADAY_RETURNS, 
                        DataField.TURNOVER_RATE, DataField.PE_RATIO]:
                continue
            
            # Map Yahoo Finance column names to our DataField enum
            yf_column = {
                DataField.OPEN: 'Open',
                DataField.HIGH: 'High',
                DataField.LOW: 'Low',
                DataField.CLOSE: 'Close',
                DataField.VOLUME: 'Volume'
            }[field]
            
            # Create time x symbol matrix with explicit dtype
            matrix = combined_df.pivot(
                columns='Symbol',
                values=yf_column
            ).values.astype(np.float64)
            
            market_data[field.value] = matrix
        
        # Get volumes and calculate turnover rate
        volumes = market_data[DataField.VOLUME.value]
        # Create array of shares outstanding in same order as symbols
        shares_array = np.array([shares_outstanding[symbol] for symbol in combined_df['Symbol'].unique()], dtype=np.float64)
        # Calculate turnover rate (volume / shares outstanding)
        turnover_rate = volumes / shares_array[None, :]  # Broadcasting for division
        market_data[DataField.TURNOVER_RATE.value] = turnover_rate
        
        # Calculate daily VWAP
        high_prices = market_data[DataField.HIGH.value]
        low_prices = market_data[DataField.LOW.value]
        close_prices = market_data[DataField.CLOSE.value]
        volumes = market_data[DataField.VOLUME.value]
        
        # Calculate HLCC3 typical price for each day as (high + low + close) / 3
        typical_prices = (high_prices + low_prices + close_prices) / 3
        
        # Calculate daily VWAP as (typical_price * volume) / volume
        daily_pv = typical_prices * volumes
        vwap = np.divide(daily_pv, volumes, out=np.zeros_like(daily_pv), where=volumes!=0)
        
        market_data[DataField.VWAP.value] = vwap
        
        # Calculate intraday returns as (close - open) / open
        open_prices = market_data[DataField.OPEN.value]
        intraday_returns = np.divide(close_prices - open_prices, open_prices, 
                                   out=np.zeros_like(close_prices), where=open_prices!=0)
        market_data[DataField.INTRADAY_RETURNS.value] = intraday_returns
        
        # Calculate P/E ratio by scaling the base P/E with price changes
        close_prices = market_data[DataField.CLOSE.value]
        # Get initial prices (first day) for scaling
        initial_prices = close_prices[0]
        # Create array of base P/E ratios in same order as symbols
        base_pe_array = np.array([pe_ratios_dict[symbol] for symbol in combined_df['Symbol'].unique()], dtype=np.float64)
        
        # Scale P/E ratios based on price changes from initial day
        # This assumes earnings remain constant during the period
        price_scalars = close_prices / initial_prices[None, :]
        pe_ratios = base_pe_array[None, :] * price_scalars
        
        # Cap extreme P/E values at ±1000 to handle outliers
        pe_ratios = np.clip(pe_ratios, -1000, 1000)
        market_data[DataField.PE_RATIO.value] = pe_ratios
        
        # Calculate daily returns (close-to-close)
        returns = np.zeros_like(close_prices, dtype=np.float64)
        returns[:-1] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]  # t+1 / t - 1
        returns[-1] = np.nan  # Last day has no next day to calculate return
        market_data[DataField.RETURNS.value] = returns
        
        # Calculate 5-day forward returns for fitness evaluation
        forward_returns = np.zeros_like(close_prices, dtype=np.float64)
        for i in range(len(close_prices)-5):
            forward_returns[i] = (close_prices[i+5] / close_prices[i]) - 1
        forward_returns[-5:] = np.nan  # Last 5 days can't have 5-day forward returns
        
        self._data = market_data
        self._returns = forward_returns
        
        return market_data, forward_returns
    
    def split_data(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.4
    ) -> Tuple[
        Tuple[Dict[str, np.ndarray], np.ndarray],
        Tuple[Dict[str, np.ndarray], np.ndarray]
    ]:
        """
        Split data into train and validation sets.
        
        Args:
            train_ratio: Proportion of data for training (default: 0.6)
            val_ratio: Proportion of data for validation (default: 0.4)
            
        Returns:
            Tuple of (train_data, val_data) where each is a tuple of
            (market_data, forward_returns)
        """
        market_data, forward_returns = self.load_data()
        
        # Calculate split index
        n_samples = len(forward_returns)
        train_idx = int(n_samples * train_ratio)
        
        # Split market data
        train_market_data = {
            k: v[:train_idx] for k, v in market_data.items()
        }
        val_market_data = {
            k: v[train_idx:] for k, v in market_data.items()
        }
        
        # Split forward returns
        train_returns = forward_returns[:train_idx]
        val_returns = forward_returns[train_idx:]
        
        return (
            (train_market_data, train_returns),
            (val_market_data, val_returns)
        )

    def validate_returns(self) -> None:
        """Validate that returns and intraday returns are different"""
        market_data, _ = self.load_data()
        
        returns = market_data[DataField.RETURNS.value]
        intraday = market_data[DataField.INTRADAY_RETURNS.value]
        
        # Check if arrays are identical
        are_identical = np.allclose(returns, intraday, equal_nan=True)
        
        # Print some statistics
        print("Returns vs Intraday Returns Validation:")
        print(f"Arrays identical: {are_identical}")
        print(f"Returns mean: {np.nanmean(returns):.6f}")
        print(f"Intraday mean: {np.nanmean(intraday):.6f}")
        print(f"Returns std: {np.nanstd(returns):.6f}")
        print(f"Intraday std: {np.nanstd(intraday):.6f}")
        
        # Print first few non-zero differences
        diff = returns - intraday
        non_zero = np.where(~np.isclose(returns, intraday, equal_nan=True))
        if len(non_zero[0]) > 0:
            print("\nFirst few differences:")
            for i in range(min(5, len(non_zero[0]))):
                t, s = non_zero[0][i], non_zero[1][i]
                print(f"t={t}, symbol={s}: returns={returns[t,s]:.6f}, intraday={intraday[t,s]:.6f}") 

    def get_dates(self) -> pd.DatetimeIndex:
        """
        Get the actual dates from the loaded data, without time components.
        
        Returns:
            pd.DatetimeIndex: Index of dates (without time components) for which we have data
        """
        # Download data for first symbol to get the date index
        # We use the first symbol since all symbols should have the same dates after pivoting
        ticker = yf.Ticker(self.symbols[0])
        df = ticker.history(start=self.start_date, end=self.end_date)
        # Convert to date-only index by normalizing to midnight UTC and then converting to date
        return pd.DatetimeIndex([d.date() for d in df.index])
    
    def get_split_dates(self, train_ratio: float = 0.6) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        Get the dates for train and validation sets.
        
        Args:
            train_ratio: Proportion of data for training (default: 0.6)
            
        Returns:
            Tuple[pd.DatetimeIndex, pd.DatetimeIndex]: (train_dates, val_dates)
        """
        dates = self.get_dates()
        train_idx = int(len(dates) * train_ratio)
        
        return dates[:train_idx], dates[train_idx:] 