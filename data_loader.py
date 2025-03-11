import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class DataLoader:
    """
    Loads and preprocesses stock price data from CSV files for CWMR-RL training.
    This class handles loading multiple stocks, computing returns, and 
    preparing the data for the RL environment.
    """
    
    def __init__(self, data_dir, max_stocks=20, start_date=None, end_date=None):
        """
        Initialize the DataLoader with directory and filtering parameters.
        
        Args:
            data_dir (str): Directory containing CSV stock data files
            max_stocks (int): Maximum number of stocks to include
            start_date (str): Start date for filtering data (YYYY-MM-DD)
            end_date (str): End date for filtering data (YYYY-MM-DD)
        """
        self.data_dir = Path(data_dir)
        self.max_stocks = max_stocks
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.stock_data = None
        self.returns_data = None
        self.symbols = []
        
    def load_stock_files(self, min_history_days=1000):
        """
        Load stock data from CSV files, filtering for stocks with enough history.
        
        Args:
            min_history_days (int): Minimum number of trading days required
            
        Returns:
            DataFrame: Combined DataFrame of stock prices
        """
        print(f"Loading stock data from {self.data_dir}...")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files.")
        
        # Dictionary to store individual stock DataFrames
        stock_dfs = {}
        
        # Process each CSV file to extract price data
        for i, csv_file in enumerate(csv_files):
            if i >= self.max_stocks:
                break
                
            symbol = csv_file.stem
            
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Check if required columns exist
                required_cols = ['Date', 'Adjusted Close']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Missing required columns in {csv_file}")
                
                # Parse dates with dayfirst=True to handle the format correctly
                try:
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
                except:
                    # Fallback to mixed format if it fails
                    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
                
                # Filter by date range if specified
                if self.start_date:
                    df = df[df['Date'] >= self.start_date]
                if self.end_date:
                    df = df[df['Date'] <= self.end_date]
                
                # Skip if not enough history
                if len(df) < min_history_days:
                    continue
                
                # Keep only date and adjusted close price
                df = df[['Date', 'Adjusted Close']]
                df = df.rename(columns={'Adjusted Close': symbol})
                df = df.sort_values('Date')
                
                stock_dfs[symbol] = df
                self.symbols.append(symbol)
                
                if len(stock_dfs) % 5 == 0:
                    print(f"Processed {len(stock_dfs)} stocks...")
                
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
        
        # Merge all stock DataFrames on the Date column
        if not stock_dfs:
            raise ValueError("No valid stock data found")
            
        print(f"Successfully loaded {len(stock_dfs)} stocks.")
        
        merged_df = None
        for symbol, df in stock_dfs.items():
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='Date', how='inner')
        
        merged_df.set_index('Date', inplace=True)
        self.stock_data = merged_df
        
        print(f"Final dataset contains {len(self.stock_data)} trading days for {len(self.symbols)} stocks.")
        return self.stock_data
    
    def compute_returns(self, method='log'):
        """
        Compute returns from price data.
        
        Args:
            method (str): 'log' for log returns, 'pct' for percentage returns
            
        Returns:
            DataFrame: Returns data
        """
        if self.stock_data is None:
            raise ValueError("No stock data loaded. Call load_stock_files first.")
        
        if method == 'log':
            # Log returns: ln(P_t / P_{t-1})
            self.returns_data = np.log(self.stock_data / self.stock_data.shift(1)).dropna()
        else:
            # Percentage returns: (P_t / P_{t-1}) - 1
            self.returns_data = (self.stock_data / self.stock_data.shift(1) - 1).dropna()
            
        return self.returns_data
    
    def split_data(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """
        Split the data into training, validation, and test sets.
        
        Args:
            train_ratio (float): Proportion for training
            val_ratio (float): Proportion for validation
            test_ratio (float): Proportion for testing
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        if self.returns_data is None:
            raise ValueError("No returns data computed. Call compute_returns first.")
            
        # Check that ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
        
        data = self.returns_data.values
        n = len(data)
        
        # Calculate split indices
        train_idx = int(n * train_ratio)
        val_idx = train_idx + int(n * val_ratio)
        
        # Split the data
        train_data = data[:train_idx]
        val_data = data[train_idx:val_idx]
        test_data = data[val_idx:]
        
        print(f"Data split: Train {train_data.shape}, Validation {val_data.shape}, Test {test_data.shape}")
        
        return train_data, val_data, test_data
    
    def get_data_stats(self):
        """
        Calculate and return statistics about the data.
        
        Returns:
            dict: Dictionary of statistics
        """
        if self.returns_data is None:
            raise ValueError("No returns data computed. Call compute_returns first.")
            
        stats = {
            'mean': self.returns_data.mean(),
            'std': self.returns_data.std(),
            'min': self.returns_data.min(),
            'max': self.returns_data.max(),
            'median': self.returns_data.median(),
            'skew': self.returns_data.skew(),
            'kurtosis': self.returns_data.kurtosis()
        }
        
        return stats
    
    def plot_prices(self, n_stocks=5):
        """
        Plot price data for a subset of stocks.
        
        Args:
            n_stocks (int): Number of stocks to plot
        """
        if self.stock_data is None:
            raise ValueError("No stock data loaded. Call load_stock_files first.")
            
        plt.figure(figsize=(12, 6))
        sample_stocks = self.symbols[:n_stocks]
        
        for symbol in sample_stocks:
            # Normalize to start at 1.0 for better comparison
            normalized = self.stock_data[symbol] / self.stock_data[symbol].iloc[0]
            plt.plot(self.stock_data.index, normalized, label=symbol)
            
        plt.title('Normalized Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True)
        plt.savefig('stock_prices.png')
        plt.close()
    
    def plot_returns(self, n_stocks=5):
        """
        Plot return distributions for a subset of stocks.
        
        Args:
            n_stocks (int): Number of stocks to plot
        """
        if self.returns_data is None:
            raise ValueError("No returns data computed. Call compute_returns first.")
            
        plt.figure(figsize=(12, 6))
        sample_stocks = self.symbols[:n_stocks]
        
        for symbol in sample_stocks:
            self.returns_data[symbol].hist(bins=50, alpha=0.3, label=symbol)
            
        plt.title('Return Distributions')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig('return_distributions.png')
        plt.close()
        
    def create_mean_reversion_features(self, window_sizes=[5, 10, 20, 50]):
        """
        Create mean reversion features for the returns data.
        
        Args:
            window_sizes (list): List of window sizes for calculating z-scores
            
        Returns:
            tuple: (aligned_returns_df, features_df)
        """
        if self.returns_data is None:
            raise ValueError("No returns data computed. Call compute_returns first.")
            
        features = {}
        
        # 1. Z-scores for price levels
        for symbol in self.symbols:
            prices = self.stock_data[symbol]
            for window in window_sizes:
                # Price z-score
                rolling_mean = prices.rolling(window=window).mean()
                rolling_std = prices.rolling(window=window).std()
                z_score = (prices - rolling_mean) / (rolling_std + 1e-8)
                features[f"{symbol}_price_zscore_{window}"] = z_score
                
                # Returns z-score
                returns = self.returns_data[symbol]
                ret_mean = returns.rolling(window=window).mean()
                ret_std = returns.rolling(window=window).std()
                ret_zscore = (returns - ret_mean) / (ret_std + 1e-8)
                features[f"{symbol}_return_zscore_{window}"] = ret_zscore
        
        # 2. Momentum and mean reversion indicators
        for symbol in self.symbols:
            prices = self.stock_data[symbol]
            returns = self.returns_data[symbol]
            
            # Price-based features
            for window in window_sizes:
                # Distance from moving average
                ma = prices.rolling(window=window).mean()
                distance = (prices - ma) / prices
                features[f"{symbol}_ma_dist_{window}"] = distance
                
                # Momentum
                momentum = prices.pct_change(window)
                features[f"{symbol}_momentum_{window}"] = momentum
                
                # Volatility
                volatility = returns.rolling(window=window).std()
                features[f"{symbol}_volatility_{window}"] = volatility
        
        # 3. Cross-sectional features
        for window in window_sizes:
            # Cross-sectional z-scores
            cs_mean = self.returns_data.rolling(window=window).mean()
            cs_std = self.returns_data.rolling(window=window).std()
            
            for symbol in self.symbols:
                # Cross-sectional ranking
                rank = self.returns_data[symbol].rolling(window=window).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                )
                features[f"{symbol}_cs_rank_{window}"] = rank
                
                # Cross-sectional momentum
                cs_momentum = (self.returns_data[symbol] - cs_mean.mean(axis=1)) / (cs_std.mean(axis=1) + 1e-8)
                features[f"{symbol}_cs_momentum_{window}"] = cs_momentum
        
        # Create DataFrame and align with returns data
        features_df = pd.DataFrame(features, index=self.stock_data.index)
        
        # Forward fill and backward fill to handle any remaining NaN values
        features_df = features_df.ffill().bfill()  # Updated to use newer methods
        
        # Ensure alignment with returns data
        aligned_returns = self.returns_data
        aligned_features = features_df.loc[aligned_returns.index]
        
        # Print shapes for debugging
        print(f"Features shape: {aligned_features.shape}")
        print(f"Aligned returns shape: {aligned_returns.shape}")
        
        return aligned_returns, aligned_features
    
if __name__ == "__main__":
    # Example usage
    data_dir = "../stock_market_data/sp500/csv"
    loader = DataLoader(data_dir, max_stocks=10, start_date="2010-01-01", end_date="2020-12-31")
    
    # Load data
    price_data = loader.load_stock_files(min_history_days=1000)
    
    # Compute returns
    returns_data = loader.compute_returns(method='log')
    
    # Display basic statistics
    stats = loader.get_data_stats()
    print("\nData Statistics:")
    for stat, values in stats.items():
        print(f"{stat}: Mean across stocks = {values.mean():.6f}")
    
    # Create visualizations
    loader.plot_prices(n_stocks=5)
    loader.plot_returns(n_stocks=5)
    
    # Split the data
    train_data, val_data, test_data = loader.split_data()
    
    # Generate mean reversion features
    returns, features = loader.create_mean_reversion_features()
    print(f"\nFeatures shape: {features.shape}")
    print(f"Aligned returns shape: {returns.shape}") 