import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from data_loader import DataLoader
from cwmr_env import CWMRTradingEnv
from rl_trainer import RLTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='CWMR-RL: Confidence-Weighted Mean Reversion with Reinforcement Learning')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='/Users/ryanmathieu/Downloads/stock_market_data/sp500/csv',
                        help='Directory containing stock data CSV files')
    parser.add_argument('--max_stocks', type=int, default=20,
                        help='Maximum number of stocks to include in portfolio')
    parser.add_argument('--start_date', type=str, default='2010-01-01',
                        help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2020-12-31',
                        help='End date for analysis (YYYY-MM-DD)')
    
    # CWMR parameters
    parser.add_argument('--window_size', type=int, default=10,
                        help='Size of observation window for mean reversion')
    parser.add_argument('--transaction_cost', type=float, default=0.001,
                        help='Transaction cost as a fraction')
    parser.add_argument('--confidence_bound', type=float, default=0.1,
                        help='Initial confidence bound for CWMR')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='Target mean reversion threshold')
    
    # RL parameters
    parser.add_argument('--total_timesteps', type=int, default=100000,
                        help='Total timesteps for RL training')
    parser.add_argument('--eval_freq', type=int, default=10000,
                        help='Evaluation frequency during training')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate for RL algorithm')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for RL algorithm')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for RL algorithm')
    parser.add_argument('--reward_scaling', type=float, default=1.0,
                        help='Scaling factor for rewards')
    parser.add_argument('--n_envs', type=int, default=1,
                        help='Number of parallel environments')
    
    # Training control
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Proportion of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Proportion of data for testing')
    parser.add_argument('--tune_hyperparams', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--use_features', action='store_true',
                        help='Use technical features for state representation')
    
    # Output parameters
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for logs')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory for saving models')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory for saving results')
    parser.add_argument('--model_name', type=str, default='ppo_cwmr',
                        help='Name of the model')
    
    args = parser.parse_args()
    return args

def main():
    # Parse command-line arguments
    args = parse_args()
    
    print("\n===== CWMR-RL: Confidence-Weighted Mean Reversion with Reinforcement Learning =====")
    print(f"Running with arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data_loader = DataLoader(
        data_dir=args.data_dir,
        max_stocks=args.max_stocks,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Load stock price data
    stock_data = data_loader.load_stock_files(min_history_days=1000)
    
    # Calculate returns
    returns_data = data_loader.compute_returns(method='log')
    
    # Display basic statistics
    stats = data_loader.get_data_stats()
    print("\nData Statistics:")
    for stat, values in stats.items():
        print(f"{stat}: Mean across stocks = {values.mean():.6f}")
    
    # Create visualizations
    data_loader.plot_prices(n_stocks=5)
    data_loader.plot_returns(n_stocks=5)
    
    # Step 2: Create features for mean reversion
    print("\n2. Creating mean reversion features...")
    if args.use_features:
        returns, features = data_loader.create_mean_reversion_features()
        print(f"Features shape: {features.shape}")
        print(f"Aligned returns shape: {returns.shape}")
    else:
        returns = returns_data.values
        features = None
    
    # Step 3: Split data into train, validation, and test sets
    print("\n3. Splitting data...")
    train_data, val_data, test_data = data_loader.split_data(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Split features if they exist
    if features is not None:
        feature_data = features.values
        n = len(feature_data)
        train_idx = int(n * args.train_ratio)
        val_idx = train_idx + int(n * args.val_ratio)
        
        train_features = feature_data[:train_idx]
        val_features = feature_data[train_idx:val_idx]
        test_features = feature_data[val_idx:]
    else:
        train_features = None
        val_features = None
        test_features = None
    
    # Step 4: Initialize the RL trainer
    print("\n4. Initializing RL trainer...")
    trainer = RLTrainer(
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        results_dir=args.results_dir
    )
    
    # Step 5: Hyperparameter tuning (optional)
    if args.tune_hyperparams:
        print("\n5a. Performing hyperparameter tuning...")
        best_params = trainer.hyperparameter_tuning(
            train_returns=train_data,
            val_returns=val_data,
            train_features=train_features,
            val_features=val_features
        )
        
        # Update arguments with best parameters
        args.window_size = best_params['window_size']
        args.transaction_cost = best_params['transaction_cost']
        args.confidence_bound = best_params['confidence_bound']
        args.epsilon = best_params['epsilon']
        args.learning_rate = best_params['learning_rate']
        
        print("\nUsing best hyperparameters for training:")
        print(f"  window_size: {args.window_size}")
        print(f"  transaction_cost: {args.transaction_cost}")
        print(f"  confidence_bound: {args.confidence_bound}")
        print(f"  epsilon: {args.epsilon}")
        print(f"  learning_rate: {args.learning_rate}")
    
    # Step 6: Prepare the environments
    print("\n5. Preparing environments...")
    train_env, eval_env = trainer.prepare_environment(
        train_returns=train_data,
        train_features=train_features,
        val_returns=val_data,
        val_features=val_features,
        window_size=args.window_size,
        transaction_cost=args.transaction_cost,
        confidence_bound=args.confidence_bound,
        epsilon=args.epsilon,
        reward_scaling=args.reward_scaling,
        include_feature_history=False,
        n_envs=args.n_envs
    )
    
    # Step 7: Train the RL model
    print("\n6. Training RL model...")
    model = trainer.train(
        train_env=train_env,
        eval_env=eval_env,
        model_name=args.model_name,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma
    )
    
    # Step 8: Create test environment
    print("\n7. Evaluating model on test data...")
    test_env = CWMRTradingEnv(
        returns_data=test_data,
        feature_data=test_features,
        window_size=args.window_size,
        transaction_cost=args.transaction_cost,
        confidence_bound=args.confidence_bound,
        epsilon=args.epsilon,
        reward_scaling=args.reward_scaling
    )
    
    # Step 9: Evaluate the model
    eval_results = trainer.evaluate(
        model=model,
        test_env=test_env,
        n_episodes=1,
        deterministic=True,
        render=False
    )
    
    # Step 10: Plot evaluation results
    trainer.plot_evaluation_results(
        results=eval_results,
        model_name=args.model_name
    )
    
    # Step 11: Compare with baselines
    print("\n8. Comparing with baseline strategies...")
    comparison_results = trainer.compare_with_baseline(
        test_returns=test_data,
        test_features=test_features,
        model=model
    )
    
    print("\n===== CWMR-RL Complete =====")
    print(f"Results saved to {args.results_dir}")

if __name__ == "__main__":
    main() 