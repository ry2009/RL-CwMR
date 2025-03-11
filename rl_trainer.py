import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import seaborn as sns

from cwmr_env import CWMRTradingEnv

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics during training.
    """
    def __init__(self, eval_env, eval_freq=1000, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation
            obs, info = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = self.eval_env.step(action)
                done = done or truncated
            
            # Get metrics
            metrics = self.eval_env.get_performance_metrics()
            
            # Log to tensorboard
            for metric_name, metric_value in metrics.items():
                self.logger.record(f"eval/{metric_name}", metric_value)
                
        return True

class RLTrainer:
    """
    Trainer class for reinforcement learning models applied to the CWMR trading environment.
    """
    def __init__(self, 
                 log_dir='logs',
                 save_dir='models',
                 results_dir='results'):
        """
        Initialize the RL trainer.
        
        Args:
            log_dir (str): Directory for logs
            save_dir (str): Directory for saving models
            results_dir (str): Directory for saving results
        """
        self.log_dir = Path(log_dir)
        self.save_dir = Path(save_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories if they don't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_environment(self, 
                          train_returns, 
                          train_features=None,
                          val_returns=None,
                          val_features=None,
                          window_size=10,
                          transaction_cost=0.001,
                          confidence_bound=0.1,
                          epsilon=0.01,
                          reward_scaling=1.0,
                          include_feature_history=False,
                          n_envs=1):
        """
        Prepare training and evaluation environments.
        
        Args:
            train_returns (numpy.ndarray): Training returns data
            train_features (numpy.ndarray, optional): Training feature data
            val_returns (numpy.ndarray, optional): Validation returns data
            val_features (numpy.ndarray, optional): Validation feature data
            window_size (int): Size of the observation window
            transaction_cost (float): Transaction cost as a fraction
            confidence_bound (float): Initial confidence bound for CWMR
            epsilon (float): Target mean reversion threshold
            reward_scaling (float): Scaling factor for rewards
            include_feature_history (bool): Whether to include feature history in state
            n_envs (int): Number of parallel environments
            
        Returns:
            tuple: (vectorized training environment, evaluation environment)
        """
        # Environment configuration
        env_config = {
            'window_size': window_size,
            'transaction_cost': transaction_cost,
            'confidence_bound': confidence_bound,
            'epsilon': epsilon,
            'reward_scaling': reward_scaling,
            'include_feature_history': include_feature_history
        }
        
        # Create a function to initialize a training environment
        def make_train_env():
            env = CWMRTradingEnv(
                returns_data=train_returns,
                feature_data=train_features,
                **env_config
            )
            env = Monitor(env)
            return env
        
        # Create vectorized environment for training
        if n_envs > 1:
            train_env = make_vec_env(
                make_train_env, 
                n_envs=n_envs, 
                vec_env_cls=SubprocVecEnv,
                monitor_dir=str(self.log_dir / 'monitor')
            )
        else:
            train_env = make_train_env()
            
        # Create evaluation environment if validation data is provided
        if val_returns is not None:
            eval_env = CWMRTradingEnv(
                returns_data=val_returns,
                feature_data=val_features,
                **env_config
            )
        else:
            # Use a separate instance with the same training data
            eval_env = CWMRTradingEnv(
                returns_data=train_returns,
                feature_data=train_features,
                **env_config
            )
        
        return train_env, eval_env
    
    def train(self, 
             train_env, 
             eval_env,
             model_name='ppo_cwmr',
             total_timesteps=100000,
             eval_freq=10000,
             n_eval_episodes=1,
             learning_rate=3e-4,
             batch_size=64,
             gamma=0.99,
             policy_kwargs=None):
        """
        Train the RL model.
        
        Args:
            train_env (gym.Env): Training environment
            eval_env (gym.Env): Evaluation environment
            model_name (str): Name of the model
            total_timesteps (int): Total number of training timesteps
            eval_freq (int): Evaluation frequency
            n_eval_episodes (int): Number of episodes for evaluation
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            gamma (float): Discount factor
            policy_kwargs (dict, optional): Additional policy arguments
            
        Returns:
            stable_baselines3.PPO: Trained PPO model
        """
        # Set up model directories
        model_dir = self.save_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Default policy kwargs if not provided
        if policy_kwargs is None:
            policy_kwargs = {
                'net_arch': [128, 128]
            }
        
        # Initialize PPO model
        print("Initializing PPO model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            verbose=1,
            tensorboard_log=str(self.log_dir),
            policy_kwargs=policy_kwargs
        )
        
        # Set up callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(self.log_dir / 'eval'),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        )
        
        custom_callback = TensorboardCallback(
            eval_env=eval_env,
            eval_freq=eval_freq
        )
        
        # Train the model
        print(f"Training model for {total_timesteps} timesteps...")
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, custom_callback],
            tb_log_name=model_name
        )
        
        # Save final model
        final_model_path = model_dir / "final_model.zip"
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return model
    
    def evaluate(self, model, test_env, n_episodes=1, deterministic=True, render=False):
        """
        Evaluate a trained model.
        
        Args:
            model (stable_baselines3.PPO): Trained model
            test_env (gym.Env): Test environment
            n_episodes (int): Number of evaluation episodes
            deterministic (bool): Whether to use deterministic actions
            render (bool): Whether to render the environment
            
        Returns:
            dict: Evaluation metrics
        """
        # Lists to store results
        episode_returns = []
        episode_sharpe_ratios = []
        episode_drawdowns = []
        episode_weights = []
        
        # Evaluate over multiple episodes
        for episode in range(n_episodes):
            print(f"Evaluating episode {episode + 1}/{n_episodes}...")
            
            # Reset environment
            obs, info = test_env.reset()
            done = False
            truncated = False
            
            # Single episode evaluation
            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = test_env.step(action)
                
                if render:
                    test_env.render()
                    
            # Collect metrics after episode
            metrics = test_env.get_performance_metrics()
            episode_returns.append(metrics['total_return'])
            episode_sharpe_ratios.append(metrics['sharpe_ratio'])
            episode_drawdowns.append(metrics['max_drawdown'])
            episode_weights.append(test_env.weight_history)
            
        # Aggregate results
        results = {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_sharpe': np.mean(episode_sharpe_ratios),
            'mean_drawdown': np.mean(episode_drawdowns),
            'all_returns': episode_returns,
            'all_sharpe_ratios': episode_sharpe_ratios,
            'all_drawdowns': episode_drawdowns,
            'weight_history': episode_weights[0] if episode_weights else None,
            'portfolio_values': test_env.portfolio_values
        }
        
        # Print summary
        print("\nEvaluation Results:")
        print(f"Mean Return: {results['mean_return']:.4f}")
        print(f"Mean Sharpe Ratio: {results['mean_sharpe']:.4f}")
        print(f"Mean Max Drawdown: {results['mean_drawdown']:.4f}")
        
        return results
    
    def plot_evaluation_results(self, results, model_name='model'):
        """
        Plot evaluation results.
        
        Args:
            results (dict): Evaluation results
            model_name (str): Model name for plot titles
        """
        # Create results directory if it doesn't exist
        results_dir = self.results_dir / model_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Portfolio value over time
        plt.figure(figsize=(12, 6))
        plt.plot(results['portfolio_values'])
        plt.title(f'Portfolio Value - {model_name}')
        plt.xlabel('Timesteps')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.savefig(results_dir / 'portfolio_value.png')
        plt.close()
        
        # 2. Portfolio weights over time (heatmap)
        if results['weight_history'] is not None:
            weights = np.array(results['weight_history'])
            plt.figure(figsize=(14, 8))
            plt.imshow(weights.T, aspect='auto', cmap='viridis')
            plt.colorbar(label='Weight')
            plt.title(f'Portfolio Weights Over Time - {model_name}')
            plt.xlabel('Timesteps')
            plt.ylabel('Assets')
            plt.savefig(results_dir / 'weight_heatmap.png')
            plt.close()
            
            # 3. Weight evolution for top assets
            plt.figure(figsize=(14, 8))
            for i in range(min(5, weights.shape[1])):
                plt.plot(weights[:, i], label=f'Asset {i+1}')
            plt.title(f'Top Asset Weights Over Time - {model_name}')
            plt.xlabel('Timesteps')
            plt.ylabel('Weight')
            plt.legend()
            plt.grid(True)
            plt.savefig(results_dir / 'top_weights.png')
            plt.close()
        
        # 4. Performance comparison (if available)
        if 'benchmark_values' in results:
            plt.figure(figsize=(12, 6))
            plt.plot(results['portfolio_values'], label='RL-CWMR')
            plt.plot(results['benchmark_values'], label='Buy and Hold')
            plt.title(f'Performance Comparison - {model_name}')
            plt.xlabel('Timesteps')
            plt.ylabel('Portfolio Value')
            plt.legend()
            plt.grid(True)
            plt.savefig(results_dir / 'performance_comparison.png')
            plt.close()
        
        # Save numerical results
        with open(results_dir / 'results_summary.txt', 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Mean Return: {results['mean_return']:.4f}\n")
            f.write(f"Mean Sharpe Ratio: {results['mean_sharpe']:.4f}\n")
            f.write(f"Mean Max Drawdown: {results['mean_drawdown']:.4f}\n")
        
        print(f"Evaluation plots saved to {results_dir}")

    def compare_with_baseline(self, test_returns, test_features=None, model=None):
        """
        Compare RL model with baseline strategies.
        
        Args:
            test_returns (numpy.ndarray): Test returns data
            test_features (numpy.ndarray, optional): Test feature data
            model (stable_baselines3.PPO, optional): Trained RL model
            
        Returns:
            dict: Comparison results
        """
        # 1. Equal weight (buy and hold)
        def equal_weight_strategy(returns):
            n_assets = returns.shape[1]
            n_steps = returns.shape[0]
            
            # Equal weights for all assets
            weights = np.ones((n_steps, n_assets)) / n_assets
            
            # Calculate portfolio values
            portfolio_values = [1.0]
            for t in range(n_steps):
                portfolio_return = np.sum(weights[t] * returns[t])
                portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                
            return {
                'strategy': 'Equal Weight',
                'portfolio_values': portfolio_values,
                'final_value': portfolio_values[-1],
                'total_return': portfolio_values[-1] - 1.0
            }
            
        # 2. Basic CWMR (without RL)
        def basic_cwmr_strategy(returns, window_size=10, epsilon=0.01, confidence_bound=0.1, transaction_cost=0.001):
            """
            Improved implementation of the basic CWMR strategy with proper numerical stability.
            
            Args:
                returns (numpy.ndarray): Returns data [time, n_assets]
                window_size (int): Lookback window for mean reversion
                epsilon (float): Loss threshold
                confidence_bound (float): Initial confidence bound
                transaction_cost (float): Transaction cost rate
                
            Returns:
                dict: Strategy results
            """
            n_assets = returns.shape[1]
            n_steps = returns.shape[0]
            
            # Initialize CWMR parameters
            weights = np.ones(n_assets) / n_assets  # Initial portfolio weights
            sigma = np.eye(n_assets) * confidence_bound  # Initial covariance matrix
            
            # Storage for results
            weight_history = [weights.copy()]
            portfolio_values = [1.0]
            portfolio_returns = []
            
            # Main CWMR loop
            for t in range(window_size, n_steps):
                # Get historical returns for mean reversion signal
                hist_returns = returns[t-window_size:t]
                
                # Calculate mean reversion signal
                mean_returns = np.mean(hist_returns, axis=0)
                curr_returns = returns[t]
                
                # Price prediction (mean reversion assumption)
                price_prediction = mean_returns - curr_returns
                
                # Ensure numerical stability of prediction
                if np.isnan(price_prediction).any() or np.isinf(price_prediction).any():
                    portfolio_return = np.sum(weights * curr_returns)
                    portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                    weight_history.append(weights.copy())
                    portfolio_returns.append(portfolio_return)
                    continue
                    
                # Calculate loss
                loss = max(0, epsilon - np.dot(weights, price_prediction))
                
                if loss > 0:
                    # Calculate confidence
                    var = np.dot(np.dot(price_prediction, sigma), price_prediction)
                    
                    # Ensure numerical stability
                    if var < 1e-10:
                        portfolio_return = np.sum(weights * curr_returns)
                        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                        weight_history.append(weights.copy())
                        portfolio_returns.append(portfolio_return)
                        continue
                        
                    # Calculate lambda with clipping for stability
                    lambda_value = np.clip(loss / var, 0, 1e6)
                    
                    # Update weights
                    new_weights = weights + lambda_value * np.dot(sigma, price_prediction)
                    
                    # Project weights onto simplex
                    new_weights = np.maximum(new_weights, 0)
                    weight_sum = np.sum(new_weights)
                    if weight_sum > 0:
                        new_weights = new_weights / weight_sum
                    else:
                        new_weights = np.ones(n_assets) / n_assets
                        
                    # Update covariance matrix
                    sigma_price = np.dot(sigma, price_prediction)
                    denominator = 1.0 / lambda_value + np.dot(price_prediction, sigma_price)
                    if denominator > 1e-10:
                        sigma = sigma - np.outer(sigma_price, sigma_price) / denominator
                        
                        # Ensure positive definiteness
                        eigvals, eigvecs = np.linalg.eigh(sigma)
                        eigvals = np.maximum(eigvals, 1e-8)
                        sigma = eigvecs.dot(np.diag(eigvals)).dot(eigvecs.T)
                    
                    # Calculate transaction costs
                    turnover = np.sum(np.abs(new_weights - weights))
                    transaction_cost_value = transaction_cost * turnover
                    
                    # Update portfolio value
                    portfolio_return = np.sum(weights * curr_returns) * (1 - transaction_cost_value)
                    portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                    
                    # Store updated weights
                    weights = new_weights.copy()
                    
                else:
                    # No update needed
                    portfolio_return = np.sum(weights * curr_returns)
                    portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                
                weight_history.append(weights.copy())
                portfolio_returns.append(portfolio_return)
            
            # Calculate metrics
            portfolio_returns = np.array(portfolio_returns)
            sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-10) * np.sqrt(252)
            max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values)
            
            return {
                'strategy': 'Basic CWMR',
                'portfolio_values': portfolio_values,
                'final_value': portfolio_values[-1],
                'total_return': portfolio_values[-1] - 1.0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'weights': weight_history
            }
            
        # 3. GRPO-CWMR
        def grpo_cwmr_strategy(returns, window_size=10, epsilon=0.01, confidence_bound=0.1, transaction_cost=0.001, 
                              group_size=5, n_groups=4):
            """
            GRPO-CWMR: Group Relativity Policy Optimization enhanced CWMR.
            This implementation uses group-based relative performance to guide portfolio updates.
            
            Args:
                returns (numpy.ndarray): Returns data [time, n_assets]
                window_size (int): Lookback window for mean reversion
                epsilon (float): Loss threshold
                confidence_bound (float): Initial confidence bound
                transaction_cost (float): Transaction cost rate
                group_size (int): Size of each asset group
                n_groups (int): Number of groups to maintain
                
            Returns:
                dict: Strategy results
            """
            n_assets = returns.shape[1]
            n_steps = returns.shape[0]
            
            # Initialize CWMR parameters
            weights = np.ones(n_assets) / n_assets
            sigma = np.eye(n_assets) * confidence_bound
            
            # Storage for results
            weight_history = [weights.copy()]
            portfolio_values = [1.0]
            portfolio_returns = []
            
            # Initialize group assignments - ensure at least 2 groups
            n_total_groups = max(2, min(n_groups, n_assets // max(1, group_size)))
            group_assignments = np.zeros(n_assets, dtype=int)
            
            for t in range(window_size, n_steps):
                # Update group assignments based on recent performance
                if t % window_size == 0:
                    # Calculate recent performance for each asset
                    recent_returns = returns[t-window_size:t]
                    asset_performance = np.mean(recent_returns, axis=0)
                    
                    # Sort assets by performance and assign to groups
                    sorted_indices = np.argsort(asset_performance)[::-1]
                    for i, idx in enumerate(sorted_indices):
                        group_assignments[idx] = min(i // max(1, group_size), n_total_groups - 1)
                
                # Get current returns and calculate mean reversion signal
                hist_returns = returns[t-window_size:t]
                mean_returns = np.mean(hist_returns, axis=0)
                curr_returns = returns[t]
                
                # Calculate group-relative price predictions
                price_prediction = np.zeros(n_assets)
                for g in range(n_total_groups):
                    group_mask = (group_assignments == g)
                    if np.any(group_mask):
                        group_mean = np.mean(mean_returns[group_mask])
                        group_curr = np.mean(curr_returns[group_mask])
                        # Relative mean reversion within group
                        price_prediction[group_mask] = (mean_returns[group_mask] - curr_returns[group_mask]) * \
                                                    (1 + (group_mean - group_curr))
                
                # Ensure numerical stability
                if np.isnan(price_prediction).any() or np.isinf(price_prediction).any():
                    portfolio_return = np.sum(weights * curr_returns)
                    portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                    weight_history.append(weights.copy())
                    portfolio_returns.append(portfolio_return)
                    continue
                
                # Calculate loss with group relativity
                group_weights = np.array([np.sum(weights[group_assignments == g]) for g in range(n_total_groups)])
                group_pred = np.array([np.mean(price_prediction[group_assignments == g]) for g in range(n_total_groups)])
                
                # Combined loss from individual and group predictions
                individual_loss = max(0, epsilon - np.dot(weights, price_prediction))
                group_loss = max(0, epsilon - np.dot(group_weights, group_pred))
                loss = 0.7 * individual_loss + 0.3 * group_loss  # Weighted combination
                
                if loss > 0:
                    # Calculate confidence
                    var = np.dot(np.dot(price_prediction, sigma), price_prediction)
                    
                    if var < 1e-10:
                        portfolio_return = np.sum(weights * curr_returns)
                        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                        weight_history.append(weights.copy())
                        portfolio_returns.append(portfolio_return)
                        continue
                    
                    # Calculate lambda with group-aware scaling
                    base_lambda = np.clip(loss / var, 0, 1e6)
                    group_scales = np.ones(n_assets)  # Default scaling of 1
                    
                    # Apply group-specific scaling if we have multiple groups
                    if n_total_groups > 1:
                        for g in range(n_total_groups):
                            group_mask = (group_assignments == g)
                            group_scales[group_mask] = 1 + 0.2 * (g / (n_total_groups - 1))  # Higher groups get larger updates
                    
                    # Update weights with group-relative scaling
                    lambda_value = base_lambda * group_scales
                    new_weights = weights + np.mean(lambda_value) * np.dot(sigma, price_prediction)
                    
                    # Project weights onto simplex
                    new_weights = np.maximum(new_weights, 0)
                    weight_sum = np.sum(new_weights)
                    if weight_sum > 0:
                        new_weights = new_weights / weight_sum
                    else:
                        new_weights = np.ones(n_assets) / n_assets
                    
                    # Update covariance matrix with group awareness
                    sigma_price = np.dot(sigma, price_prediction)
                    denominator = 1.0 / np.mean(lambda_value) + np.dot(price_prediction, sigma_price)
                    if denominator > 1e-10:
                        sigma = sigma - np.outer(sigma_price, sigma_price) / denominator
                        
                        # Ensure positive definiteness
                        eigvals, eigvecs = np.linalg.eigh(sigma)
                        eigvals = np.maximum(eigvals, 1e-8)
                        sigma = eigvecs.dot(np.diag(eigvals)).dot(eigvecs.T)
                    
                    # Calculate transaction costs
                    turnover = np.sum(np.abs(new_weights - weights))
                    transaction_cost_value = transaction_cost * turnover
                    
                    # Update portfolio value
                    portfolio_return = np.sum(weights * curr_returns) * (1 - transaction_cost_value)
                    portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                    
                    # Store updated weights
                    weights = new_weights.copy()
                    
                else:
                    # No update needed
                    portfolio_return = np.sum(weights * curr_returns)
                    portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                
                weight_history.append(weights.copy())
                portfolio_returns.append(portfolio_return)
            
            # Calculate metrics
            portfolio_returns = np.array(portfolio_returns)
            sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-10) * np.sqrt(252)
            max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values)
            
            return {
                'strategy': 'GRPO-CWMR',
                'portfolio_values': portfolio_values,
                'final_value': portfolio_values[-1],
                'total_return': portfolio_values[-1] - 1.0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'weights': weight_history
            }
        
        # 4. RL-enhanced CWMR
        def rl_cwmr_strategy(returns, features, model, window_size=10):
            """
            RL-enhanced CWMR strategy that uses a trained RL model to make portfolio decisions.
            
            Args:
                returns (numpy.ndarray): Returns data [time, n_assets]
                features (numpy.ndarray): Feature data (if available)
                model: Trained RL model
                window_size (int): Lookback window
                
            Returns:
                dict: Strategy results
            """
            from cwmr_env import CWMRTradingEnv
            
            # Convert to numpy if they're pandas DataFrames
            if hasattr(returns, 'values'):
                returns_np = returns.values
            else:
                returns_np = returns
            
            if features is not None and hasattr(features, 'values'):
                features_np = features.values
            else:
                features_np = features
            
            # Create environment for evaluation with same parameters used in training
            env = CWMRTradingEnv(
                returns_data=returns_np, 
                feature_data=features_np,
                window_size=window_size,
                include_feature_history=True  # Match training environment setting
            )
            
            # Run environment with model
            obs, info = env.reset()
            
            print(f"Observation shape: {obs.shape}")
            print(f"Expected shape from model: {model.observation_space.shape}")
            
            # If observation shapes don't match, let's reshape to match model's expectation
            # This can happen if the training environment had different feature dimensions
            if obs.shape != model.observation_space.shape[0]:
                # Create a new environment with the right observation shape
                use_features = (model.observation_space.shape[0] > returns_np.shape[1] * (window_size + 1))
                
                if use_features:
                    # Training used features, but shapes don't match - pad or truncate
                    env = CWMRTradingEnv(
                        returns_data=returns_np,
                        feature_data=features_np,
                        window_size=window_size,
                        include_feature_history=True,
                    )
                    obs, info = env.reset()
                    
                    # If still mismatched, create a dummy adapter
                    if obs.shape != model.observation_space.shape[0]:
                        orig_shape = obs.shape[0]
                        target_shape = model.observation_space.shape[0]
                        
                        if orig_shape < target_shape:
                            # Pad observation
                            print(f"Padding observation from {orig_shape} to {target_shape}")
                            pad_size = target_shape - orig_shape
                            def obs_adapter(orig_obs):
                                return np.pad(orig_obs, (0, pad_size), 'constant')
                        else:
                            # Truncate observation
                            print(f"Truncating observation from {orig_shape} to {target_shape}")
                            def obs_adapter(orig_obs):
                                return orig_obs[:target_shape]
                        
                        # Apply adapter to initial observation
                        obs = obs_adapter(obs)
                else:
                    # Training didn't use features
                    env = CWMRTradingEnv(
                        returns_data=returns_np,
                        feature_data=None,
                        window_size=window_size,
                        include_feature_history=False,
                    )
                    obs, info = env.reset()
            
            # Continue with strategy evaluation
            done = False
            truncated = False
            
            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs_next, _, done, truncated, info = env.step(action)
                
                # Apply obs adapter if needed
                if 'obs_adapter' in locals():
                    obs = obs_adapter(obs_next)
                else:
                    obs = obs_next
            
            # Get performance metrics
            metrics = env.get_performance_metrics()
            
            return {
                'strategy': 'RL-CWMR',
                'portfolio_values': env.portfolio_values,
                'final_value': env.portfolio_values[-1],
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'weights': env.weight_history
            }
        
        # Run strategies
        print("Evaluating baseline strategies...")
        equal_weight_result = equal_weight_strategy(test_returns)
        basic_cwmr_result = basic_cwmr_strategy(
            returns=test_returns,
            window_size=10,
            epsilon=0.01,
            confidence_bound=0.1,
            transaction_cost=0.001
        )
        
        # 3. GRPO-CWMR
        grpo_cwmr_result = grpo_cwmr_strategy(
            returns=test_returns,
            window_size=10,
            epsilon=0.01,
            confidence_bound=0.1,
            transaction_cost=0.001,
            group_size=5,
            n_groups=4
        )
        
        results = {
            'equal_weight': equal_weight_result,
            'basic_cwmr': basic_cwmr_result,
            'grpo_cwmr': grpo_cwmr_result
        }
        
        if model is not None:
            print("Evaluating RL-CWMR strategy...")
            rl_result = rl_cwmr_strategy(test_returns, test_features, model)
            results['rl_cwmr'] = rl_result
        
        # Print comparison
        print("\nStrategy Comparison:")
        for strategy, result in results.items():
            print(f"\n{result['strategy']}:")
            
            # Safely handle total return (could be scalar or array)
            total_return = result['total_return']
            if hasattr(total_return, 'shape') and total_return.size > 1:
                total_return = total_return.mean()
            print(f"  Total Return: {float(total_return):.4f}")
            
            # Safely handle final value (could be scalar or array)  
            final_value = result['final_value']
            if hasattr(final_value, 'shape') and final_value.size > 1:
                final_value = final_value.mean()
            print(f"  Final Portfolio Value: {float(final_value):.2f}")
            
            # Safely handle Sharpe ratio if available
            if 'sharpe_ratio' in result:
                sharpe = result['sharpe_ratio']
                if hasattr(sharpe, 'shape') and sharpe.size > 1:
                    sharpe = sharpe.mean()
                print(f"  Sharpe Ratio: {float(sharpe):.4f}")
            
            # Safely handle max drawdown if available
            if 'max_drawdown' in result:
                drawdown = result['max_drawdown']
                if hasattr(drawdown, 'shape') and drawdown.size > 1:
                    drawdown = drawdown.mean()
                print(f"  Max Drawdown: {float(drawdown):.4f}")
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        for strategy, result in results.items():
            plt.plot(result['portfolio_values'], label=result['strategy'])
        
        plt.title('Strategy Comparison')
        plt.xlabel('Timesteps')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.results_dir / 'strategy_comparison.png')
        plt.close()
        
        return results
    
    def hyperparameter_tuning(self, 
                             train_returns, 
                             val_returns, 
                             train_features=None, 
                             val_features=None):
        """
        Perform hyperparameter tuning for the CWMR-RL model.
        
        Args:
            train_returns (DataFrame): Training returns data
            val_returns (DataFrame): Validation returns data
            train_features (DataFrame): Training feature data
            val_features (DataFrame): Validation feature data
            
        Returns:
            dict: Best hyperparameters
        """
        print("\n5a. Performing hyperparameter tuning...")
        
        # Define parameter grid for grid search
        param_grid = {
            'window_size': [10, 15, 20, 30],
            'transaction_cost': [0.0001, 0.0005, 0.001, 0.002],
            'confidence_bound': [0.05, 0.1, 0.2, 0.5],
            'epsilon': [0.001, 0.005, 0.01, 0.02],
            'learning_rate': [0.0001, 0.0003, 0.001],
            'gamma': [0.95, 0.97, 0.99],
            'reward_scaling': [0.1, 0.5, 1.0, 2.0],
        }
        
        # Intelligently select parameter combinations to test (Latin Hypercube Sampling)
        # This is more efficient than testing all combinations
        from sklearn.model_selection import ParameterSampler
        import scipy.stats as stats
        
        param_distributions = {
            'window_size': stats.randint(5, 40),
            'transaction_cost': stats.loguniform(0.0001, 0.01),
            'confidence_bound': stats.loguniform(0.01, 1.0),
            'epsilon': stats.loguniform(0.001, 0.1),
            'learning_rate': stats.loguniform(0.00003, 0.01),
            'gamma': stats.uniform(0.9, 0.099),  # 0.9 to 0.999
            'reward_scaling': stats.loguniform(0.1, 10.0),
        }
        
        n_trials = 10  # Number of parameter combinations to try
        random_state = 42  # For reproducibility
        
        # Sample parameter combinations
        param_list = list(ParameterSampler(
            param_distributions, n_iter=n_trials, random_state=random_state
        ))
        
        # If n_trials is very small, add some sensible defaults
        if n_trials <= 5:
            param_list.extend([
                {'window_size': 10, 'transaction_cost': 0.001, 'confidence_bound': 0.1, 
                 'epsilon': 0.01, 'learning_rate': 0.0003, 'gamma': 0.99, 'reward_scaling': 1.0},
                {'window_size': 20, 'transaction_cost': 0.0005, 'confidence_bound': 0.05, 
                 'epsilon': 0.005, 'learning_rate': 0.0001, 'gamma': 0.97, 'reward_scaling': 0.5}
            ])
        
        best_reward = -np.inf
        best_params = None
        
        # For each trial, train a model with different hyperparameters
        for i, params in enumerate(param_list):
            print(f"\nTrial {i+1}/{len(param_list)}")
            print(f"Parameters: {params}")
            
            # Integer parameters need to be converted from numpy types
            if 'window_size' in params:
                params['window_size'] = int(params['window_size'])
            
            # Prepare environment with current parameters
            train_env, eval_env = self.prepare_environment(
                train_returns=train_returns,
                train_features=train_features,
                val_returns=val_returns,
                val_features=val_features,
                window_size=params['window_size'],
                transaction_cost=params['transaction_cost'],
                confidence_bound=params['confidence_bound'],
                epsilon=params['epsilon'],
                reward_scaling=params.get('reward_scaling', 1.0),
            )
            
            # Create and train model with limited steps for faster evaluation
            model = PPO(
                policy='MlpPolicy',
                env=train_env,
                learning_rate=params['learning_rate'],
                gamma=params.get('gamma', 0.99),
                verbose=0
            )
            
            # Train for a limited number of steps to quickly evaluate
            tuning_steps = 20000
            model.learn(total_timesteps=tuning_steps)
            
            # Evaluate model on validation environment
            total_reward = 0
            n_eval_episodes = 3
            
            for _ in range(n_eval_episodes):
                obs, info = eval_env.reset()
                done = False
                truncated = False
                episode_reward = 0
                
                while not done and not truncated:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = eval_env.step(action)
                    episode_reward += reward
                
                total_reward += episode_reward
            
            mean_reward = total_reward / n_eval_episodes
            
            # Calculate additional metrics
            metrics = eval_env.get_performance_metrics()
            sharpe = metrics['sharpe_ratio']
            max_dd = metrics['max_drawdown']
            
            # Combined score weighting return, Sharpe, and drawdown
            # Higher Sharpe and returns are better, lower drawdown is better
            combined_score = mean_reward + 0.5 * sharpe - 0.3 * max_dd
            
            print(f"Mean Reward: {mean_reward:.4f}")
            print(f"Sharpe Ratio: {sharpe:.4f}")
            print(f"Max Drawdown: {max_dd:.4f}")
            print(f"Combined Score: {combined_score:.4f}")
            
            if combined_score > best_reward:
                best_reward = combined_score
                best_params = params
                print("New best parameters found!")
        
        print("\nBest parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        return best_params 