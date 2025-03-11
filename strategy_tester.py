import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import time
import os
from tqdm import tqdm

# Import our modules
from cwmr_env import CWMRTradingEnv
from rl_trainer import RLTrainer
from data_loader import DataLoader

class StrategyTester:
    """
    A comprehensive tester for evaluating different portfolio strategies including:
    - Traditional CWMR with different parameters
    - RL-enhanced CWMR with PPO
    - Multi-agent CWMR ensemble
    - GRPO-CWMR (Group Relativity Policy Optimization)
    """
    
    def __init__(self, 
                 data_dir,
                 results_dir='results/strategy_test',
                 max_stocks=20,
                 start_date=None,
                 end_date=None,
                 window_size=10):
        """
        Initialize the strategy tester.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing stock data in CSV format
        results_dir : str
            Directory to save test results
        max_stocks : int
            Maximum number of stocks to include
        start_date : str
            Start date for the test period (YYYY-MM-DD)
        end_date : str
            End date for the test period (YYYY-MM-DD)
        window_size : int
            Lookback window size for strategies
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.max_stocks = max_stocks
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = DataLoader(
            data_dir=data_dir,
            max_stocks=max_stocks,
            start_date=start_date,
            end_date=end_date
        )
        
        # Load and prepare data
        self.data_loader.load_stock_files()
        self.data_loader.compute_returns()
        self.returns_data = self.data_loader.returns_data.values
        
        # Create mean reversion features
        self.data_loader.create_mean_reversion_features()
        self.feature_data = self.data_loader.features.values if hasattr(self.data_loader, 'features') else None
        
        # Split data
        self.train_returns, self.val_returns, self.test_returns = self.data_loader.split_data()
        if self.feature_data is not None:
            train_idx = self.train_returns.index
            val_idx = self.val_returns.index
            test_idx = self.test_returns.index
            
            self.train_features = self.data_loader.features.loc[train_idx].values
            self.val_features = self.data_loader.features.loc[val_idx].values
            self.test_features = self.data_loader.features.loc[test_idx].values
        else:
            self.train_features = None
            self.val_features = None
            self.test_features = None
        
        # Initialize RL trainer
        self.rl_trainer = RLTrainer(
            log_dir=os.path.join(self.results_dir, 'logs'),
            save_dir=os.path.join(self.results_dir, 'models'),
            results_dir=os.path.join(self.results_dir, 'results')
        )
        
        # Store results
        self.results = {}
    
    def test_traditional_cwmr(self, 
                              confidence_bounds=[0.05, 0.1, 0.2], 
                              epsilons=[0.001, 0.01, 0.1]):
        """
        Test traditional CWMR with different parameter settings.
        
        Parameters:
        -----------
        confidence_bounds : list
            List of confidence bound values to test
        epsilons : list
            List of epsilon values to test
        
        Returns:
        --------
        dict
            Dictionary of performance metrics for each parameter setting
        """
        print("Testing traditional CWMR strategies...")
        results = {}
        
        for cb in confidence_bounds:
            for eps in epsilons:
                strategy_name = f"CWMR_cb{cb}_eps{eps}"
                print(f"Testing {strategy_name}...")
                
                # Create environment
                # Convert to numpy array if we have a DataFrame
                test_returns_np = self.test_returns.values if hasattr(self.test_returns, 'values') else self.test_returns
                test_features_np = self.test_features.values if hasattr(self.test_features, 'values') else self.test_features
                
                env = CWMRTradingEnv(
                    returns_data=test_returns_np,
                    feature_data=test_features_np,
                    window_size=self.window_size,
                    confidence_bound=cb,
                    epsilon=eps
                )
                
                # Run CWMR strategy (without RL)
                portfolio_values = []
                weights_history = []
                
                obs = env.reset()
                done = False
                
                while not done:
                    # For traditional CWMR, we don't use external action - we use the internal CWMR update
                    # We pass a neutral action (0.5 for all assets)
                    action = np.ones(self.max_stocks) / self.max_stocks
                    obs, reward, done, truncated, info = env.step(action)
                    
                    portfolio_values.append(info['portfolio_value'])
                    weights_history.append(info['weights'])
                
                # Calculate performance metrics
                returns = np.array(portfolio_values[1:]) / np.array(portfolio_values[:-1]) - 1
                
                results[strategy_name] = {
                    'final_value': portfolio_values[-1],
                    'sharpe_ratio': env.get_sharpe_ratio(),
                    'max_drawdown': env.get_max_drawdown(),
                    'portfolio_values': portfolio_values,
                    'weights_history': weights_history,
                    'returns': returns,
                    'params': {'confidence_bound': cb, 'epsilon': eps}
                }
                
        self.results['traditional_cwmr'] = results
        return results
    
    def test_rl_cwmr(self, 
                     total_timesteps=50000,
                     transaction_cost=0.001,
                     reward_scaling=1.0):
        """
        Test CWMR enhanced with Reinforcement Learning.
        
        Parameters:
        -----------
        total_timesteps : int
            Number of training steps for the RL agent
        transaction_cost : float
            Transaction cost as fraction of trade value
        reward_scaling : float
            Scaling factor for reward
            
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        print("Testing RL-enhanced CWMR...")
        results = {}
        
        # Convert to numpy arrays if needed
        train_returns_np = self.train_returns.values if hasattr(self.train_returns, 'values') else self.train_returns
        train_features_np = self.train_features.values if hasattr(self.train_features, 'values') else self.train_features
        val_returns_np = self.val_returns.values if hasattr(self.val_returns, 'values') else self.val_returns
        val_features_np = self.val_features.values if hasattr(self.val_features, 'values') else self.val_features
        test_returns_np = self.test_returns.values if hasattr(self.test_returns, 'values') else self.test_returns
        test_features_np = self.test_features.values if hasattr(self.test_features, 'values') else self.test_features
        
        # Prepare environments
        train_env, eval_env = self.rl_trainer.prepare_environment(
            train_returns=train_returns_np,
            train_features=train_features_np,
            val_returns=val_returns_np,
            val_features=val_features_np,
            window_size=self.window_size,
            transaction_cost=transaction_cost,
            confidence_bound=0.1,  # Default value, RL will learn to adjust
            epsilon=0.01,  # Default value, RL will learn to adjust
            reward_scaling=reward_scaling
        )
        
        # Train RL model
        model = self.rl_trainer.train(
            train_env=train_env,
            eval_env=eval_env,
            model_name="ppo_cwmr_test",
            total_timesteps=total_timesteps,
            eval_freq=5000,
            n_eval_episodes=1
        )
        
        # Create test environment
        test_env = CWMRTradingEnv(
            returns_data=test_returns_np,
            feature_data=test_features_np,
            window_size=self.window_size,
            transaction_cost=transaction_cost,
            confidence_bound=0.1,
            epsilon=0.01,
            reward_scaling=reward_scaling
        )
        
        # Evaluate model on test data
        eval_results = self.rl_trainer.evaluate(
            model=model,
            test_env=test_env,
            n_episodes=1,
            deterministic=True
        )
        
        # Store results
        results['rl_cwmr'] = {
            'final_value': eval_results['final_value'],
            'sharpe_ratio': eval_results['sharpe_ratio'],
            'max_drawdown': eval_results['max_drawdown'],
            'portfolio_values': eval_results['portfolio_values'],
            'weights_history': eval_results['weights_history'],
            'returns': eval_results['returns']
        }
        
        self.results['rl_cwmr'] = results
        return results
    
    def test_cwmr_ensemble(self, num_agents=5):
        """
        Test an ensemble of CWMR agents with different parameters.
        The ensemble uses equal weight allocation among agents.
        
        Parameters:
        -----------
        num_agents : int
            Number of CWMR agents in the ensemble
            
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        print(f"Testing CWMR ensemble with {num_agents} agents...")
        results = {}
        
        # Convert to numpy arrays if needed
        test_returns_np = self.test_returns.values if hasattr(self.test_returns, 'values') else self.test_returns
        test_features_np = self.test_features.values if hasattr(self.test_features, 'values') else self.test_features
        
        # Generate parameter combinations for different agents
        # Use a wider range of parameters for diverse agent behavior
        confidence_bounds = np.linspace(0.05, 0.3, num_agents)  # Increased upper bound
        epsilons = np.linspace(0.001, 0.1, num_agents)
        
        print("Agent parameters:")
        for i in range(num_agents):
            print(f"  Agent {i+1}: confidence_bound={confidence_bounds[i]:.3f}, epsilon={epsilons[i]:.4f}")
        
        # Initialize agents with different parameters
        agents = []
        for i in range(num_agents):
            env = CWMRTradingEnv(
                returns_data=test_returns_np,
                feature_data=test_features_np,
                window_size=self.window_size,
                confidence_bound=confidence_bounds[i],
                epsilon=epsilons[i]
            )
            agents.append({
                'env': env,
                'params': {
                    'confidence_bound': confidence_bounds[i],
                    'epsilon': epsilons[i]
                }
            })
        
        # Run ensemble strategy
        portfolio_values = [1.0]
        weights_history = []
        agent_portfolio_values = [[] for _ in range(num_agents)]
        agent_weights_history = [[] for _ in range(num_agents)]
        
        # Reset environments
        observations = []
        for agent in agents:
            obs = agent['env'].reset()
            observations.append(obs)
        
        # Simulation loop
        done = False
        t = 0
        print(f"Running ensemble simulation for {len(test_returns_np)} timesteps...")
        
        try:
            with tqdm(total=len(test_returns_np), desc="Ensemble Progress") as pbar:
                while not done and t < len(test_returns_np):
                    ensemble_weights = np.zeros(self.max_stocks)
                    agent_weights = []
                    
                    # Get weights from each agent
                    for i, agent in enumerate(agents):
                        action = np.ones(self.max_stocks) / self.max_stocks  # Neutral action
                        obs, reward, done, truncated, info = agent['env'].step(action)
                        observations[i] = obs
                        
                        weights = info['weights']
                        agent_weights.append(weights)
                        agent_portfolio_values[i].append(info['portfolio_value'])
                        agent_weights_history[i].append(weights.copy())
                        
                        # Contribute equally to ensemble weights
                        ensemble_weights += weights / num_agents
                    
                    weights_history.append(ensemble_weights.copy())
                    
                    # Calculate portfolio return based on ensemble weights
                    next_return = np.dot(ensemble_weights, test_returns_np[t])
                    current_portfolio_value = portfolio_values[-1] * (1 + next_return)
                    portfolio_values.append(current_portfolio_value)
                    
                    # Update progress bar with current portfolio value
                    pbar.set_postfix({"Portfolio Value": f"{current_portfolio_value:.4f}"})
                    pbar.update(1)
                    
                    t += 1
        
        except Exception as e:
            print(f"Error in ensemble simulation: {str(e)}")
            print(f"Completed {t} of {len(test_returns_np)} timesteps before error")
            if len(portfolio_values) <= 1:
                print("No valid portfolio values generated. Cannot continue.")
                return {'cwmr_ensemble': {
                    'final_value': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 1.0,
                    'portfolio_values': [1.0, 0.0],
                    'weights_history': [],
                    'returns': np.array([-1.0]),
                    'error': str(e)
                }}
        
        # Calculate performance metrics
        returns = np.array(portfolio_values[1:]) / np.array(portfolio_values[:-1]) - 1
        
        # Handle potential NaN values in returns
        valid_returns = returns[~np.isnan(returns)]
        if len(valid_returns) == 0:
            print("Warning: All returns are NaN. Using zeros instead.")
            valid_returns = np.zeros_like(returns)
        
        sharpe_ratio = np.mean(valid_returns) / (np.std(valid_returns) + 1e-10) * np.sqrt(252)  # Annualized, avoid div by zero
        
        # Calculate max drawdown
        cumulative_returns = np.cumprod(1 + valid_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / (peak + 1e-10)  # Avoid div by zero
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 1.0
        
        print(f"Ensemble Results:")
        print(f"  Final Portfolio Value: {portfolio_values[-1]:.4f}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"  Max Drawdown: {max_drawdown:.4f}")
        
        # Compare individual agent performance
        print("Individual Agent Performance:")
        for i in range(num_agents):
            if len(agent_portfolio_values[i]) > 0:
                agent_final_value = agent_portfolio_values[i][-1]
                print(f"  Agent {i+1}: Final Value = {agent_final_value:.4f}")
        
        # Plot ensemble vs individual agents
        plt.figure(figsize=(12, 8))
        for i in range(num_agents):
            if len(agent_portfolio_values[i]) > 0:
                plt.plot(agent_portfolio_values[i], 
                         alpha=0.5, 
                         label=f"Agent {i+1} (CB={confidence_bounds[i]:.3f}, ε={epsilons[i]:.4f})")
        
        plt.plot(portfolio_values, linewidth=2, color='black', label="Ensemble")
        plt.title('Multi-Agent CWMR Ensemble vs Individual Agents')
        plt.xlabel('Timestep')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create figure directory if it doesn't exist
        fig_dir = os.path.join(self.results_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, 'ensemble_comparison.png'))
        plt.close()
        
        # Store results
        results['cwmr_ensemble'] = {
            'final_value': portfolio_values[-1],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'weights_history': weights_history,
            'returns': valid_returns,
            'agent_portfolio_values': agent_portfolio_values,
            'agent_weights_history': agent_weights_history,
            'agent_params': [agent['params'] for agent in agents]
        }
        
        self.results['cwmr_ensemble'] = results
        
        # Save detailed ensemble results
        ensemble_data = {
            'portfolio_values': portfolio_values,
            'weights_history': weights_history,
            'agent_portfolio_values': agent_portfolio_values,
            'agent_params': [{'confidence_bound': agent['params']['confidence_bound'], 
                              'epsilon': agent['params']['epsilon']} 
                             for agent in agents]
        }
        
        # Create a DataFrame with agent performances
        agent_perf_df = pd.DataFrame({
            'agent_id': [f"Agent {i+1}" for i in range(num_agents)],
            'confidence_bound': [agent['params']['confidence_bound'] for agent in agents],
            'epsilon': [agent['params']['epsilon'] for agent in agents],
            'final_value': [agent_portfolio_values[i][-1] if len(agent_portfolio_values[i]) > 0 else 0 
                           for i in range(num_agents)]
        })
        
        # Add ensemble performance
        agent_perf_df = pd.concat([
            agent_perf_df,
            pd.DataFrame({
                'agent_id': ['Ensemble'],
                'confidence_bound': [np.nan],
                'epsilon': [np.nan],
                'final_value': [portfolio_values[-1]]
            })
        ])
        
        # Save agent performance comparison
        os.makedirs(os.path.join(self.results_dir, 'data'), exist_ok=True)
        agent_perf_df.to_csv(os.path.join(self.results_dir, 'data', 'agent_performance.csv'), index=False)
        
        # Plot agent parameter vs performance
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(agent_perf_df['confidence_bound'][:-1], agent_perf_df['final_value'][:-1])
        plt.xlabel('Confidence Bound')
        plt.ylabel('Final Portfolio Value')
        plt.grid(True, alpha=0.3)
        plt.title('Effect of Confidence Bound on Performance')
        
        plt.subplot(1, 2, 2)
        plt.scatter(agent_perf_df['epsilon'][:-1], agent_perf_df['final_value'][:-1])
        plt.xlabel('Epsilon')
        plt.ylabel('Final Portfolio Value')
        plt.grid(True, alpha=0.3)
        plt.title('Effect of Epsilon on Performance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'agent_parameter_analysis.png'))
        plt.close()
        
        return results
    
    def test_rl_cwmr_dynamic_params(self, 
                                   total_timesteps=50000,
                                   transaction_cost=0.001,
                                   reward_scaling=1.0):
        """
        Test RL-CWMR with dynamic parameter adjustment.
        The RL agent learns to adjust confidence_bound and epsilon parameters.
        
        Parameters:
        -----------
        total_timesteps : int
            Number of training steps for the RL agent
        transaction_cost : float
            Transaction cost as fraction of trade value
        reward_scaling : float
            Scaling factor for reward
            
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        print("Testing RL-CWMR with dynamic parameter adjustment...")
        results = {}
        
        # Convert to numpy arrays if needed
        train_returns_np = self.train_returns.values if hasattr(self.train_returns, 'values') else self.train_returns
        train_features_np = self.train_features.values if hasattr(self.train_features, 'values') else self.train_features
        val_returns_np = self.val_returns.values if hasattr(self.val_returns, 'values') else self.val_returns
        val_features_np = self.val_features.values if hasattr(self.val_features, 'values') else self.val_features
        test_returns_np = self.test_returns.values if hasattr(self.test_returns, 'values') else self.test_returns
        test_features_np = self.test_features.values if hasattr(self.test_features, 'values') else self.test_features
        
        # This would require a modified environment where the action space includes
        # parameters adjustment. For this implementation, we'll create a wrapper
        # around our existing environment.
        
        class DynamicParamCWMREnv(CWMRTradingEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
                # Modify action space to include parameter adjustments
                # First part of action is portfolio weights, second part is parameters
                self.action_space_weights = self.action_space
                
                # Additional action dimensions for confidence_bound and epsilon
                # We'll map these to reasonable ranges
                self.has_dynamic_params = True
                
            def step(self, action):
                # Split action into portfolio weights and parameter adjustments
                if isinstance(action, np.ndarray) and len(action) > self.returns_data.shape[1]:
                    weights_action = action[:self.returns_data.shape[1]]
                    param_action = action[self.returns_data.shape[1]:]
                    
                    # Map param_action to actual parameter values
                    # param_action[0] -> confidence_bound in [0.01, 0.5]
                    # param_action[1] -> epsilon in [0.001, 0.1]
                    self.confidence_bound = 0.01 + param_action[0] * 0.49
                    self.epsilon = 0.001 + param_action[1] * 0.099
                else:
                    weights_action = action
                
                return super().step(weights_action)
        
        # Create environments with dynamic parameters
        train_env_dynamic = DynamicParamCWMREnv(
            returns_data=train_returns_np,
            feature_data=train_features_np,
            window_size=self.window_size,
            transaction_cost=transaction_cost,
            confidence_bound=0.1,  # Initial value
            epsilon=0.01,  # Initial value
            reward_scaling=reward_scaling
        )
        
        val_env_dynamic = DynamicParamCWMREnv(
            returns_data=val_returns_np,
            feature_data=val_features_np,
            window_size=self.window_size,
            transaction_cost=transaction_cost,
            confidence_bound=0.1,  # Initial value
            epsilon=0.01,  # Initial value
            reward_scaling=reward_scaling
        )
        
        # Create test environment
        test_env_dynamic = DynamicParamCWMREnv(
            returns_data=test_returns_np,
            feature_data=test_features_np,
            window_size=self.window_size,
            transaction_cost=transaction_cost,
            confidence_bound=0.1,  # Initial value
            epsilon=0.01,  # Initial value
            reward_scaling=reward_scaling
        )
        
        # Create PPO model with custom policy network architecture
        # that outputs both portfolio weights and parameter adjustments
        from stable_baselines3.common.policies import ActorCriticPolicy
        
        class CWMRDynamicParamPolicy(ActorCriticPolicy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Customize the network architecture if needed
        
        # Train the model
        model = PPO(
            policy=CWMRDynamicParamPolicy,
            env=train_env_dynamic,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1
        )
        
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate on test data
        obs = test_env_dynamic.reset()
        done = False
        portfolio_values = [1.0]
        weights_history = []
        param_history = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Split action
            weights_action = action[:self.max_stocks]
            param_action = action[self.max_stocks:] if len(action) > self.max_stocks else None
            
            obs, reward, done, truncated, info = test_env_dynamic.step(action)
            
            portfolio_values.append(info['portfolio_value'])
            weights_history.append(info['weights'])
            
            if param_action is not None:
                param_history.append({
                    'confidence_bound': test_env_dynamic.confidence_bound,
                    'epsilon': test_env_dynamic.epsilon
                })
        
        # Calculate performance metrics
        returns = np.array(portfolio_values[1:]) / np.array(portfolio_values[:-1]) - 1
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        
        # Calculate max drawdown
        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = np.max(drawdown)
        
        results['rl_cwmr_dynamic'] = {
            'final_value': portfolio_values[-1],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'weights_history': weights_history,
            'param_history': param_history,
            'returns': returns
        }
        
        self.results['rl_cwmr_dynamic'] = results
        return results
    
    def test_grpo_cwmr(self, 
                       group_size=5, 
                       n_groups=4,
                       transaction_cost=0.001):
        """
        Test GRPO-CWMR (Group Relativity Policy Optimization).
        
        Parameters:
        -----------
        group_size : int
            Size of each stock group
        n_groups : int
            Number of groups to use
        transaction_cost : float
            Transaction cost as fraction of trade value
            
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        print("Testing GRPO-CWMR...")
        
        # Convert to numpy arrays if needed
        test_returns_np = self.test_returns.values if hasattr(self.test_returns, 'values') else self.test_returns
        test_features_np = self.test_features.values if hasattr(self.test_features, 'values') else self.test_features
        
        # Use the existing implementation in rl_trainer.py
        results = self.rl_trainer.compare_with_baseline(
            test_returns=test_returns_np,
            test_features=test_features_np,
            model=None  # We don't need an RL model for this test
        )
        
        # Extract GRPO-CWMR results from the baseline comparison
        grpo_results = {
            'final_value': results['grpo_cwmr']['final_value'],
            'sharpe_ratio': results['grpo_cwmr']['sharpe_ratio'],
            'max_drawdown': results['grpo_cwmr']['max_drawdown'],
            'portfolio_values': results['grpo_cwmr']['portfolio_values'],
            'returns': np.diff(results['grpo_cwmr']['portfolio_values']) / results['grpo_cwmr']['portfolio_values'][:-1]
        }
        
        self.results['grpo_cwmr'] = {'grpo_cwmr': grpo_results}
        return self.results['grpo_cwmr']
    
    def compare_all_strategies(self):
        """
        Run all tests and compare the strategies.
        
        Returns:
        --------
        dict
            Complete results dictionary
        """
        print("Running comprehensive strategy comparison...")
        
        # Prepare summary
        summary = {
            'strategy': [],
            'final_value': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'annualized_return': []
        }
        
        # Traditional CWMR
        if 'traditional_cwmr' in self.results:
            for name, result in self.results['traditional_cwmr'].items():
                summary['strategy'].append(name)
                summary['final_value'].append(result['final_value'])
                summary['sharpe_ratio'].append(result['sharpe_ratio'])
                summary['max_drawdown'].append(result['max_drawdown'])
                
                # Calculate annualized return
                total_return = result['final_value'] - 1.0
                years = len(self.test_returns) / 252  # Assuming 252 trading days per year
                ann_return = (1 + total_return) ** (1 / years) - 1
                summary['annualized_return'].append(ann_return)
        
        # RL-CWMR
        if 'rl_cwmr' in self.results:
            summary['strategy'].append('RL-CWMR')
            result = self.results['rl_cwmr']['rl_cwmr']
            summary['final_value'].append(result['final_value'])
            summary['sharpe_ratio'].append(result['sharpe_ratio'])
            summary['max_drawdown'].append(result['max_drawdown'])
            
            total_return = result['final_value'] - 1.0
            years = len(self.test_returns) / 252
            ann_return = (1 + total_return) ** (1 / years) - 1
            summary['annualized_return'].append(ann_return)
        
        # CWMR Ensemble
        if 'cwmr_ensemble' in self.results:
            summary['strategy'].append('CWMR-Ensemble')
            result = self.results['cwmr_ensemble']['cwmr_ensemble']
            summary['final_value'].append(result['final_value'])
            summary['sharpe_ratio'].append(result['sharpe_ratio'])
            summary['max_drawdown'].append(result['max_drawdown'])
            
            total_return = result['final_value'] - 1.0
            years = len(self.test_returns) / 252
            ann_return = (1 + total_return) ** (1 / years) - 1
            summary['annualized_return'].append(ann_return)
        
        # RL-CWMR with Dynamic Parameters
        if 'rl_cwmr_dynamic' in self.results:
            summary['strategy'].append('RL-CWMR-Dynamic')
            result = self.results['rl_cwmr_dynamic']['rl_cwmr_dynamic']
            summary['final_value'].append(result['final_value'])
            summary['sharpe_ratio'].append(result['sharpe_ratio'])
            summary['max_drawdown'].append(result['max_drawdown'])
            
            total_return = result['final_value'] - 1.0
            years = len(self.test_returns) / 252
            ann_return = (1 + total_return) ** (1 / years) - 1
            summary['annualized_return'].append(ann_return)
        
        # GRPO-CWMR
        if 'grpo_cwmr' in self.results:
            summary['strategy'].append('GRPO-CWMR')
            result = self.results['grpo_cwmr']['grpo_cwmr']
            summary['final_value'].append(result['final_value'])
            summary['sharpe_ratio'].append(result['sharpe_ratio'])
            summary['max_drawdown'].append(result['max_drawdown'])
            
            total_return = result['final_value'] - 1.0
            years = len(self.test_returns) / 252
            ann_return = (1 + total_return) ** (1 / years) - 1
            summary['annualized_return'].append(ann_return)
        
        # Create summary dataframe
        summary_df = pd.DataFrame(summary)
        
        # Sort by final value
        summary_df = summary_df.sort_values('final_value', ascending=False)
        
        # Plot results
        self._plot_results(summary_df)
        
        # Save summary
        summary_df.to_csv(os.path.join(self.results_dir, 'strategy_comparison.csv'), index=False)
        
        print("\nStrategy Comparison Summary:")
        print(summary_df)
        
        return summary_df
    
    def _plot_results(self, summary_df):
        """
        Plot the comparison results.
        
        Parameters:
        -----------
        summary_df : pandas.DataFrame
            DataFrame containing strategy comparison summary
        """
        # Create figure directory
        fig_dir = os.path.join(self.results_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        # Ensure all values are scalar (not arrays)
        for col in ['final_value', 'sharpe_ratio', 'max_drawdown', 'annualized_return']:
            if col in summary_df:
                # Convert any arrays to their mean value
                summary_df[col] = summary_df[col].apply(
                    lambda x: float(x.mean()) if hasattr(x, 'mean') else float(x) if not pd.isna(x) else 0.0
                )
        
        # Plot final values
        plt.figure(figsize=(12, 8))
        bars = plt.bar(summary_df['strategy'], summary_df['final_value'])
        plt.title('Final Portfolio Value by Strategy')
        plt.xlabel('Strategy')
        plt.ylabel('Final Portfolio Value')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'final_values.png'))
        
        # Plot Sharpe ratios if available
        if 'sharpe_ratio' in summary_df.columns:
            plt.figure(figsize=(12, 8))
            bars = plt.bar(summary_df['strategy'], summary_df['sharpe_ratio'])
            plt.title('Sharpe Ratio by Strategy')
            plt.xlabel('Strategy')
            plt.ylabel('Sharpe Ratio')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'sharpe_ratios.png'))
        
        # Plot max drawdowns if available
        if 'max_drawdown' in summary_df.columns:
            plt.figure(figsize=(12, 8))
            bars = plt.bar(summary_df['strategy'], summary_df['max_drawdown'])
            plt.title('Maximum Drawdown by Strategy')
            plt.xlabel('Strategy')
            plt.ylabel('Maximum Drawdown')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'max_drawdowns.png'))
        
        # Plot portfolio value over time for best strategies
        plt.figure(figsize=(15, 10))
        
        # Get top 5 strategies by final value (or fewer if we have less than 5)
        n_top = min(5, len(summary_df))
        top_strategies = summary_df.head(n_top)['strategy'].tolist()
        
        for strategy in top_strategies:
            try:
                if strategy.startswith('CWMR_cb'):
                    # Extract parameters
                    parts = strategy.split('_')
                    cb = float(parts[1][2:])
                    eps = float(parts[2][3:])
                    
                    portfolio_values = self.results['traditional_cwmr'][strategy]['portfolio_values']
                    plt.plot(portfolio_values, label=strategy)
                elif strategy == 'RL-CWMR':
                    portfolio_values = self.results['rl_cwmr']['rl_cwmr']['portfolio_values']
                    plt.plot(portfolio_values, label=strategy)
                elif strategy == 'CWMR-Ensemble':
                    portfolio_values = self.results['cwmr_ensemble']['cwmr_ensemble']['portfolio_values']
                    plt.plot(portfolio_values, label=strategy)
                elif strategy == 'RL-CWMR-Dynamic':
                    portfolio_values = self.results['rl_cwmr_dynamic']['rl_cwmr_dynamic']['portfolio_values']
                    plt.plot(portfolio_values, label=strategy)
                elif strategy == 'GRPO-CWMR':
                    portfolio_values = self.results['grpo_cwmr']['grpo_cwmr']['portfolio_values']
                    plt.plot(portfolio_values, label=strategy)
            except (KeyError, TypeError) as e:
                print(f"Warning: Could not plot portfolio values for strategy {strategy}: {str(e)}")
                continue
        
        plt.title('Portfolio Value Over Time (Top Strategies)')
        plt.xlabel('Trading Day')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'portfolio_value_time_series.png'))
        
        # If RL-CWMR-Dynamic was used, plot the parameter evolution
        if 'RL-CWMR-Dynamic' in top_strategies and 'rl_cwmr_dynamic' in self.results:
            param_history = self.results['rl_cwmr_dynamic']['rl_cwmr_dynamic']['param_history']
            
            if param_history and len(param_history) > 0:
                try:
                    # Extract parameters
                    confidence_bounds = [p['confidence_bound'] for p in param_history]
                    epsilons = [p['epsilon'] for p in param_history]
                    
                    # Plot confidence bounds
                    plt.figure(figsize=(15, 6))
                    plt.plot(confidence_bounds)
                    plt.title('Evolution of Confidence Bound Parameter')
                    plt.xlabel('Trading Day')
                    plt.ylabel('Confidence Bound')
                    plt.grid(linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(fig_dir, 'confidence_bound_evolution.png'))
                    
                    # Plot epsilons
                    plt.figure(figsize=(15, 6))
                    plt.plot(epsilons)
                    plt.title('Evolution of Epsilon Parameter')
                    plt.xlabel('Trading Day')
                    plt.ylabel('Epsilon')
                    plt.grid(linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(fig_dir, 'epsilon_evolution.png'))
                except Exception as e:
                    print(f"Warning: Could not plot parameter evolution: {str(e)}")
        
        plt.close('all')  # Close all open figures

if __name__ == "__main__":
    # Example usage
    tester = StrategyTester(
        data_dir="/path/to/stock/data",
        max_stocks=20,
        start_date="2010-01-01",
        end_date="2020-12-31"
    )
    
    # Run all tests and generate comparison
    summary = tester.compare_all_strategies()
