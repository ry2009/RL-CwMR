import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CWMRTradingEnv(gym.Env):
    """
    A CWMR (Confidence-Weighted Mean Reversion) trading environment 
    enhanced with RL capabilities for adaptive portfolio optimization.
    
    This environment allows an RL agent to make portfolio adjustments
    based on mean-reversion signals. The CWMR strategy assumes prices
    tend to revert to their mean, and RL is used to optimize the
    confidence levels and reallocation decisions.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 returns_data, 
                 feature_data=None,
                 window_size=10, 
                 initial_portfolio_value=1e6,
                 transaction_cost=0.001,
                 reward_scaling=1.0,
                 confidence_bound=0.1,
                 epsilon=0.01,
                 include_feature_history=False):
        """
        Initialize the CWMR trading environment.
        
        Args:
            returns_data (numpy.ndarray): Asset returns data [time, n_assets]
            feature_data (numpy.ndarray, optional): Additional feature data [time, n_features]
            window_size (int): Size of the observation window
            initial_portfolio_value (float): Initial portfolio value
            transaction_cost (float): Transaction cost as a fraction
            reward_scaling (float): Scaling factor for rewards
            confidence_bound (float): Initial confidence bound for CWMR
            epsilon (float): Target mean reversion threshold
            include_feature_history (bool): Whether to include feature history in state
        """
        super(CWMRTradingEnv, self).__init__()
        
        self.returns_data = returns_data
        self.feature_data = feature_data
        self.window_size = window_size
        self.initial_portfolio_value = initial_portfolio_value
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.confidence_bound = confidence_bound
        self.epsilon = epsilon
        self.include_feature_history = include_feature_history
        
        # Environment dimensions
        self.n_assets = returns_data.shape[1]
        self.n_timesteps = returns_data.shape[0]
        
        # Features dimensions (if provided)
        self.n_features = 0
        if feature_data is not None:
            self.n_features = feature_data.shape[1]
        
        # Action space: Portfolio weight adjustments [-1, 1] for each asset
        # These are used to scale the CWMR updates dynamically
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Observation space includes:
        # 1. Historical returns for each asset
        # 2. Current portfolio weights
        # 3. Past confidence levels
        # 4. Optional feature data
        
        # Calculate observation dimensions
        obs_dim = self.n_assets * window_size  # Historical returns
        obs_dim += self.n_assets               # Current portfolio weights
        obs_dim += 1                           # Current confidence level
        
        if self.include_feature_history and self.feature_data is not None:
            # Add feature dimensions if using feature history
            obs_dim += self.n_features * window_size
        elif self.feature_data is not None:
            # Otherwise just the current features
            obs_dim += self.n_features
            
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        
        Returns:
            numpy.ndarray: Initial observation
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Start after the window size
        self.current_step = self.window_size
        
        # Initialize portfolio value
        self.portfolio_value = self.initial_portfolio_value
        
        # Initialize portfolio weights (equal weights)
        self.weights = np.ones(self.n_assets) / self.n_assets
        
        # Initialize CWMR parameters
        # Mean vector (current portfolio weights)
        self.mean = self.weights.copy()
        
        # Covariance matrix (diagonal with confidence_bound)
        self.sigma = np.eye(self.n_assets) * self.confidence_bound
        
        # Historical performance tracking
        self.portfolio_values = [self.portfolio_value]
        self.returns_history = []
        self.weight_history = [self.weights.copy()]
        self.action_history = []
        
        # Get initial observation
        obs = self._get_observation()
        
        # For gym compatibility, also return an info dict
        info = {}
        
        return obs, info
    
    def _get_observation(self):
        """
        Construct the observation from the current state.
        
        Returns:
            numpy.ndarray: Current observation
        """
        # Get historical returns
        hist_returns = self.returns_data[self.current_step - self.window_size:self.current_step]
        
        # Flatten the historical returns
        obs = hist_returns.flatten()
        
        # Add current portfolio weights
        obs = np.concatenate([obs, self.weights])
        
        # Add confidence information (trace of covariance matrix)
        confidence_info = np.array([np.trace(self.sigma)])
        obs = np.concatenate([obs, confidence_info])
        
        # Add feature data if available
        if self.feature_data is not None:
            if self.include_feature_history:
                # Include historical feature data
                hist_features = self.feature_data[self.current_step - self.window_size:self.current_step]
                obs = np.concatenate([obs, hist_features.flatten()])
            else:
                # Include only current feature data
                current_features = self.feature_data[self.current_step - 1]
                obs = np.concatenate([obs, current_features])
        
        return obs.astype(np.float32)
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (numpy.ndarray): RL agent's action to scale CWMR updates
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        # Ensure action is within bounds
        action = np.clip(action, -1.0, 1.0)
        self.action_history.append(action)
        
        # Get the current returns
        returns = self.returns_data[self.current_step]
        
        # Calculate portfolio return before rebalancing
        portfolio_return = np.sum(self.weights * returns)
        
        # Record the return
        self.returns_history.append(portfolio_return)
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_values.append(self.portfolio_value)
        
        # Calculate CWMR update
        new_weights = self._cwmr_update(returns, action)
        
        # Calculate transaction costs
        transaction_cost = self.transaction_cost * np.sum(np.abs(new_weights - self.weights))
        
        # Apply transaction costs
        self.portfolio_value *= (1 - transaction_cost)
        
        # Update weights
        self.weights = new_weights
        self.weight_history.append(self.weights.copy())
        
        # Advance to the next step
        self.current_step += 1
        
        # Check if we've reached the end of the data
        done = self.current_step >= self.n_timesteps - 1
        truncated = False  # Not truncating episodes early
        
        # Calculate reward (log return with penalty for transaction costs)
        reward = np.log(1 + portfolio_return) - transaction_cost
        
        # Scale reward by the reward_scaling factor
        reward *= self.reward_scaling
        
        # Get next observation
        if not done:
            next_obs = self._get_observation()
        else:
            next_obs = np.zeros(self.observation_space.shape)
        
        # Create info dictionary with metrics
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost,
            'weights': self.weights.copy()
        }
        
        return next_obs, reward, done, truncated, info
    
    def _cwmr_update(self, returns, action):
        """
        Update portfolio weights using the CWMR algorithm,
        modified by the RL agent's action.
        
        Args:
            returns (numpy.ndarray): Current returns for each asset
            action (numpy.ndarray): RL agent's action to scale CWMR updates
            
        Returns:
            numpy.ndarray: Updated portfolio weights
        """
        # Safety check for NaN or infinite returns
        if np.isnan(returns).any() or np.isinf(returns).any():
            # If returns contain NaNs or infinities, keep current weights
            return self.weights.copy()
            
        # Calculate mean reversion signal: x_t+1 - x_t
        # Predict returns will revert to the mean
        mean_returns = np.mean(self.returns_data[self.current_step - self.window_size:self.current_step], axis=0)
        price_prediction = -1 * (returns - mean_returns)  # Predict opposite movement from last return
        
        # Safety check for NaN or infinite price predictions
        if np.isnan(price_prediction).any() or np.isinf(price_prediction).any():
            return self.weights.copy()
        
        # Calculate the loss
        loss = 1 - np.dot(self.weights, price_prediction)
        
        # Check if loss exceeds threshold
        if loss > self.epsilon:
            # Calculate confidence
            var = np.dot(np.dot(price_prediction, self.sigma), price_prediction)
            
            # Safety check for numerical stability
            if var < 1e-10:
                return self.weights.copy()
                
            # Update step size using action to scale
            # This is where RL influences the CWMR update
            confidence_scale = 1.0 + np.mean(action)  # Use agent's action to scale confidence
            
            # Ensure confidence_scale is positive
            confidence_scale = max(0.01, confidence_scale)
            
            # Calculate update step
            lambda_value = max(0, (loss - self.epsilon) / var) * confidence_scale
            
            # Safety check for numerical stability
            if lambda_value > 1e6:
                lambda_value = 1e6
                
            # Update mean vector (portfolio weights)
            self.mean = self.mean + lambda_value * self.sigma.dot(price_prediction)
            
            # Check for NaN
            if np.isnan(self.mean).any():
                self.mean = self.weights.copy()
                return self.weights.copy()
            
            # Update covariance matrix with safety check
            denom = (1 / lambda_value + var)
            if denom < 1e-10:
                # Skip covariance update if denominator is too small
                pass
            else:
                self.sigma = self.sigma - lambda_value * np.outer(
                    self.sigma.dot(price_prediction),
                    price_prediction.dot(self.sigma)
                ) / denom
            
            # Ensure covariance matrix stays positive definite
            eigvals, eigvecs = np.linalg.eigh(self.sigma)
            eigvals = np.maximum(eigvals, 1e-8)
            self.sigma = eigvecs.dot(np.diag(eigvals)).dot(eigvecs.T)
            
            # Project back to simplex (normalize to ensure weights sum to 1)
            self.mean = self._simplex_projection(self.mean)
        
        # Return the updated weights
        return self.mean
    
    def _simplex_projection(self, v, z=1.0):
        """
        Project a vector onto the probability simplex.
        
        Args:
            v (numpy.ndarray): Vector to project
            z (float): Sum constraint
            
        Returns:
            numpy.ndarray: Projected vector
        """
        # Handle NaN values
        if np.isnan(v).any() or np.sum(v) == 0:
            # If v contains NaNs or all zeros, return equal weights
            return np.ones(len(v)) / len(v) * z
            
        # Ensure all weights are non-negative
        v = np.maximum(v, 0)
        
        # Check if already a probability vector
        sum_v = np.sum(v)
        if sum_v == z:
            return v
        
        # Handle the case where sum is zero or very small
        if sum_v < 1e-10:
            return np.ones(len(v)) / len(v) * z
            
        # Normalize
        return v * (z / sum_v)
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            returns = self.returns_history[-1] if self.returns_history else 0
            print(f"Step: {self.current_step}, "
                  f"Portfolio Value: {self.portfolio_value:.2f}, "
                  f"Return: {returns:.4f}, "
                  f"Weights: {np.array2string(self.weights, precision=4, suppress_small=True)}")
            
        return None
    
    def get_sharpe_ratio(self):
        """
        Calculate the Sharpe ratio based on the returns history.
        
        Returns:
            float: Sharpe ratio
        """
        if len(self.returns_history) < 2:
            return 0.0
            
        returns = np.array(self.returns_history)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)  # Annualized
        return sharpe
    
    def get_max_drawdown(self):
        """
        Calculate the maximum drawdown based on the portfolio value history.
        
        Returns:
            float: Maximum drawdown
        """
        if len(self.portfolio_values) < 2:
            return 0.0
            
        # Convert to numpy array
        portfolio_values = np.array(self.portfolio_values)
        
        # Calculate the running maximum
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdown
        drawdown = (running_max - portfolio_values) / running_max
        
        # Get maximum drawdown
        max_drawdown = np.max(drawdown)
        
        return max_drawdown
    
    def get_performance_metrics(self):
        """
        Calculate various performance metrics.
        
        Returns:
            dict: Dictionary of performance metrics
        """
        metrics = {
            'final_portfolio_value': self.portfolio_value,
            'total_return': (self.portfolio_value / self.initial_portfolio_value) - 1,
            'sharpe_ratio': self.get_sharpe_ratio(),
            'max_drawdown': self.get_max_drawdown()
        }
        
        # Calculate annualized return if we have enough data
        if len(self.returns_history) > 0:
            n_days = len(self.returns_history)
            metrics['annualized_return'] = ((1 + metrics['total_return']) ** (252 / n_days)) - 1
            
        return metrics 