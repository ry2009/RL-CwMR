# CWMR-RL: Confidence-Weighted Mean Reversion with Reinforcement Learning

This project implements an enhanced trading strategy that combines **Confidence-Weighted Mean Reversion (CWMR)** with **Reinforcement Learning (RL)** for optimized portfolio allocation in financial markets. The approach significantly improves upon traditional CWMR by using RL to dynamically adjust trading parameters, confidence levels, and portfolio weights based on market conditions.

## Overview

CWMR is a portfolio allocation strategy based on the principle that asset prices tend to revert to their mean. The traditional CWMR algorithm maintains a probabilistic model over portfolio weights, updating them when the expected return falls below a certain threshold.

Our CWMR-RL enhancement uses Reinforcement Learning to:

1. **Dynamically adjust confidence levels** for different market regimes
2. **Scale portfolio weight updates** based on detected mean reversion strengths
3. **Optimize trading parameters** (window size, epsilon, etc.) for current market conditions
4. **Reduce transaction costs** by making smarter reallocation decisions

## Key Features

- **Integrated RL-CWMR Framework**: Combines traditional CWMR with Proximal Policy Optimization (PPO)
- **Mean Reversion Feature Engineering**: Auto-generates technical indicators relevant to mean reversion
- **Transaction Cost Awareness**: Balances trading frequency with transaction costs
- **Multiple Strategy Comparison**: Compares RL-CWMR against equal-weight, traditional CWMR, and GRPO-CWMR baselines
- **Hyperparameter Optimization**: Includes utilities for tuning model parameters
- **Comprehensive Metrics**: Tracks Sharpe ratio, drawdowns, returns, and other key performance indicators

## Strategy Variants

This project implements several variants of CWMR-based portfolio allocation strategies:

1. **Equal Weight**: Simple baseline that equally distributes capital across all assets
2. **Basic CWMR**: Traditional Confidence-Weighted Mean Reversion with numerical stability improvements
3. **GRPO-CWMR**: Group Relativity Policy Optimization enhanced CWMR that groups stocks by performance characteristics
4. **RL-CWMR**: Reinforcement Learning enhanced CWMR that dynamically adjusts strategy parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CWMR-RL.git
cd CWMR-RL

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Test

For a quick test of all strategies with minimal settings:

```bash
./run.sh --quick
```

### Medium Test

For a more thorough test with technical features and 20 stocks:

```bash
./run.sh --medium
```

### Comprehensive Test

For a full-scale test with parameter tuning and 100 stocks:

```bash
./run.sh --comprehensive
```

### Strategy Comparison

For a detailed comparison of all implemented portfolio strategies:

```bash
./run_comparison.sh
```

### Full Options

```bash
python main.py \
  --data_dir="/path/to/stock/data" \
  --max_stocks=20 \
  --start_date="2010-01-01" \
  --end_date="2020-12-31" \
  --window_size=10 \
  --transaction_cost=0.001 \
  --confidence_bound=0.1 \
  --epsilon=0.01 \
  --total_timesteps=100000 \
  --use_features \
  --tune_hyperparams
```

### Key Arguments

- `--data_dir`: Directory containing stock data in CSV format
- `--max_stocks`: Maximum number of stocks to include in portfolio
- `--window_size`: Number of historical periods to consider for mean reversion
- `--transaction_cost`: Transaction cost as a fraction
- `--use_features`: Enable technical feature generation for improved state representation
- `--tune_hyperparams`: Perform hyperparameter tuning before training

## Project Structure

```
CWMR-RL/
├── data_loader.py        # Data loading and preprocessing utilities
├── cwmr_env.py           # RL environment implementing CWMR
├── rl_trainer.py         # Training and evaluation utilities
├── main.py               # Main script for running experiments
├── run.sh                # Shell script for common configurations
├── run_comparison.sh     # Script for comprehensive strategy comparison
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Required Data Format

The system expects stock data in CSV format with at least the following columns:
- `Date`: Trading date in YYYY-MM-DD format
- `Adjusted Close`: Adjusted closing price

CSV files should be named by ticker symbol (e.g., AAPL.csv, MSFT.csv).

## Performance Metrics

The system tracks and visualizes:
- Portfolio value over time for all strategies
- Strategy comparison based on returns, Sharpe ratio, and drawdowns
- Portfolio weights evolution
- Detailed metrics for each strategy
- Heatmaps of weight evolution for the best-performing strategy

## Technical Features

When enabled, the system generates various technical features:
- Price and returns z-scores at multiple time scales
- Moving average distance indicators
- Momentum and volatility features
- Cross-sectional features comparing stocks against each other

## Future Work and Potential Enhancements

The CWMR-RL project provides a solid foundation for advanced portfolio optimization, but there are numerous directions for further development:

### 1. Advanced Machine Learning Enhancements

#### 1.1 Architecture Improvements
- **Multi-head Attention Mechanisms**: Implement transformer-based architectures to better capture relationships between assets
- **Recurrent Policy Networks**: Replace standard MLPs with LSTM or GRU networks to better capture temporal patterns
- **Graph Neural Networks**: Represent stocks as nodes in a graph based on industry sectors, correlations, or other relationships
- **Ensemble Methods**: Combine multiple RL agents with different objectives or initialization parameters

#### 1.2 Training Optimizations
- **Curriculum Learning**: Gradually increase task difficulty during training by starting with simpler market conditions
- **Meta-Learning**: Develop models that can quickly adapt to new market regimes with minimal data
- **Distributional RL**: Use quantile regression or categorical DQN approaches to model the full distribution of returns
- **Hindsight Experience Replay**: Learn from unsuccessful episodes by replacing target returns with achieved returns

### 2. Financial Model Enhancements

#### 2.1 Market Regime Modeling
- **Regime Detection**: Implement explicit market regime detection (bull, bear, sideways, etc.)
- **Separate CWMR Models**: Train specialized models for different market regimes
- **Macro Feature Integration**: Incorporate macroeconomic indicators (interest rates, GDP, inflation, etc.)
- **Implicit Volatility Forecasting**: Adjust strategy based on predicted market volatility

#### 2.2 Risk Management
- **CVaR Optimization**: Replace or supplement Sharpe ratio with Conditional Value at Risk (CVaR) measures
- **Dynamic Risk Budgeting**: Adjust risk appetite based on market conditions
- **Drawdown Control**: Implement explicit drawdown control mechanisms
- **Kelly Criterion**: Apply optimal betting size principles to position sizing

#### 2.3 Portfolio Construction
- **Factor Integration**: Incorporate traditional factor models (Fama-French, etc.)
- **Hierarchical Portfolio Construction**: Allocate first to sectors/industries, then to individual stocks
- **Multi-period Optimization**: Consider multi-step ahead returns instead of single-step optimization
- **Black-Litterman Integration**: Combine market equilibrium with CWMR views

### 3. Technical and Implementation Improvements

#### 3.1 Scalability
- **Distributed Training**: Implement distributed training across multiple machines
- **GPU Acceleration**: Optimize tensor operations for GPU processing
- **Database Integration**: Replace CSV files with a proper database backend
- **Incremental Learning**: Support model updates with new data without full retraining

#### 3.2 Feature Engineering
- **Automated Feature Selection**: Implement feature importance analysis and selection
- **Alternative Data Integration**: Incorporate sentiment data, news flow, or other alternative data
- **Feature Compression**: Use autoencoders to create more efficient feature representations
- **Cross-asset Features**: Create features that capture relationships between different asset classes

#### 3.3 Evaluation Framework
- **Benchmark Extensions**: Compare against more sophisticated baselines (ETFs, mutual funds, etc.)
- **Monte Carlo Simulations**: Create synthetic market data for robust testing
- **Stress Testing**: Evaluate performance under extreme market conditions
- **Transaction Cost Models**: Implement more realistic transaction cost models with market impact

### 4. Practical Applications

#### 4.1 Real-world Deployment
- **Live Trading Integration**: Connect to brokerage APIs for real-time trading
- **Paper Trading Mode**: Implement a simulation mode with real-time market data
- **Scheduled Rebalancing**: Support various rebalancing frequencies (daily, weekly, monthly)
- **Tax-Aware Trading**: Consider tax implications in trading decisions

#### 4.2 User Interface
- **Web Dashboard**: Create an interactive web interface for strategy monitoring
- **Strategy Customization UI**: Allow users to adjust parameters without coding
- **Performance Visualization**: Enhance visualization of strategy performance and attribution
- **Alerting System**: Implement alerts for significant portfolio changes or market events

### 5. Advanced Research Directions

#### 5.1 Multi-agent Systems
- **Competitive RL Agents**: Create multiple agents competing in the same market environment
- **Cooperative Learning**: Implement agent specialization across different market conditions
- **Market Impact Modeling**: Model how agent actions affect market prices
- **Agent Populations**: Evolve a population of agents with genetic algorithms

#### 5.2 Explainable AI
- **Decision Attribution**: Explain model decisions in terms of specific market signals
- **Counterfactual Analysis**: Implement "what if" scenario analysis for trades
- **Feature Importance Visualization**: Show which features drive specific decisions
- **Rule Extraction**: Distill RL policies into interpretable trading rules

#### 5.3 Theoretical Extensions
- **Information Theory Approaches**: Apply concepts like entropy to measure market efficiency
- **Causal Discovery**: Identify causal relationships between market variables
- **Robust Optimization**: Create strategies that perform well under model uncertainty
- **Game Theory Integration**: Model market interactions as multi-player games

## Getting Started on Enhancements

For team members looking to contribute to enhancements:

1. **Fork the repository** and create a feature branch
2. **Select an area of focus** from the ideas above
3. **Create a design document** outlining your approach
4. **Implement a minimal working example** to validate the concept
5. **Run comparative tests** against the current implementation
6. **Document your methodology and results** thoroughly
7. **Submit a pull request** with your enhancements

We recommend starting with smaller, well-defined enhancements before tackling more ambitious projects. The modular design of CWMR-RL makes it relatively straightforward to extend and enhance specific components without rewriting the entire system.

## References

- Li, B., Hoi, S.C., Sahoo, D. and Liu, Z.Y., 2011. Moving average reversion strategy for on-line portfolio selection. Artificial Intelligence, 222, pp.104-123.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O., 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
- Yang, G., Luo, W., Zhang, T. et al., 2022. GRPO: Group relative policy optimization for improving online portfolio selection. Applied Intelligence, 52, pp.12033–12045.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This implementation builds upon research in online portfolio selection and reinforcement learning
- Portions of the CWMR algorithm are based on the work of Li et al. 
- The GRPO approach is inspired by Yang et al. # RL-CwMR
