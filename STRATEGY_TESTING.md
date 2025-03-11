# CWMR Strategy Testing

## Overview

This document outlines the approach, implementation, and expected results for testing different variants of the Confidence-Weighted Mean Reversion (CWMR) portfolio allocation strategy, focusing especially on the Reinforcement Learning (RL) enhancements.

The testing framework uses the `strategy_tester.py` and `test_strategies.py` files to systematically evaluate multiple CWMR strategy variants across different market conditions and parameters.

## Strategy Testing Files

### 1. `strategy_tester.py`

This file implements the `StrategyTester` class, which provides a comprehensive framework for:

- **Loading and preprocessing financial data** - Handles data loading, return calculations, feature engineering, and train/validation/test splits.
- **Testing traditional CWMR** - Evaluates standard CWMR with different parameter settings (confidence bounds and epsilon values).
- **Testing RL-enhanced CWMR** - Uses PPO (Proximal Policy Optimization) to learn optimal portfolio weights.
- **Testing CWMR ensemble** - Combines multiple CWMR agents with different parameters for robust performance.
- **Testing RL-CWMR with dynamic parameters** - The RL agent learns not only portfolio weights but also optimal CWMR parameters.
- **Testing GRPO-CWMR** - Implements the Group Relativity Policy Optimization enhancements to CWMR.
- **Comprehensive comparison** - Generates performance metrics, visualizations, and comparisons across all strategies.

### 2. `test_strategies.py`

This file serves as the entry point to run strategy tests. It:

- **Parses command-line arguments** - Supports various configuration options like data directory, number of stocks, date ranges, etc.
- **Initializes the StrategyTester** - Sets up the testing framework with the provided parameters.
- **Runs individual or all strategy tests** - Executes tests based on user input.
- **Generates comparison reports** - Creates charts and tables comparing strategy performance.

## What We're Attempting to Do

Our primary goal is to extract additional alpha from the CWMR strategy by leveraging reinforcement learning techniques. The original CWMR paper demonstrated strong performance for online portfolio selection, but it has some limitations:

1. **Static parameters** - Traditional CWMR uses fixed confidence bounds and epsilon parameters throughout the trading period.
2. **Limited market adaptability** - It doesn't adapt to changing market regimes or conditions.
3. **No feature integration** - The basic algorithm doesn't utilize technical or fundamental features.
4. **Passive reallocation** - Weight updates are mechanical rather than strategically optimized.

Our RL enhancements aim to address these limitations by:

1. **Dynamically adjusting CWMR parameters** - Using RL to find optimal confidence bounds and epsilon values for each market condition.
2. **Incorporating technical features** - Integrating mean reversion indicators and other technical signals.
3. **Optimizing weight updates** - Learning when to follow CWMR signals strongly and when to be more conservative.
4. **Reducing transaction costs** - Making smarter reallocation decisions that balance returns and trading costs.
5. **Combining multiple strategies** - Testing ensemble approaches and group-based optimizations.

## Approach to Testing Different CWMR Variants

### 1. Traditional CWMR Testing
- Test a grid of confidence bound (0.05, 0.1, 0.2) and epsilon (0.001, 0.01, 0.1) parameters
- Evaluate performance metrics including final portfolio value, Sharpe ratio, and maximum drawdown
- Identify the best parameter combinations for different market conditions

### 2. RL-CWMR Testing
- Train a PPO agent to learn optimal portfolio weights given the current market state
- Evaluate the trained agent on unseen test data
- Compare performance against traditional CWMR with optimal fixed parameters

### 3. CWMR Ensemble Testing
- Create an ensemble of CWMR agents with different parameter settings
- Combine their portfolio allocations using equal weighting
- Evaluate whether the ensemble approach reduces variance and improves robustness

### 4. Dynamic Parameter RL-CWMR Testing
- Extend the RL action space to include not just portfolio weights but also CWMR parameters
- Train an agent to simultaneously optimize weights and algorithm parameters
- Evaluate whether dynamic parameter adjustment improves performance

### 5. GRPO-CWMR Testing
- Implement GRPO (Group Relativity Policy Optimization) approach to CWMR
- Group stocks by characteristics and apply CWMR within and across groups
- Test whether this hierarchical approach improves diversification and returns

## Expected Results

Based on prior research and preliminary experiments, we anticipate:

1. **RL-enhanced strategies will outperform traditional CWMR** - By adapting to market conditions and optimizing parameters dynamically, RL approaches should achieve better risk-adjusted returns.

2. **Dynamic parameter adjustment will show benefits** - We expect to see the confidence bound and epsilon parameters changing in response to volatility regimes and market conditions.

3. **Ensemble approaches will reduce variance** - While they may not always have the highest returns, ensemble strategies should show more consistent performance with lower drawdowns.

4. **GRPO approach will improve diversification** - By grouping stocks and applying CWMR at different levels, we expect better sector balance and diversification.

5. **Higher Sharpe ratios for RL strategies** - By optimizing for risk-adjusted returns, RL-enhanced approaches should achieve better Sharpe ratios than traditional methods.

6. **Reduced transaction costs** - RL strategies should learn to make more efficient weight updates that reduce unnecessary trading.

## Performance Metrics

We'll evaluate all strategies using:

1. **Total return** - Final portfolio value relative to starting value
2. **Sharpe ratio** - Risk-adjusted return measure (annualized)
3. **Maximum drawdown** - Largest peak-to-trough decline
4. **Annualized return** - Geometric average annual return
5. **Portfolio weight turnover** - Measure of trading activity
6. **Performance in different market regimes** - Bull, bear, and sideways markets

## Visualization Outputs

The testing framework will generate:

1. **Strategy comparison charts** - Bar charts comparing final values, Sharpe ratios, and drawdowns
2. **Portfolio value time series** - Line charts showing portfolio value evolution for top strategies
3. **Parameter evolution charts** - For dynamic parameter strategies, showing how parameters change over time
4. **Weight allocation heatmaps** - Visualizing how portfolio weights change across assets and time

## Running the Tests

To run a comprehensive comparison of all strategies:

```bash
python test_strategies.py --data_dir="/path/to/stock/data" --max_stocks=20 --start_date="2010-01-01" --end_date="2020-12-31" --test_all
```

To test just specific strategies:

```bash
python test_strategies.py --data_dir="/path/to/stock/data" --max_stocks=20 --test_traditional --test_rl
```

## Conclusion

By systematically testing multiple CWMR variants, we aim to extract additional alpha from this established mean reversion strategy. The comprehensive StrategyTester framework enables fair comparison across approaches and provides rich visualizations and metrics to understand the strengths and weaknesses of each method.

The results will demonstrate whether and how reinforcement learning can enhance traditional algorithmic trading strategies, providing insights for both academic research and practical portfolio management. 