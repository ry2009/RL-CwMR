#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for comparing different CWMR-based portfolio strategies with RL enhancements.
This script leverages the insights from the original CWMR paper to extract additional alpha.
"""

import os
import argparse
from strategy_tester import StrategyTester

def parse_args():
    """
    Parse command line arguments for the strategy test script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Compare different CWMR-based portfolio selection strategies with RL enhancements'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/Users/ryanmathieu/Downloads/stock_market_data/sp500/csv',
        help='Directory containing stock data in CSV format'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/strategy_test',
        help='Directory to save test results'
    )
    
    parser.add_argument(
        '--max_stocks',
        type=int,
        default=20,
        help='Maximum number of stocks to include in the portfolio'
    )
    
    parser.add_argument(
        '--start_date',
        type=str,
        default='2015-01-01',
        help='Start date for testing (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end_date',
        type=str,
        default='2020-12-31',
        help='End date for testing (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--window_size',
        type=int,
        default=10,
        help='Lookback window size for mean reversion strategies'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=50000,
        help='Number of training steps for RL models'
    )
    
    parser.add_argument(
        '--transaction_cost',
        type=float,
        default=0.001,
        help='Transaction cost as a fraction of trade value'
    )
    
    parser.add_argument(
        '--test_rl',
        action='store_true',
        help='Include RL-based strategies in the test (may take longer)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run a quick test with fewer parameters and training steps'
    )
    
    parser.add_argument(
        '--multi_agent',
        action='store_true',
        help='Focus on testing multi-agent CWMR ensemble approach'
    )
    
    parser.add_argument(
        '--num_agents',
        type=int,
        default=5,
        help='Number of agents to use in the multi-agent ensemble'
    )
    
    return parser.parse_args()

def main():
    """
    Main function to run the strategy comparison tests.
    """
    args = parse_args()
    
    # If quick mode, reduce parameters
    if args.quick:
        args.max_stocks = 10
        args.timesteps = 10000
        print("Running in quick mode with reduced parameters")
    
    print(f"Starting strategy comparison with {args.max_stocks} stocks")
    print(f"Testing period: {args.start_date} to {args.end_date}")
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize the strategy tester
    tester = StrategyTester(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        max_stocks=args.max_stocks,
        start_date=args.start_date,
        end_date=args.end_date,
        window_size=args.window_size
    )
    
    # If multi-agent mode, focus on testing the multi-agent approach
    if args.multi_agent:
        print("\n=== Testing Multi-Agent CWMR Ensemble Strategy ===")
        print(f"Using {args.num_agents} agents with different parameters")
        
        # Test CWMR ensemble with specified number of agents
        ensemble_results = tester.test_cwmr_ensemble(num_agents=args.num_agents)
        
        # Also test traditional CWMR with a single set of parameters for comparison
        print("\n=== Testing Traditional CWMR for Comparison ===")
        tester.test_traditional_cwmr(
            confidence_bounds=[0.1],
            epsilons=[0.01]
        )
        
        # Compare against GRPO for benchmark
        print("\n=== Testing GRPO-CWMR for Benchmark ===")
        tester.test_grpo_cwmr(
            group_size=5,
            n_groups=4,
            transaction_cost=args.transaction_cost
        )
        
        # Generate comparison report
        print("\n=== Generating Multi-Agent Strategy Comparison Report ===")
        summary = tester.compare_all_strategies()
        
        return
    
    # Standard testing flow if not in multi-agent mode
    # Test traditional CWMR with different parameters
    print("\n=== Testing Traditional CWMR with Different Parameters ===")
    if args.quick:
        confidence_bounds = [0.1]
        epsilons = [0.01]
    else:
        confidence_bounds = [0.05, 0.1, 0.2]
        epsilons = [0.001, 0.01, 0.1]
    
    tester.test_traditional_cwmr(
        confidence_bounds=confidence_bounds,
        epsilons=epsilons
    )
    
    # Test CWMR ensemble
    print("\n=== Testing CWMR Ensemble Strategy ===")
    tester.test_cwmr_ensemble(num_agents=5 if not args.quick else 3)
    
    # Test GRPO-CWMR
    print("\n=== Testing GRPO-CWMR Strategy ===")
    tester.test_grpo_cwmr(
        group_size=5,
        n_groups=4,
        transaction_cost=args.transaction_cost
    )
    
    # Test RL-based strategies if requested
    if args.test_rl:
        print("\n=== Testing RL-Enhanced CWMR Strategies ===")
        
        # Test basic RL-CWMR
        print("Testing basic RL-CWMR...")
        tester.test_rl_cwmr(
            total_timesteps=args.timesteps,
            transaction_cost=args.transaction_cost,
            reward_scaling=1.0
        )
        
        # Test RL-CWMR with dynamic parameter adjustment
        print("Testing RL-CWMR with dynamic parameter adjustment...")
        tester.test_rl_cwmr_dynamic_params(
            total_timesteps=args.timesteps,
            transaction_cost=args.transaction_cost,
            reward_scaling=1.0
        )
    
    # Compare all strategies
    print("\n=== Generating Strategy Comparison Report ===")
    summary = tester.compare_all_strategies()
    
    print("\nStrategy comparison complete!")
    print(f"Results saved to {args.results_dir}")

if __name__ == "__main__":
    main()
