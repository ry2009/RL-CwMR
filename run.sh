#!/bin/bash

# Run CWMR-RL on S&P 500 data
# This script provides shortcuts for common configurations

# Default settings
DATA_DIR="/Users/ryanmathieu/Downloads/stock_market_data/sp500/csv"
MAX_STOCKS=20
START_DATE="2010-01-01"
END_DATE="2020-12-31"
WINDOW_SIZE=10
TRANSACTION_COST=0.001
CONFIDENCE_BOUND=0.1
EPSILON=0.01
TOTAL_TIMESTEPS=100000
MODEL_NAME="ppo_cwmr"
USE_FEATURES=""
TUNE_HYPERPARAMS=""

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)
      # Quick test run with minimal settings
      MAX_STOCKS=5
      TOTAL_TIMESTEPS=10000
      shift
      ;;
    --full)
      # Full run with extended settings
      MAX_STOCKS=50
      TOTAL_TIMESTEPS=500000
      USE_FEATURES="--use_features"
      TUNE_HYPERPARAMS="--tune_hyperparams"
      shift
      ;;
    --comprehensive)
      # Comprehensive test with optimal settings
      MAX_STOCKS=100
      TOTAL_TIMESTEPS=1000000
      USE_FEATURES="--use_features"
      TUNE_HYPERPARAMS="--tune_hyperparams"
      WINDOW_SIZE=20
      TRANSACTION_COST=0.0005
      CONFIDENCE_BOUND=0.05
      EPSILON=0.005
      MODEL_NAME="ppo_cwmr_comprehensive"
      shift
      ;;
    --medium)
      # Medium test with balanced settings - faster than comprehensive
      MAX_STOCKS=20
      TOTAL_TIMESTEPS=50000
      USE_FEATURES="--use_features"
      WINDOW_SIZE=15
      TRANSACTION_COST=0.0005
      CONFIDENCE_BOUND=0.05
      EPSILON=0.005
      MODEL_NAME="ppo_cwmr_medium"
      shift
      ;;
    --features)
      # Use technical features
      USE_FEATURES="--use_features"
      shift
      ;;
    --tune)
      # Perform hyperparameter tuning
      TUNE_HYPERPARAMS="--tune_hyperparams"
      shift
      ;;
    --nasdaq)
      # Use NASDAQ data instead of S&P 500
      DATA_DIR="/Users/ryanmathieu/Downloads/stock_market_data/nasdaq/csv"
      shift
      ;;
    --nyse)
      # Use NYSE data instead of S&P 500
      DATA_DIR="/Users/ryanmathieu/Downloads/stock_market_data/nyse/csv"
      shift
      ;;
    --forbes)
      # Use Forbes 2000 data instead of S&P 500
      DATA_DIR="/Users/ryanmathieu/Downloads/stock_market_data/forbes2000/csv"
      shift
      ;;
    --stocks=*)
      # Set number of stocks
      MAX_STOCKS="${1#*=}"
      shift
      ;;
    --steps=*)
      # Set number of training steps
      TOTAL_TIMESTEPS="${1#*=}"
      shift
      ;;
    --name=*)
      # Set model name
      MODEL_NAME="${1#*=}"
      shift
      ;;
    --help)
      echo "CWMR-RL Run Script"
      echo "Usage: ./run.sh [options]"
      echo ""
      echo "Options:"
      echo "  --quick               Quick test run with minimal settings"
      echo "  --medium              Medium test with balanced settings (faster than comprehensive)"
      echo "  --full                Full run with extended settings"
      echo "  --comprehensive       Comprehensive test with optimal settings"
      echo "  --features            Use technical features"
      echo "  --tune                Perform hyperparameter tuning"
      echo "  --nasdaq              Use NASDAQ data instead of S&P 500"
      echo "  --nyse                Use NYSE data instead of S&P 500"
      echo "  --forbes              Use Forbes 2000 data instead of S&P 500"
      echo "  --stocks=N            Set number of stocks to N"
      echo "  --steps=N             Set number of training steps to N"
      echo "  --name=NAME           Set model name to NAME"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run ./run.sh --help for usage information."
      exit 1
      ;;
  esac
done

# Execute the Python script with the configured parameters
python main.py \
  --data_dir="$DATA_DIR" \
  --max_stocks=$MAX_STOCKS \
  --start_date="$START_DATE" \
  --end_date="$END_DATE" \
  --window_size=$WINDOW_SIZE \
  --transaction_cost=$TRANSACTION_COST \
  --confidence_bound=$CONFIDENCE_BOUND \
  --epsilon=$EPSILON \
  --total_timesteps=$TOTAL_TIMESTEPS \
  --model_name="$MODEL_NAME" \
  $USE_FEATURES \
  $TUNE_HYPERPARAMS

# Make the script executable
# chmod +x run.sh 