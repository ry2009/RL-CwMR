#!/bin/bash

# Comprehensive comparison of all portfolio strategies
# This script runs all the CWMR variants and compares their performance

echo "===== CWMR-RL Strategy Comparison ====="
echo "Comparing Equal Weight, Basic CWMR, GRPO-CWMR, and RL-CWMR strategies"

# Default settings - balanced for a meaningful comparison
DATA_DIR="/Users/ryanmathieu/Downloads/stock_market_data/sp500/csv"
MAX_STOCKS=30
START_DATE="2010-01-01"
END_DATE="2020-12-31"
WINDOW_SIZE=15
TRANSACTION_COST=0.0005
CONFIDENCE_BOUND=0.05
EPSILON=0.005
TOTAL_TIMESTEPS=50000
MODEL_NAME="ppo_cwmr_comparison"
USE_FEATURES="--use_features"

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
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
    --data=*)
      # Set data directory
      DATA_DIR="${1#*=}"
      shift
      ;;
    --start=*)
      # Set start date
      START_DATE="${1#*=}"
      shift
      ;;
    --end=*)
      # Set end date
      END_DATE="${1#*=}"
      shift
      ;;
    --name=*)
      # Set model name
      MODEL_NAME="${1#*=}"
      shift
      ;;
    --tune)
      # Perform hyperparameter tuning
      TUNE_HYPERPARAMS="--tune_hyperparams"
      shift
      ;;
    --no-features)
      # Disable features
      USE_FEATURES=""
      shift
      ;;
    --help)
      echo "CWMR-RL Strategy Comparison"
      echo "Usage: ./run_comparison.sh [options]"
      echo ""
      echo "Options:"
      echo "  --stocks=N            Set number of stocks to N (default: 30)"
      echo "  --steps=N             Set number of training steps to N (default: 50000)"
      echo "  --data=PATH           Set data directory"
      echo "  --start=DATE          Set start date (YYYY-MM-DD)"
      echo "  --end=DATE            Set end date (YYYY-MM-DD)"
      echo "  --name=NAME           Set model name"
      echo "  --tune                Perform hyperparameter tuning"
      echo "  --no-features         Disable technical features"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run ./run_comparison.sh --help for usage information."
      exit 1
      ;;
  esac
done

# Print settings
echo ""
echo "Running with settings:"
echo "  Data directory:      $DATA_DIR"
echo "  Number of stocks:    $MAX_STOCKS"
echo "  Time period:         $START_DATE to $END_DATE"
echo "  Window size:         $WINDOW_SIZE"
echo "  Transaction cost:    $TRANSACTION_COST"
echo "  Training steps:      $TOTAL_TIMESTEPS"
echo "  Technical features:  $([ -n "$USE_FEATURES" ] && echo "Enabled" || echo "Disabled")"
echo "  Hyperparameter tune: $([ -n "$TUNE_HYPERPARAMS" ] && echo "Enabled" || echo "Disabled")"
echo ""

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

# Check if the run was successful
if [ $? -eq 0 ]; then
  echo ""
  echo "===== Strategy Comparison Complete ====="
  echo "Results saved to results/$MODEL_NAME"
  echo ""
  echo "Key files:"
  echo "  - results/$MODEL_NAME/strategy_comparison.png     (Portfolio value evolution)"
  echo "  - results/$MODEL_NAME/strategy_metrics_comparison.png (Performance metrics)"
  echo "  - results/$MODEL_NAME/best_strategy_weights.png   (Best strategy weights)"
  echo ""
  echo "To view the results, open the PNG files in the results directory"
else
  echo ""
  echo "Error: Strategy comparison failed"
  echo "Check the error messages above for details"
fi 