"""
Run backtest for a given model.
"""

from mlb_betting_system import backtesting


def main():
    predictions_path = "data/predictions.parquet"
    actuals_path = "data/actual_results.parquet"
    backtesting.backtest(predictions_path, actuals_path)


if __name__ == "__main__":
    main()
