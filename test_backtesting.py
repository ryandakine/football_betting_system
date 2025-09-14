"""
Test script for backtesting module.
"""

from mlb_betting_system import backtesting


def test_backtest():
    backtesting.backtest("data/predictions.parquet", "data/actual_results.parquet")
    )
