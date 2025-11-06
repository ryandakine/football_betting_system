#!/usr/bin/env python3
"""
Run NFL Backtest with Lower Thresholds
========================================

Run backtest with relaxed confidence/edge thresholds to generate actual bets
and train the AI Council on betting outcomes.
"""

import asyncio
import logging
from nfl_system.backtester import NFLBacktester, BacktestSettings
from nfl_system.gold_standard_nfl_config import NFLGoldStandardConfig, NFLThresholdsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # Create custom config with lower thresholds
    config = NFLGoldStandardConfig()
    config.thresholds = NFLThresholdsConfig(
        min_edge_threshold=0.01,      # Lower edge threshold
        confidence_threshold=0.50,    # Lower confidence threshold
    )
    
    # Create backtest settings
    settings = BacktestSettings(
        seasons=["2020", "2021", "2022", "2023", "2024"],
        bankroll_start=10000.0,
        unit_size=100.0,
        max_unit_multiplier=3.0,
        max_exposure=0.12,
    )
    
    # Run backtest
    backtester = NFLBacktester(nfl_config=config, settings=settings)
    results = await backtester.run_comprehensive_backtest()
    backtester.display_results(results)
    
    # Save results
    import json
    with open("backtest_results_adjusted.json", "w") as f:
        # Make results JSON serializable
        results_clean = {
            "seasons_tested": results["seasons_tested"],
            "total_games": results["total_games"],
            "total_bets": results["total_bets"],
            "winning_bets": results["winning_bets"],
            "win_rate": float(results["win_rate"]),
            "average_roi": float(results["average_roi"]),
            "total_profit": float(results["total_profit"]),
        }
        json.dump(results_clean, f, indent=2)
    
    logger.info("✓ Results saved to backtest_results_adjusted.json")


if __name__ == "__main__":
    asyncio.run(main())
