#!/usr/bin/env python3
"""
Run NFL Backtest with Bug Fixes
================================

Runs the IMPROVED NFL backtester with all critical bugs fixed:
✅ Proper American odds conversion
✅ Correct Kelly Criterion 
✅ Edge validation
✅ Accurate Sharpe ratio
✅ Statistical significance testing

See NFL_BACKTESTER_BUGS.md for details on what was fixed.
"""

import asyncio
import logging
from typing import Optional, Sequence

from nfl_system.backtester_improved import NFLBacktesterImproved, BacktestSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_nfl_backtest_improved")


async def main(seasons: Optional[Sequence[str]] = None) -> None:
    """
    Run improved NFL backtest with bug fixes applied.
    
    Args:
        seasons: Optional list of seasons (e.g., ["2021", "2022", "2023"])
                If None, uses default range (last 7 years)
    """
    logger.info("🏈 Starting NFL Backtest (IMPROVED - Bug Fixes Applied)")
    logger.info("=" * 70)
    logger.info("Fixes:")
    logger.info("  1. ✅ American odds conversion (+200 → 3.0x, not 2.0x)")
    logger.info("  2. ✅ Profit calculation (stake * (multiplier - 1))")
    logger.info("  3. ✅ Proper Kelly Criterion implementation")
    logger.info("  4. ✅ Edge validation from first principles")
    logger.info("  5. ✅ Sharpe ratio without over-annualization")
    logger.info("  6. ✅ Statistical significance testing")
    logger.info("  7. ✅ Division/conference analysis")
    logger.info("=" * 70)
    
    backtester = NFLBacktesterImproved()
    
    try:
        results = await backtester.run_comprehensive_backtest(seasons=seasons)
        backtester.display_results(results)
        
        # Additional insights
        print("\n" + "=" * 70)
        print("📝 NOTES:")
        print("=" * 70)
        print("• If no bets placed: check your historical data has required columns")
        print("• P-value < 0.05 = statistically significant (95% confidence)")
        print("• Win rate > 52.4% needed to beat -110 vig")
        print("• Sharpe ratio > 1.5 = excellent risk-adjusted returns")
        print("• ROI comparisons:")
        print("    - NCAA system: See backtest_ncaa_improved.py results")
        print("    - Original NFL: See original backtester results")
        print("\n💡 Compare results to original backtester to see impact of bug fixes!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.info("\n💡 To run backtests, you need:")
        logger.info("   1. Historical NFL data in: data/football/historical/nfl/")
        logger.info("   2. Required columns: game_id, home_team, away_team, edge_value, confidence, odds, actual_result")
        logger.info("   3. Optional: division, conference (for division analysis)")
        logger.info("\n📚 See fetch_historical_nfl_data.py to collect this data")
    except Exception as e:
        logger.error(f"❌ Error running backtest: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run IMPROVED NFL backtest with bug fixes"
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        help="Optional list of seasons to run (e.g. 2021 2022 2023). Defaults to last 7 years.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.seasons))
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
