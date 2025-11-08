#!/usr/bin/env python3
"""Simple test script for NCAA backtester"""

import asyncio
import sys
sys.path.insert(0, '.')

from college_football_system.backtester import NCAABacktester, BacktestSettings


async def main():
    print("=" * 70)
    print("NCAA BACKTESTER TEST")
    print("=" * 70)

    # Configure for 2023 season only
    settings = BacktestSettings(seasons=['2023'])

    # Initialize backtester
    backtester = NCAABacktester(settings=settings)

    print(f"\nInitialized backtester for seasons: {settings.seasons}")
    print(f"Starting bankroll: ${settings.bankroll_start:,.2f}")
    print(f"Unit size: ${settings.unit_size:.2f}")
    print(f"Min edge threshold: {settings.min_edge_threshold:.1%}")
    print(f"Confidence threshold: {settings.confidence_threshold:.1%}")

    # Run backtest
    print("\nRunning backtest...")
    results = await backtester.run_comprehensive_backtest()

    # Display results
    print("\n")
    backtester.display_results(results)

    return results


if __name__ == "__main__":
    results = asyncio.run(main())
