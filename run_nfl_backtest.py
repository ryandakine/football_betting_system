#!/usr/bin/env python3
"""
Run NFL Gold-Standard Backtest
==============================

Convenience script for exercising the NFL backtesting harness once
historical datasets are available under data/football/historical/nfl/.
"""

import asyncio
import logging
from typing import Optional, Sequence

from nfl_system.backtester import NFLBacktester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_nfl_backtest")


async def main(seasons: Optional[Sequence[str]] = None) -> None:
    backtester = NFLBacktester()
    results = await backtester.run_comprehensive_backtest(seasons=seasons)
    backtester.display_results(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the NFL backtest using data/football/historical/nfl/"
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        help="Optional list of seasons to run (e.g. 2021 2022 2023). Defaults to configured range.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.seasons))
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
