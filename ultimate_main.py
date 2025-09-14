#!/usr/bin/env python3
"""
Ultimate Integrated Tri-Model Betting System v5.0 (Final, Corrected)
"""

#!/usr/bin/env python3
"""
Ultimate Integrated Tri-Model Betting System v5.0 (Final, Corrected)
"""

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from advanced_components import ConfigValidator, IntelligentCache, SystemMonitor
from alerts.manager import AlertSeverity, AlertType, SmartAlertManager
from cached_game_selection import GameSelector

# Your final, proven components
# --- This is the fix ---
from enhanced_odds_fetcher import EnhancedOddsFetcher, StructuredOdds
from tri_model_api_config import get_alert_config, get_trimodel_api_keys
from trimodel_recommendation_engine import AdvancedTriModelRecommendationEngine, EnhancedTriModelBet

# -------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


class UltimateIntegratedBettingSystem:
    # ... (The rest of your main orchestrator class)
    # No other changes are needed in this file.

    def __init__(self, bankroll: float = 500.0, **kwargs):
        self.bankroll = bankroll
        self.config = kwargs
        self.start_time = datetime.now()

        # Load all necessary configurations
        api_keys = get_trimodel_api_keys()
        alert_config = get_alert_config()
        odds_config = get_odds_api_config()

        if not api_keys.get("odds_api"):
            raise ValueError(
                "FATAL: The Odds API key is missing. Please check your .env file."
            )

        # --- THIS IS THE FIX ---
        # Initialize components with their specific configurations
        self.odds_fetcher = EnhancedOddsFetcher(
            api_key=api_keys["odds_api"],
            config=odds_config,  # Pass the odds configuration
        )
        self.ai_analyzer = GameSelector(alert_manager=None)  # Keep it simple for now
        self.recommendation_engine = AdvancedTriModelRecommendationEngine(
            bankroll=self.bankroll, config=self.config
        )
        self.alert_manager = SmartAlertManager(config=alert_config)

        logger.info("🚀 ULTIMATE INTEGRATED BETTING SYSTEM INITIALIZED")
        logger.info(f"   💰 Bankroll: ${bankroll:,.2f}")

    async def run_gold_standard_pipeline(self):
        """Executes the complete, enhanced production pipeline."""
        logger.info("🎯 STARTING GOLD STANDARD PIPELINE")
        run_id = self.start_time.strftime("%Y%m%d_%H%M%S")

        async with self.alert_manager as alert_manager:
            try:
                # Phase 1: Fetch all market data
                market_data = await self.odds_fetcher.get_all_odds_with_props()
                if not market_data or not market_data.games:
                    logger.warning("Pipeline halted: No market data available.")
                    await alert_manager.send_alert(
                        AlertType.SYSTEM_HEALTH,
                        "Pipeline halted",
                        AlertSeverity.WARNING,
                        {"reason": "No market data"},
                    )
                    return

                # Phase 2: Run AI analysis
                ai_results = await self.ai_analyzer.analyze_games(
                    [g.__dict__ for g in market_data.games]
                )
                if not ai_results:
                    logger.warning("Pipeline halted: AI analysis returned no results.")
                    return

                # Phase 3: Generate recommendations
                recommendations = (
                    self.recommendation_engine.generate_comprehensive_recommendations(
                        market_data, ai_results
                    )
                )

                # Phase 4: Optimize portfolio (simplified for now)
                final_portfolio = recommendations[:10]  # Take top 10 potential bets

                # Phase 5: Report and alert
                self._display_report(final_portfolio)
                if final_portfolio:
                    await alert_manager.send_alert(
                        AlertType.NEW_BETS,
                        f"Generated {len(final_portfolio)} new bets.",
                        AlertSeverity.HIGH,
                        {"bets": [asdict(b) for b in final_portfolio]},
                    )

            except Exception as e:
                logger.error(
                    f"❌ A critical error occurred in the pipeline: {e}", exc_info=True
                )
                await alert_manager.send_alert(
                    AlertType.SYSTEM_HEALTH,
                    f"Critical Pipeline Failure: {e}",
                    AlertSeverity.CRITICAL,
                )

    def _display_report(self, portfolio: list[EnhancedTriModelBet]):
        """Prints a clean summary of the final betting portfolio."""
        print("\n" + "=" * 80 + "\n🏆 ULTIMATE SYSTEM - FINAL PORTFOLIO\n" + "=" * 80)
        if not portfolio:
            print("No high-value betting opportunities found today.")
            return

        total_stake = sum(bet.stake for bet in portfolio)
        total_ev = sum(bet.expected_value for bet in portfolio)
        print(
            f"Total Bets: {len(portfolio)} | Total Stake: ${total_stake:.2f} | Total EV: ${total_ev:.2f}"
        )
        print("-" * 80)
        for i, bet in enumerate(portfolio, 1):
            print(
                f"  {i}. {bet.selection} @ {bet.odds} | Stake: ${bet.stake:.2f} | EV: ${bet.expected_value:.2f}"
            )
        print("=" * 80)


async def main():
    """Main execution function."""
    try:
        system = UltimateIntegratedBettingSystem(bankroll=500.0)
        await system.run_gold_standard_pipeline()
    except ValueError as e:
        logger.critical(f"Failed to initialize system: {e}")


if __name__ == "__main__":
    # Ensure your launcher script (launch_system.py) calls this main function.
    asyncio.run(main())
