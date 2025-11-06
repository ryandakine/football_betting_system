#!/usr/bin/env python3
"""
Final Production MLB Betting System (v1.1)
Integrates all components with a consistent, modern data flow.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from api_config import get_api_keys
from game_selection import GameSelector

# Your final, proven components
from odds_fetcher import GameInfo, OddsFetcher, StructuredOdds
from recommendation_engine import FinalBet, RecommendationEngine

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProductionBettingSystem:
    """Orchestrates the entire betting pipeline from data collection to final reporting."""

    def __init__(self, bankroll=1000.0, max_exposure_pct=0.10):
        self.bankroll = bankroll
        self.max_exposure = bankroll * max_exposure_pct
        self.start_time = datetime.now()

        self.results = {
            "system_info": {
                "run_id": self.start_time.strftime("%Y%m%d_%H%M%S"),
                "bankroll": bankroll,
                "max_exposure": self.max_exposure,
            },
            "odds_collection": {},
            "ai_analysis": {},
            "recommendations": {},
            "final_portfolio": {},
        }

        logger.info("🚀 Production Betting System initialized")
        logger.info(f"   Bankroll: ${bankroll:,.2f}")
        logger.info(
            f"   Max exposure: ${self.max_exposure:,.2f} ({max_exposure_pct:.1%})"
        )

    async def run_production_pipeline(self):
        """Executes the complete production betting pipeline."""
        logger.info("🎯 STARTING PRODUCTION MLB BETTING PIPELINE")
        logger.info("=" * 60)

        try:
            odds_data = await self._collect_comprehensive_odds()
            if not odds_data.games:
                logger.warning("Halting: No games found by OddsFetcher.")
                return False

            ai_analysis = await self._run_ai_game_analysis(odds_data.games)
            if not ai_analysis:
                logger.warning("Halting: AI analysis returned no results.")
                return False

            recommendations = self._generate_smart_recommendations(
                odds_data, ai_analysis
            )
            optimized_portfolio = self._optimize_betting_portfolio(recommendations)
            self._generate_production_reports(optimized_portfolio)

            logger.info("✅ PRODUCTION PIPELINE COMPLETED SUCCESSFULLY!")
            return True

        except Exception as e:
            logger.error(f"❌ Critical pipeline failure: {e}", exc_info=True)
            return False

    async def _collect_comprehensive_odds(self) -> StructuredOdds:
        """Phase 1: Collects comprehensive odds using the OddsFetcher."""
        logger.info("📊 PHASE 1: Comprehensive Odds Collection")
        api_keys = get_api_keys()
        fetcher = OddsFetcher(api_key=api_keys["odds_api"])
        structured_odds = await fetcher.get_all_odds_with_props()

        # This no longer calls to_legacy_format, solving the bug.
        self.results["odds_collection"] = {
            "games_found": len(structured_odds.games),
            "h2h_opportunities": len(structured_odds.h2h_bets),
            "prop_opportunities": len(structured_odds.prop_bets),
        }
        return structured_odds

    async def _run_ai_game_analysis(self, games_list: list[GameInfo]) -> dict[str, Any]:
        """Phase 2: Runs AI analysis on all available games."""
        logger.info("🤖 PHASE 2: AI-Powered Game Analysis")
        selector = GameSelector()
        # The AI analyzer needs dicts, so we convert the dataclasses just for this step
        games_as_dicts = [game.__dict__ for game in games_list]
        analysis_results = await selector.analyze_games(games_as_dicts)

        self.results["ai_analysis"] = analysis_results
        logger.info(f"   ✅ AI analysis complete for {len(analysis_results)} games.")
        return analysis_results

    def _generate_smart_recommendations(
        self, odds_data: StructuredOdds, ai_analysis: dict
    ) -> list[FinalBet]:
        """Phase 3: Generates betting recommendations using the RecommendationEngine."""
        logger.info("💡 PHASE 3: Smart Recommendation Generation")
        engine = RecommendationEngine()
        # The engine is designed to take the StructuredOdds object directly
        recommendations = engine.generate_recommendations(odds_data, ai_analysis)
        self.results["recommendations"] = [bet.__dict__ for bet in recommendations]
        return recommendations

    def _optimize_betting_portfolio(
        self, recommendations: list[FinalBet]
    ) -> list[FinalBet]:
        """Phase 4: Applies bankroll management and filters for the best bets."""
        logger.info("💰 PHASE 4: Portfolio Optimization")

        best_bet_per_game: dict[str, FinalBet] = {}
        for bet in recommendations:
            if (
                bet.game_id not in best_bet_per_game
                or bet.priority_score > best_bet_per_game[bet.game_id].priority_score
            ):
                best_bet_per_game[bet.game_id] = bet

        sorted_bets = sorted(
            best_bet_per_game.values(), key=lambda x: x.priority_score, reverse=True
        )

        final_portfolio: list[FinalBet] = []
        current_exposure = 0.0
        for bet in sorted_bets:
            if current_exposure + bet.stake <= self.max_exposure:
                final_portfolio.append(bet)
                current_exposure += bet.stake

        self.results["final_portfolio"] = {
            "bet_count": len(final_portfolio),
            "total_staked": current_exposure,
            "total_ev": sum(b.expected_value for b in final_portfolio),
            "bets": [bet.__dict__ for bet in final_portfolio],
        }
        logger.info(f"   ✅ Portfolio optimized to {len(final_portfolio)} final bets.")
        return final_portfolio

    def _generate_production_reports(self, portfolio: list[FinalBet]):
        """Phase 5: Generates and saves final console and file reports."""
        logger.info("📋 PHASE 5: Production Reporting")

        summary = self.results["final_portfolio"]
        report = [
            "\n" + "=" * 60,
            "📬 FINAL BETTING PORTFOLIO",
            "=" * 60,
            f"TOTALS: {summary['bet_count']} Bets | Total Stake: ${summary['total_staked']:.2f} | Total EV: ${summary['total_ev']:.2f}",
            "-" * 60,
        ]
        if not portfolio:
            report.append("No bets met the portfolio criteria today.")
        else:
            for i, bet in enumerate(portfolio, 1):
                report.append(
                    f"{i}. {bet.selection} ({bet.odds}) | Units: {bet.units} | Stake: ${bet.stake:.2f} | EV: ${bet.expected_value:.2f}"
                )
        print("\n".join(report))

        output_dir = Path("production_runs")
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / f"run_{self.results['system_info']['run_id']}.json"

        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"   💾 Full production run data saved to {filepath}")


if __name__ == "__main__":
    system = ProductionBettingSystem(bankroll=1000.0, max_exposure_pct=0.10)
    asyncio.run(system.run_production_pipeline())
