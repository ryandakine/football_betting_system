#!/usr/bin/env python3
"""
Football Production Betting System (NFL & College Football)
Adapted from MLB system for football betting with enhanced schedule handling.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from football_game_selection import FootballGameSelector, FootballSelectionConfig

# Football-specific components
from football_odds_fetcher import FootballOddsFetcher, GameInfo, StructuredOdds
from football_recommendation_engine import FinalBet, FootballRecommendationEngine

from api_config import get_api_keys

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FootballProductionBettingSystem:
    """Orchestrates the entire football betting pipeline from data collection to final reporting."""

    def __init__(self, bankroll=1000.0, max_exposure_pct=0.10, sport_type="nfl", test_mode=False, fake_money=True, no_api=False):
        self.bankroll = bankroll
        self.max_exposure = bankroll * max_exposure_pct
        self.start_time = datetime.now()
        self.sport_type = sport_type  # "nfl" or "ncaaf"
        self.test_mode = test_mode  # Use sample data instead of live data
        self.no_api = no_api  # Skip API calls, use mock data
        self.fake_money = fake_money  # Use fake money for testing

        # Performance tracking for fake money testing
        self.performance_metrics = {
            "predictions_made": 0,
            "predictions_correct": 0,
            "total_edge_analyzed": 0,
            "avg_edge_found": 0.0,
            "games_analyzed": 0,
            "recommendations_generated": 0,
            "accuracy_by_market": {"moneyline": 0, "spreads": 0, "totals": 0, "props": 0},
            "edge_distribution": [],
            "confidence_distribution": [],
        }

        # Sport-specific configurations
        self.sport_configs = {
            "nfl": {
                "sport_key": "americanfootball_nfl",
                "markets": ["h2h", "spreads", "totals", "player_props"],
                "schedule_days": ["thursday", "sunday", "monday"],
                "max_games_per_day": 16,
            },
            "ncaaf": {
                "sport_key": "americanfootball_ncaaf",
                "markets": ["h2h", "spreads", "totals", "player_props"],
                "schedule_days": ["thursday", "friday", "saturday"],
                "max_games_per_day": 100,  # College football has many games
            },
        }

        self.results = {
            "system_info": {
                "run_id": self.start_time.strftime("%Y%m%d_%H%M%S"),
                "bankroll": bankroll,
                "max_exposure": self.max_exposure,
                "sport_type": sport_type,
                "test_mode": test_mode,
                "fake_money": fake_money,
                "mode": "FAKE MONEY TESTING" if fake_money else "LIVE BETTING",
            },
            "odds_collection": {},
            "ai_analysis": {},
            "recommendations": {},
            "final_portfolio": {},
            "performance_metrics": self.performance_metrics,
        }

        logger.info("🚀 Football Production Betting System initialized")
        logger.info(f"   Sport: {sport_type.upper()}")
        logger.info(f"   Bankroll: ${bankroll:,.2f}")
        logger.info(f"   Max exposure: ${self.max_exposure:,.2f} ({max_exposure_pct:.1%})")
        logger.info(f"   Mode: {'FAKE MONEY TESTING' if fake_money else 'LIVE BETTING'}")
        logger.info(f"   Test Mode: {test_mode}")

        if fake_money:
            logger.info("💰 FAKE MONEY MODE: No real bets will be placed")
            logger.info("📊 FOCUS: Prediction accuracy and performance monitoring")

    async def run_production_pipeline(self):
        """Executes the complete production football betting pipeline."""
        logger.info(f"🎯 STARTING PRODUCTION {self.sport_type.upper()} BETTING PIPELINE")
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

            recommendations = self._generate_smart_recommendations(odds_data, ai_analysis)
            optimized_portfolio = self._optimize_betting_portfolio(recommendations)
            self._generate_production_reports(optimized_portfolio)

            logger.info("✅ PRODUCTION PIPELINE COMPLETED SUCCESSFULLY!")
            return True

        except Exception as e:
            logger.error(f"❌ Critical pipeline failure: {e}", exc_info=True)
            return False

    async def _collect_comprehensive_odds(self) -> StructuredOdds:
        """Phase 1: Collects comprehensive odds using the FootballOddsFetcher."""
        logger.info("📊 PHASE 1: Comprehensive Odds Collection")

        if self.no_api:
            logger.info("🔧 NO-API MODE: Generating mock data instead of API calls")
            from mock_data_generator import FootballMockDataGenerator
            mock_generator = FootballMockDataGenerator(self.sport_type)
            structured_odds = await mock_generator.generate_mock_odds_data()
        else:
            api_keys = get_api_keys()
            sport_config = self.sport_configs[self.sport_type]

            async with FootballOddsFetcher(
                api_key=api_keys["odds_api"],
                sport_key=sport_config["sport_key"],
                markets=sport_config["markets"],
            ) as fetcher:
                structured_odds = await fetcher.get_all_odds_with_props()

        self.results["odds_collection"] = {
            "games_found": len(structured_odds.games),
            "h2h_opportunities": len(structured_odds.h2h_bets),
            "spread_opportunities": len(structured_odds.spread_bets),
            "total_opportunities": len(structured_odds.total_bets),
            "prop_opportunities": len(structured_odds.prop_bets),
            "sport_type": self.sport_type,
        }
        return structured_odds

    async def _run_ai_game_analysis(self, games: list[GameInfo]) -> dict:
        """Phase 2: Runs AI analysis on games."""
        logger.info("🤖 PHASE 2: AI Game Analysis")

        # Use Perplexity as primary for live context and smart analysis
        config = FootballSelectionConfig(
            sport_type=self.sport_type,
            primary_provider="perplexity",
            fallback_providers=["claude", "chatgpt", "grok", "gemini"],
            enable_ensemble=True,
            consensus_threshold=2,
        )
        selector = FootballGameSelector(config)
        analysis_results = await selector.analyze_games([g.to_dict() for g in games])

        self.results["ai_analysis"] = {
            "games_analyzed": len(games),
            "ai_recommendations": len(analysis_results.get("recommendations", [])),
            "confidence_scores": analysis_results.get("confidence_distribution", {}),
            "sport_type": self.sport_type,
        }
        return analysis_results

    def _generate_smart_recommendations(
        self, odds_data: StructuredOdds, ai_analysis: dict
    ) -> list[FinalBet]:
        """Phase 3: Generates smart betting recommendations."""
        logger.info("💡 PHASE 3: Smart Recommendation Generation")

        engine = FootballRecommendationEngine(bankroll=self.bankroll)

        recommendations = engine.generate_recommendations(odds_data, ai_analysis)

        self.results["recommendations"] = {
            "total_recommendations": len(recommendations),
            "by_market_type": self._categorize_recommendations(recommendations),
            "total_expected_value": sum(
                r.expected_value for r in recommendations if r.expected_value
            ),
            "sport_type": self.sport_type,
        }
        return recommendations

    def _optimize_betting_portfolio(self, recommendations: list[FinalBet]) -> list[FinalBet]:
        """Phase 4: Optimizes the betting portfolio."""
        logger.info("📈 PHASE 4: Portfolio Optimization")

        # Sort by expected value and apply risk management
        sorted_recs = sorted(recommendations, key=lambda x: x.expected_value or 0, reverse=True)

        # Apply Kelly Criterion and risk limits
        optimized_portfolio = []
        total_exposure = 0.0

        for rec in sorted_recs:
            if self.fake_money:
                # In fake money mode, track all recommendations for performance analysis
                optimized_portfolio.append(rec)
                # Track performance metrics
                self._track_performance_metrics(rec)
            else:
                # Live betting mode - apply risk management
                if total_exposure + rec.stake <= self.max_exposure:
                    optimized_portfolio.append(rec)
                    total_exposure += rec.stake
                else:
                    # Reduce stake to fit within limits
                    remaining_exposure = self.max_exposure - total_exposure
                    if remaining_exposure > 0:
                        rec.stake = remaining_exposure
                        optimized_portfolio.append(rec)
                    break

        self.results["final_portfolio"] = {
            "bets_selected": len(optimized_portfolio),
            "total_exposure": total_exposure,
            "portfolio_expected_value": sum(
                r.expected_value for r in optimized_portfolio if r.expected_value
            ),
            "sport_type": self.sport_type,
            "fake_money_mode": self.fake_money,
            "performance_metrics": self.performance_metrics,
        }
        return optimized_portfolio

    def _track_performance_metrics(self, recommendation: FinalBet):
        """Tracks performance metrics for fake money testing."""
        self.performance_metrics["recommendations_generated"] += 1

        # Track edge and confidence distributions
        if recommendation.expected_value:
            self.performance_metrics["total_edge_analyzed"] += 1
            self.performance_metrics["edge_distribution"].append(recommendation.expected_value)

        # Update average edge
        if self.performance_metrics["edge_distribution"]:
            self.performance_metrics["avg_edge_found"] = sum(self.performance_metrics["edge_distribution"]) / len(self.performance_metrics["edge_distribution"])

        # Track by market type (this would be enhanced with actual outcome tracking)
        bet_type = recommendation.bet_type.lower()
        if "moneyline" in bet_type:
            self.performance_metrics["accuracy_by_market"]["moneyline"] += 1
        elif "spread" in bet_type:
            self.performance_metrics["accuracy_by_market"]["spreads"] += 1
        elif "total" in bet_type:
            self.performance_metrics["accuracy_by_market"]["totals"] += 1
        else:
            self.performance_metrics["accuracy_by_market"]["props"] += 1

        logger.info(f"📊 Performance Tracking - Edge: ${recommendation.expected_value:.2f}, "
                   f"Confidence: {getattr(recommendation, 'confidence', 'N/A')}, "
                   f"Market: {bet_type}")

    def _generate_production_reports(self, portfolio: list[FinalBet]):
        """Phase 5: Generates comprehensive production reports."""
        logger.info("📊 PHASE 5: Production Reporting")

        # Save results to file
        output_dir = Path(f"data/football/{self.sport_type}")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"production_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Display summary
        self._display_summary(portfolio)

    def _categorize_recommendations(self, recommendations: list[FinalBet]) -> dict:
        """Categorizes recommendations by market type."""
        categories = {"moneyline": 0, "spreads": 0, "totals": 0, "props": 0}
        for rec in recommendations:
            if "moneyline" in rec.bet_type.lower():
                categories["moneyline"] += 1
            elif "spread" in rec.bet_type.lower():
                categories["spreads"] += 1
            elif "total" in rec.bet_type.lower():
                categories["totals"] += 1
            else:
                categories["props"] += 1
        return categories

    def _display_summary(self, portfolio: list[FinalBet]):
        """Displays a summary of the betting portfolio or performance metrics."""
        logger.info("=" * 60)
        if self.fake_money:
            logger.info(f"🏈 {self.sport_type.upper()} PERFORMANCE ANALYSIS - FAKE MONEY MODE")
        else:
            logger.info(f"🏈 {self.sport_type.upper()} BETTING PORTFOLIO SUMMARY")
        logger.info("=" * 60)

        if not portfolio:
            logger.info("No recommendations generated for this session.")
            return

        if self.fake_money:
            # Display performance metrics for fake money testing
            logger.info("📊 PERFORMANCE METRICS:")
            logger.info(f"   Recommendations Generated: {self.performance_metrics['recommendations_generated']}")
            logger.info(f"   Average Edge Found: ${self.performance_metrics['avg_edge_found']:.2f}")
            logger.info(f"   Total Edge Analyzed: {self.performance_metrics['total_edge_analyzed']}")

            logger.info("\n📈 MARKET TYPE DISTRIBUTION:")
            for market, count in self.performance_metrics['accuracy_by_market'].items():
                if count > 0:
                    logger.info(f"   {market.title()}: {count} recommendations")

            # Calculate potential ROI based on recommendations
            total_potential_ev = sum(r.expected_value for r in portfolio if r.expected_value)
            logger.info(f"\n💰 POTENTIAL PERFORMANCE:")
            logger.info(f"   Total Expected Value: ${total_potential_ev:,.2f}")
            logger.info(f"   Average EV per Bet: ${total_potential_ev/len(portfolio):.2f}")

            logger.info("\n🎯 TOP RECOMMENDATIONS (by Expected Value):")
            sorted_portfolio = sorted(portfolio, key=lambda x: x.expected_value or 0, reverse=True)
            for i, bet in enumerate(sorted_portfolio[:5], 1):
                matchup = (
                    f"{bet.away_team} at {bet.home_team}"
                    if (bet.home_team or bet.away_team)
                    else bet.game_id
                )
                logger.info(f"{i}. {matchup} - {bet.bet_type}")
                logger.info(f"   EV: ${bet.expected_value:.2f} | Confidence: {getattr(bet, 'confidence', 'N/A')}")

        else:
            # Display traditional betting portfolio summary
            total_stake = sum(r.stake for r in portfolio)
            total_ev = sum(r.expected_value for r in portfolio if r.expected_value)

            logger.info(f"Total Bets: {len(portfolio)}")
            logger.info(f"Total Stake: ${total_stake:,.2f}")
            logger.info(f"Total Expected Value: ${total_ev:,.2f}")
            logger.info(f"ROI: {(total_ev/total_stake)*100:.2f}%" if total_stake > 0 else "N/A")

            logger.info("\nTop Recommendations:")
            for i, bet in enumerate(portfolio[:5], 1):
                matchup = (
                    f"{bet.away_team} at {bet.home_team}"
                    if (bet.home_team or bet.away_team)
                    else bet.game_id
                )
                logger.info(f"{i}. {matchup} - {bet.bet_type}")
                logger.info(f"   Stake: ${bet.stake:,.2f} | EV: ${bet.expected_value:,.2f}")


async def main():
    """Main execution function."""
    import sys

    # Parse command line arguments
    sport_type = sys.argv[1] if len(sys.argv) > 1 else "nfl"
    bankroll = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0

    if sport_type not in ["nfl", "ncaaf"]:
        logger.error("Sport type must be 'nfl' or 'ncaaf'")
        return

    # Initialize and run the system
    system = FootballProductionBettingSystem(
        bankroll=bankroll, max_exposure_pct=0.10, sport_type=sport_type
    )

    success = await system.run_production_pipeline()

    if success:
        logger.info("🎉 Football betting system completed successfully!")
    else:
        logger.error("❌ Football betting system failed!")


if __name__ == "__main__":
    asyncio.run(main())
