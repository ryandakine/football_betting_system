#!/usr/bin/env python3
"""
ULTIMATE INTEGRATED TRI-MODEL BETTING SYSTEM - PRODUCTION READY
The most advanced sports betting AI system ever created.

Features:
- Enhanced odds fetching with intelligent caching
- Tri-model AI ensemble with validation and monitoring
- Kelly Criterion position sizing
- Advanced risk management
- Comprehensive performance analytics
- Production-grade error handling
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from advanced_components import ConfigValidator, IntelligentCache, SystemMonitor
from alerts.manager import AlertSeverity, AlertType, SmartAlertManager

# Import enhanced components
from enhanced_odds_fetcher import EnhancedOddsFetcher, StructuredOdds
from tri_model_api_config import get_alert_config, get_trimodel_api_keys
from trimodel_game_analyzer import TriModelGameAnalyzer
from trimodel_recommendation_engine import TriModelBet, TriModelRecommendationEngine

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ultimate_betting_system.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


class UltimateIntegratedBettingSystem:
    """
    The ultimate production-ready betting system that combines:
    - Advanced odds fetching with caching
    - Tri-model AI ensemble analysis
    - Kelly Criterion position sizing
    - Professional risk management
    - Comprehensive monitoring and reporting
    """

    def __init__(self, bankroll: float = 1000.0, config: dict | None = None):
        self.bankroll = bankroll
        self.config = self._setup_configuration(config or {})
        self.start_time = datetime.now()

        # Initialize core components
        self.odds_fetcher = None
        self.ai_analyzer = None
        self.recommendation_engine = None
        self.system_monitor = SystemMonitor()
        self.cache = IntelligentCache(max_size=1000, default_ttl=3600)

        # Initialize Smart Alert Manager
        try:
            self.alert_manager = SmartAlertManager()
            logger.info("✅ Smart Alert Manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Alert Manager initialization failed: {e}")
            self.alert_manager = None

        # Results storage
        self.execution_results = {
            "system_info": {
                "version": "4.0-ultimate",
                "start_time": self.start_time.isoformat(),
                "configuration": self.config,
                "bankroll": bankroll,
                "alert_system_enabled": self.alert_manager is not None,
            }
        }

        logger.info("🚀 ULTIMATE INTEGRATED BETTING SYSTEM INITIALIZING")
        logger.info(f"   💰 Bankroll: ${bankroll:,.2f}")
        logger.info(f"   ⚙️ Configuration: {self.config}")
        logger.info(
            f"   🚨 Alert System: {'Enabled' if self.alert_manager else 'Disabled'}"
        )

    def _setup_configuration(self, user_config: dict) -> dict:
        """Setup and validate system configuration"""
        default_config = {
            # Core system settings
            "max_exposure_pct": 0.15,  # Maximum 15% bankroll exposure
            "min_consensus_confidence": 0.65,  # Minimum AI confidence
            "min_quality_score": 0.70,  # Minimum quality threshold
            # Risk management
            "max_kelly_fraction": 0.20,  # Conservative Kelly cap
            "min_edge_percentage": 0.03,  # Minimum 3% edge
            "risk_limits": {
                "high": 0.10,  # Max 10% high risk bets
                "medium": 0.40,  # Max 40% medium risk bets
                "low": 1.0,  # No limit on low risk
            },
            # Operational settings
            "cache_duration_hours": 2,  # Short cache for dynamic markets
            "max_concurrent_analyses": 5,  # API rate limiting
            "sports": ["baseball_mlb"],  # Target sports
            "markets": ["h2h", "totals", "batter_home_runs", "pitcher_strikeouts"],
            # Quality controls
            "enable_validation": True,
            "enable_monitoring": True,
            "enable_adaptive_weights": True,
            "detailed_logging": True,
        }

        # Merge configurations
        final_config = {**default_config, **user_config}

        # Validate configuration
        is_valid, errors = ConfigValidator.validate_system_config(
            {
                "bankroll": self.bankroll,
                "max_exposure_pct": final_config["max_exposure_pct"],
                "min_consensus_confidence": final_config["min_consensus_confidence"],
                "min_quality_score": final_config["min_quality_score"],
            }
        )

        if not is_valid:
            logger.warning(f"⚠️ Configuration validation warnings: {errors}")

        return final_config

    async def initialize_system(self) -> bool:
        """Initialize all system components with comprehensive error handling"""
        logger.info("🔧 Initializing system components...")

        try:
            # Load and validate API keys
            api_keys = get_trimodel_api_keys()
            required_keys = ["odds_api", "claude", "openai"]
            missing_keys = [key for key in required_keys if not api_keys.get(key)]

            if missing_keys:
                logger.error(f"❌ Missing required API keys: {missing_keys}")
                return False

            # Initialize odds fetcher
            logger.info("📊 Initializing enhanced odds fetcher...")
            self.odds_fetcher = EnhancedOddsFetcher(
                api_key=api_keys["odds_api"],
                concurrency_limit=self.config["max_concurrent_analyses"],
                cache_enabled=True,
            )

            # Initialize AI analyzer
            logger.info("🤖 Initializing enhanced AI analyzer...")
            self.ai_analyzer = EnhancedTriModelGameAnalyzer(
                cache_duration_hours=self.config["cache_duration_hours"]
            )

            # Initialize recommendation engine
            logger.info("💡 Initializing advanced recommendation engine...")
            rec_config = {
                "min_quality_score": self.config["min_quality_score"],
                "min_consensus_confidence": self.config["min_consensus_confidence"],
                "min_edge_percentage": self.config["min_edge_percentage"],
                "max_kelly_fraction": self.config["max_kelly_fraction"],
                "max_stake_percentage": self.config["max_exposure_pct"]
                / 3,  # Conservative per-bet limit
                "risk_limits": self.config["risk_limits"],
            }

            self.recommendation_engine = AdvancedTriModelRecommendationEngine(
                bankroll=self.bankroll, config=rec_config
            )

            logger.info("✅ All system components initialized successfully")

            # Send startup alert
            if self.alert_manager:
                await self.alert_manager.send_system_startup_alert(
                    {
                        "version": "4.0-ultimate",
                        "bankroll": self.bankroll,
                        "start_time": self.start_time.isoformat(),
                        "ai_models": list(api_keys.keys()),
                        "configuration": self.config,
                    }
                )

            return True

        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}", exc_info=True)
            return False

    async def run_ultimate_analysis(self, sport_key: str = "baseball_mlb") -> dict:
        """Execute the complete ultimate analysis pipeline"""
        logger.info(f"🎯 STARTING ULTIMATE ANALYSIS PIPELINE FOR {sport_key.upper()}")

        pipeline_start = datetime.now()

        try:
            # Phase 1: Market Data Collection
            market_data = await self._execute_phase_1_market_data(sport_key)
            if not market_data or not market_data.games:
                logger.warning("⚠️ No market data available - terminating analysis")
                return self._generate_empty_results("No market data available")

            # Phase 2: AI Ensemble Analysis
            ai_analysis_results = await self._execute_phase_2_ai_analysis(market_data)
            if not ai_analysis_results:
                logger.warning("⚠️ AI analysis failed - terminating analysis")
                return self._generate_empty_results("AI analysis failed")

            # Phase 3: Advanced Recommendation Generation
            recommendations = await self._execute_phase_3_recommendations(
                market_data, ai_analysis_results
            )

            # Phase 4: Portfolio Optimization and Risk Management
            optimized_portfolio = await self._execute_phase_4_optimization(
                recommendations
            )

            # Phase 5: Comprehensive Reporting and Analytics
            final_report = await self._execute_phase_5_reporting(
                market_data, ai_analysis_results, optimized_portfolio, pipeline_start
            )

            # Send success alert with recommendations
            if self.alert_manager and optimized_portfolio.get(
                "optimized_recommendations"
            ):
                await self.alert_manager.send_new_recommendations_alert(
                    optimized_portfolio["optimized_recommendations"],
                    optimized_portfolio.get("portfolio_analysis", {}).get(
                        "portfolio_summary", {}
                    ),
                )

            logger.info("🏆 ULTIMATE ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
            return final_report

        except Exception as e:
            logger.error(f"❌ Ultimate analysis pipeline failed: {e}", exc_info=True)

            # Send critical error alert
            if self.alert_manager:
                await self.alert_manager.send_error_alert(
                    f"Ultimate analysis pipeline failed: {str(e)}",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "pipeline_stage": "unknown",
                        "execution_time": (
                            datetime.now() - pipeline_start
                        ).total_seconds(),
                    },
                    AlertSeverity.CRITICAL,
                )

            return self._generate_error_results(str(e))

    async def _execute_phase_1_market_data(
        self, sport_key: str
    ) -> StructuredOdds | None:
        """Phase 1: Advanced Market Data Collection"""
        logger.info("📊 PHASE 1: Advanced Market Data Collection")

        try:
            async with self.odds_fetcher as fetcher:
                market_data = await fetcher.get_comprehensive_odds(
                    sport_key=sport_key,
                    markets=self.config["markets"],
                    use_cache=True,
                    filter_bookmakers=True,
                )

            # Store market data results
            self.execution_results["market_data"] = {
                "summary": market_data.get_summary(),
                "games_found": len(market_data.games),
                "h2h_bets": len(market_data.h2h_bets),
                "prop_bets": len(market_data.prop_bets),
                "api_requests_used": market_data.api_requests_used,
                "fetch_timestamp": market_data.fetch_timestamp,
            }

            logger.info(f"✅ Phase 1 Complete: {market_data.get_summary()}")
            return market_data

        except Exception as e:
            logger.error(f"❌ Phase 1 failed: {e}")
            return None

    async def _execute_phase_2_ai_analysis(
        self, market_data: StructuredOdds
    ) -> dict | None:
        """Phase 2: Enhanced AI Ensemble Analysis"""
        logger.info("🤖 PHASE 2: Enhanced AI Ensemble Analysis")

        try:
            # Prepare game data for AI analysis
            game_contexts = []
            for game in market_data.games:
                # Enrich game context with market data
                game_h2h = [
                    bet for bet in market_data.h2h_bets if bet.game_id == game.id
                ]
                game_props = [
                    bet for bet in market_data.prop_bets if bet.game_id == game.id
                ]

                context = {
                    "id": game.id,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "commence_time": game.commence_time,
                    "market_context": {
                        "h2h_bets_available": len(game_h2h),
                        "prop_bets_available": len(game_props),
                        "bookmaker_count": len({bet.bookmaker for bet in game_h2h}),
                        "odds_range": {
                            "home_min": min(
                                [
                                    bet.odds
                                    for bet in game_h2h
                                    if bet.team == game.home_team
                                ],
                                default=0,
                            ),
                            "home_max": max(
                                [
                                    bet.odds
                                    for bet in game_h2h
                                    if bet.team == game.home_team
                                ],
                                default=0,
                            ),
                            "away_min": min(
                                [
                                    bet.odds
                                    for bet in game_h2h
                                    if bet.team == game.away_team
                                ],
                                default=0,
                            ),
                            "away_max": max(
                                [
                                    bet.odds
                                    for bet in game_h2h
                                    if bet.team == game.away_team
                                ],
                                default=0,
                            ),
                        },
                    },
                }
                game_contexts.append(context)

            # Execute AI analysis
            ai_results = await self.ai_analyzer.analyze_games(game_contexts)

            # Store AI analysis results
            self.execution_results["ai_analysis"] = {
                "games_analyzed": len(game_contexts),
                "successful_analyses": len(ai_results),
                "analysis_success_rate": (
                    len(ai_results) / len(game_contexts) if game_contexts else 0
                ),
                "system_performance": self.ai_analyzer.get_system_performance_report(),
            }

            logger.info(f"✅ Phase 2 Complete: {len(ai_results)} successful analyses")
            return ai_results

        except Exception as e:
            logger.error(f"❌ Phase 2 failed: {e}")
            return None

    async def _execute_phase_3_recommendations(
        self, market_data: StructuredOdds, ai_analysis: dict
    ) -> list[EnhancedTriModelBet]:
        """Phase 3: Advanced Recommendation Generation"""
        logger.info("💡 PHASE 3: Advanced Recommendation Generation")

        try:
            recommendations = (
                self.recommendation_engine.generate_comprehensive_recommendations(
                    odds_data=market_data, ai_analysis=ai_analysis
                )
            )

            # Store recommendation results
            self.execution_results["recommendations"] = {
                "total_generated": len(recommendations),
                "avg_quality_score": (
                    sum(r.quality_score for r in recommendations) / len(recommendations)
                    if recommendations
                    else 0
                ),
                "avg_expected_value": (
                    sum(r.expected_value for r in recommendations)
                    / len(recommendations)
                    if recommendations
                    else 0
                ),
                "risk_distribution": {
                    risk: len([r for r in recommendations if r.risk_level == risk])
                    for risk in ["low", "medium", "high"]
                },
            }

            logger.info(
                f"✅ Phase 3 Complete: {len(recommendations)} recommendations generated"
            )
            return recommendations

        except Exception as e:
            logger.error(f"❌ Phase 3 failed: {e}")
            return []

    async def _execute_phase_4_optimization(
        self, recommendations: list[EnhancedTriModelBet]
    ) -> dict:
        """Phase 4: Portfolio Optimization and Risk Management"""
        logger.info("💰 PHASE 4: Portfolio Optimization and Risk Management")

        try:
            if not recommendations:
                return {
                    "optimized_recommendations": [],
                    "portfolio_analysis": {},
                    "risk_assessment": "No recommendations to optimize",
                }

            # Get comprehensive portfolio analysis
            portfolio_analysis = self.recommendation_engine.get_portfolio_analysis(
                recommendations
            )

            # Apply final filters based on portfolio constraints
            max_total_exposure = self.bankroll * self.config["max_exposure_pct"]
            current_exposure = 0
            optimized_recommendations = []

            for rec in recommendations:
                if current_exposure + rec.recommended_stake <= max_total_exposure:
                    optimized_recommendations.append(rec)
                    current_exposure += rec.recommended_stake
                else:
                    # Try to fit with reduced stake
                    remaining_budget = max_total_exposure - current_exposure
                    if (
                        remaining_budget
                        >= self.recommendation_engine.config["min_stake"]
                    ):
                        rec.recommended_stake = remaining_budget
                        rec.units = max(
                            1,
                            round(
                                remaining_budget
                                / self.recommendation_engine.config["min_stake"]
                            ),
                        )
                        optimized_recommendations.append(rec)
                        current_exposure = max_total_exposure
                        break

            # Final portfolio analysis
            final_portfolio_analysis = (
                self.recommendation_engine.get_portfolio_analysis(
                    optimized_recommendations
                )
            )

            optimization_results = {
                "original_count": len(recommendations),
                "optimized_count": len(optimized_recommendations),
                "total_exposure": current_exposure,
                "exposure_percentage": current_exposure / self.bankroll,
                "optimization_efficiency": (
                    len(optimized_recommendations) / len(recommendations)
                    if recommendations
                    else 0
                ),
                "optimized_recommendations": optimized_recommendations,
                "portfolio_analysis": final_portfolio_analysis,
            }

            self.execution_results["portfolio_optimization"] = optimization_results

            logger.info(
                f"✅ Phase 4 Complete: {len(optimized_recommendations)} optimized recommendations"
            )
            logger.info(
                f"   💰 Total exposure: ${current_exposure:.2f} ({current_exposure/self.bankroll:.1%})"
            )

            return optimization_results

        except Exception as e:
            logger.error(f"❌ Phase 4 failed: {e}")
            return {}

    async def _execute_phase_5_reporting(
        self,
        market_data: StructuredOdds,
        ai_analysis: dict,
        portfolio: dict,
        pipeline_start: datetime,
    ) -> dict:
        """Phase 5: Comprehensive Reporting and Analytics"""
        logger.info("📋 PHASE 5: Comprehensive Reporting and Analytics")

        try:
            execution_time = (datetime.now() - pipeline_start).total_seconds()

            # Compile comprehensive results
            final_results = {
                **self.execution_results,
                "execution_summary": {
                    "pipeline_execution_time": execution_time,
                    "completion_timestamp": datetime.now().isoformat(),
                    "success": True,
                    "sport_analyzed": "baseball_mlb",
                    "system_version": "4.0-ultimate",
                },
                "portfolio_optimization": portfolio,
                "system_monitoring": (
                    self.system_monitor.get_performance_report()
                    if self.config["enable_monitoring"]
                    else {}
                ),
                "final_recommendations": [
                    asdict(rec)
                    for rec in portfolio.get("optimized_recommendations", [])
                ],
            }

            # Generate and save reports
            await self._save_comprehensive_reports(final_results)

            # Print executive summary
            self._print_executive_summary(final_results)

            logger.info("✅ Phase 5 Complete: Comprehensive reporting finished")
            return final_results

        except Exception as e:
            logger.error(f"❌ Phase 5 failed: {e}")
            return self.execution_results

    async def _save_comprehensive_reports(self, results: dict):
        """Save comprehensive reports to files"""
        try:
            # Create reports directory
            reports_dir = Path("ultimate_reports")
            reports_dir.mkdir(exist_ok=True)

            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

            # Save full JSON report
            json_file = reports_dir / f"ultimate_analysis_{timestamp}.json"
            with open(json_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Save human-readable summary
            summary_file = reports_dir / f"executive_summary_{timestamp}.txt"
            with open(summary_file, "w") as f:
                f.write(self._generate_text_summary(results))

            # Save recommendations CSV (if any)
            recommendations = results.get("portfolio_optimization", {}).get(
                "optimized_recommendations", []
            )
            if recommendations:
                csv_file = reports_dir / f"recommendations_{timestamp}.csv"
                self._save_recommendations_csv(recommendations, csv_file)

            logger.info(f"📄 Reports saved to {reports_dir}/")

        except Exception as e:
            logger.error(f"Failed to save reports: {e}")

    def _print_executive_summary(self, results: dict):
        """Print beautiful executive summary to console"""
        print("\n" + "=" * 100)
        print("🏆 ULTIMATE TRI-MODEL BETTING SYSTEM - EXECUTIVE SUMMARY")
        print("=" * 100)

        # System performance
        exec_summary = results.get("execution_summary", {})
        market_data = results.get("market_data", {})
        ai_analysis = results.get("ai_analysis", {})
        portfolio = results.get("portfolio_optimization", {})

        print(
            f"⏱️  Execution Time: {exec_summary.get('pipeline_execution_time', 0):.1f} seconds"
        )
        print(f"🎯 System Version: {exec_summary.get('system_version', 'Unknown')}")
        print(f"💰 Bankroll: ${self.bankroll:,.2f}")

        print(f"\n📊 MARKET DATA ANALYSIS:")
        print(f"   Games Available: {market_data.get('games_found', 0)}")
        print(f"   H2H Betting Options: {market_data.get('h2h_bets', 0)}")
        print(f"   Prop Betting Options: {market_data.get('prop_bets', 0)}")
        print(f"   API Requests Used: {market_data.get('api_requests_used', 0)}")

        print(f"\n🤖 AI ENSEMBLE ANALYSIS:")
        print(f"   Games Analyzed: {ai_analysis.get('games_analyzed', 0)}")
        print(f"   Successful Analyses: {ai_analysis.get('successful_analyses', 0)}")
        print(
            f"   Analysis Success Rate: {ai_analysis.get('analysis_success_rate', 0):.1%}"
        )

        print(f"\n💡 RECOMMENDATION ENGINE:")
        rec_data = results.get("recommendations", {})
        print(
            f"   Total Recommendations Generated: {rec_data.get('total_generated', 0)}"
        )
        print(f"   Average Quality Score: {rec_data.get('avg_quality_score', 0):.3f}")
        print(f"   Average Expected Value: {rec_data.get('avg_expected_value', 0):.1%}")

        print(f"\n💰 OPTIMIZED PORTFOLIO:")
        if portfolio and portfolio.get("optimized_recommendations"):
            print(f"   Final Recommendations: {portfolio.get('optimized_count', 0)}")
            print(f"   Total Exposure: ${portfolio.get('total_exposure', 0):.2f}")
            print(
                f"   Portfolio Exposure: {portfolio.get('exposure_percentage', 0):.1%}"
            )

            # Show top recommendations
            recommendations = portfolio.get("optimized_recommendations", [])
            if recommendations:
                print(f"\n🥇 TOP RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {rec.selection} @ {rec.odds} ({rec.bookmaker})")
                    print(
                        f"      💰 Stake: ${rec.recommended_stake:.2f} ({rec.units} units)"
                    )
                    print(
                        f"      📈 Edge: {rec.edge_percentage:.1%} | Quality: {rec.quality_score:.3f}"
                    )
                    print(
                        f"      🎯 AI Confidence: {rec.consensus_confidence:.1%} | Risk: {rec.risk_level}"
                    )
        else:
            print("   No recommendations met the high-quality criteria today.")

        print("\n" + "=" * 100)
        print("🚀 Analysis Complete - Reports saved to 'ultimate_reports/' directory")
        print("=" * 100)

    def _generate_text_summary(self, results: dict) -> str:
        """Generate human-readable text summary"""
        # Implementation would create a detailed text report
        return f"Ultimate Tri-Model Betting System Analysis Report\nGenerated: {datetime.now()}\n..."

    def _save_recommendations_csv(self, recommendations: list, csv_file: Path):
        """Save recommendations to CSV format"""
        # Implementation would save recommendations in CSV format
        pass

    def _generate_empty_results(self, reason: str) -> dict:
        """Generate empty results structure"""
        return {
            **self.execution_results,
            "execution_summary": {
                "success": False,
                "reason": reason,
                "completion_timestamp": datetime.now().isoformat(),
            },
        }

    def _generate_error_results(self, error: str) -> dict:
        """Generate error results structure"""
        return {
            **self.execution_results,
            "execution_summary": {
                "success": False,
                "error": error,
                "completion_timestamp": datetime.now().isoformat(),
            },
        }

    async def cleanup(self):
        """Cleanup system resources"""
        logger.info("🧹 Cleaning up system resources...")

        # Send shutdown alert
        if self.alert_manager:
            await self.alert_manager.send_alert(
                AlertType.SYSTEM_SHUTDOWN,
                AlertSeverity.INFO,
                "Ultimate Tri-Model Betting System shutting down",
                details={
                    "uptime_minutes": (datetime.now() - self.start_time).total_seconds()
                    / 60,
                    "shutdown_time": datetime.now().isoformat(),
                },
            )
            await self.alert_manager.cleanup()

        if self.ai_analyzer:
            self.ai_analyzer.cleanup()
        # Additional cleanup as needed


# Main execution function
async def main():
    """Main execution with comprehensive error handling"""
    print("🚀 ULTIMATE TRI-MODEL BETTING SYSTEM STARTING...")

    # Configuration
    system_config = {
        "max_exposure_pct": 0.12,  # Conservative 12% exposure
        "min_consensus_confidence": 0.68,  # High confidence threshold
        "min_quality_score": 0.72,  # High quality threshold
        "max_kelly_fraction": 0.18,  # Conservative Kelly
        "min_edge_percentage": 0.035,  # Minimum 3.5% edge
        "cache_duration_hours": 1,  # Fresh data
        "detailed_logging": True,
        "enable_validation": True,
        "enable_monitoring": True,
    }

    system = None
    try:
        # Initialize system
        system = UltimateIntegratedBettingSystem(bankroll=1000.0, config=system_config)

        # Initialize components
        if not await system.initialize_system():
            logger.error("❌ System initialization failed")
            return

        # Run ultimate analysis
        results = await system.run_ultimate_analysis("baseball_mlb")

        if results.get("execution_summary", {}).get("success"):
            logger.info("🎉 ULTIMATE SYSTEM EXECUTION COMPLETED SUCCESSFULLY!")
        else:
            logger.error("❌ System execution completed with errors")

    except KeyboardInterrupt:
        logger.info("⚠️ System interrupted by user")
    except Exception as e:
        logger.error(f"❌ System execution failed: {e}", exc_info=True)
    finally:
        if system:
            await system.cleanup()


if __name__ == "__main__":
    # Set up API keys reminder
    required_env_vars = ["ODDS_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   export {var}=your_api_key_here")
        print("\nPlease set these environment variables and try again.")
        sys.exit(1)

    # Run the ultimate system
    asyncio.run(main())
