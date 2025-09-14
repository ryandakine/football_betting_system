#!/usr/bin/env python3
"""
Learning System Integration for MLB Betting
===========================================
Integrates the self-learning feedback system with existing betting components
to create a continuously improving prediction system.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import the learning system
from self_learning_feedback_system import LearningPattern, PredictionRecord, SelfLearningFeedbackSystem

# Import existing system components (adjust imports based on your actual structure)
try:
    from gold_standard_main import GoldStandardMLBSystem
    from odds_fetcher import UltimateIntegratedBettingSystem
    from ultimate_database_manager import UltimateDatabaseManager
except ImportError:
    # Fallback imports if the modules don't exist
    UltimateDatabaseManager = None
    UltimateIntegratedBettingSystem = None
    GoldStandardMLBSystem = None

logger = logging.getLogger(__name__)


class LearningSystemIntegrator:
    """
    Integrates the self-learning feedback system with existing betting components.
    """

    def __init__(self, learning_system: SelfLearningFeedbackSystem):
        self.learning_system = learning_system
        self.integration_active = True
        self.last_learning_analysis = None

        logger.info("🔗 Learning System Integrator initialized")

    async def enhance_prediction_with_learning(
        self,
        base_prediction: dict[str, Any],
        game_data: dict[str, Any],
        model_name: str = "ensemble_model",
    ) -> dict[str, Any]:
        """
        Enhance a base prediction using learned patterns and insights.
        """
        if not self.integration_active:
            return base_prediction

        try:
            # Extract base confidence from prediction
            base_confidence = base_prediction.get("confidence", 0.5)

            # Generate enhanced prediction using learning system
            enhanced_confidence, learning_metadata = (
                await self.learning_system.generate_enhanced_prediction(
                    game_data, base_confidence, model_name
                )
            )

            # Create enhanced prediction
            enhanced_prediction = base_prediction.copy()
            enhanced_prediction["confidence"] = enhanced_confidence
            enhanced_prediction["learning_metadata"] = learning_metadata
            enhanced_prediction["original_confidence"] = base_confidence
            enhanced_prediction["learning_boost"] = learning_metadata["learning_boost"]

            # Add pattern information
            if learning_metadata["applied_patterns"]:
                enhanced_prediction["applied_patterns"] = learning_metadata[
                    "applied_patterns"
                ]
                enhanced_prediction["pattern_count"] = len(
                    learning_metadata["applied_patterns"]
                )
            else:
                enhanced_prediction["applied_patterns"] = []
                enhanced_prediction["pattern_count"] = 0

            logger.info(
                f"🎯 Enhanced prediction confidence: {base_confidence:.3f} → {enhanced_confidence:.3f} "
                f"(boost: {learning_metadata['learning_boost']:+.3f})"
            )

            return enhanced_prediction

        except Exception as e:
            logger.error(f"Error enhancing prediction: {e}")
            return base_prediction

    async def record_prediction_for_learning(
        self, prediction_data: dict[str, Any], game_data: dict[str, Any]
    ) -> str:
        """
        Record a prediction in the learning system for future analysis.
        """
        try:
            # Create prediction record
            prediction = PredictionRecord(
                prediction_id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                timestamp=datetime.now().isoformat(),
                game_id=game_data.get("game_id", "unknown"),
                home_team=game_data.get("home_team", ""),
                away_team=game_data.get("away_team", ""),
                predicted_winner=prediction_data.get("predicted_winner", ""),
                confidence=prediction_data.get("confidence", 0.5),
                stake_amount=prediction_data.get("stake_amount", 0.0),
                odds=prediction_data.get("odds", 0.0),
                expected_value=prediction_data.get("expected_value", 0.0),
                features=self._extract_features(game_data),
                model_name=prediction_data.get("model_name", "unknown"),
                strategy_type=prediction_data.get("strategy_type", "unknown"),
            )

            # Record in learning system
            await self.learning_system.record_prediction(prediction)

            logger.info(
                f"📝 Recorded prediction {prediction.prediction_id} for learning"
            )
            return prediction.prediction_id

        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
            return None

    async def update_outcome_for_learning(
        self, prediction_id: str, actual_winner: str, actual_profit: float
    ):
        """
        Update a prediction with its actual outcome for learning.
        """
        try:
            await self.learning_system.update_prediction_outcome(
                prediction_id, actual_winner, actual_profit
            )
            logger.info(f"✅ Updated outcome for prediction {prediction_id}")
        except Exception as e:
            logger.error(f"Error updating outcome: {e}")

    async def run_learning_analysis(self, force: bool = False):
        """
        Run learning analysis to update patterns and insights.
        """
        # Check if we should run analysis (every 24 hours or when forced)
        now = datetime.now()
        if (
            not force
            and self.last_learning_analysis
            and (now - self.last_learning_analysis).total_seconds() < 86400
        ):  # 24 hours
            return

        try:
            logger.info("🧠 Running learning analysis...")
            await self.learning_system.analyze_and_learn()
            self.last_learning_analysis = now

            # Get and log insights
            insights = self.learning_system.get_learning_insights()
            await self._log_learning_insights(insights)

        except Exception as e:
            logger.error(f"Error running learning analysis: {e}")

    async def get_learning_recommendations(self) -> dict[str, Any]:
        """
        Get recommendations based on learning insights.
        """
        try:
            insights = self.learning_system.get_learning_insights()

            recommendations = {
                "system_health": "good",
                "confidence_adjustments": {},
                "strategy_recommendations": [],
                "pattern_insights": [],
                "performance_summary": {},
            }

            # Analyze model performance
            for model_name, performance in insights.get(
                "model_performance", {}
            ).items():
                if performance["last_30_days_accuracy"] < 0.5:
                    recommendations["confidence_adjustments"][
                        model_name
                    ] = "reduce_confidence"
                elif performance["last_30_days_accuracy"] > 0.7:
                    recommendations["confidence_adjustments"][
                        model_name
                    ] = "increase_confidence"

            # Add strategy recommendations
            for recommendation in insights.get("learning_recommendations", []):
                recommendations["strategy_recommendations"].append(recommendation)

            # Add pattern insights
            for pattern in insights.get("top_patterns", []):
                recommendations["pattern_insights"].append(
                    {
                        "description": pattern["description"],
                        "strength": pattern["strength"],
                        "success_rate": pattern["success_rate"],
                    }
                )

            # Performance summary
            recommendations["performance_summary"] = {
                "total_predictions": insights.get("total_predictions", 0),
                "recent_accuracy": insights.get("recent_accuracy", 0.0),
                "active_patterns": insights.get("active_patterns", 0),
            }

            return recommendations

        except Exception as e:
            logger.error(f"Error getting learning recommendations: {e}")
            return {"error": str(e)}

    def _extract_features(self, game_data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract features from game data for learning.
        """
        features = {}

        # Basic game features
        features["home_team"] = game_data.get("home_team", "")
        features["away_team"] = game_data.get("away_team", "")
        features["odds"] = game_data.get("odds", 0.0)

        # Weather features
        if "weather" in game_data:
            features["weather_condition"] = game_data["weather"].get("condition", "")
            features["temperature"] = game_data["weather"].get("temperature", 0)
            features["wind_speed"] = game_data["weather"].get("wind_speed", 0)

        # Team performance features
        if "home_team_stats" in game_data:
            features["home_team_wins"] = game_data["home_team_stats"].get("wins", 0)
            features["home_team_losses"] = game_data["home_team_stats"].get("losses", 0)
            features["home_team_win_pct"] = game_data["home_team_stats"].get(
                "win_percentage", 0.0
            )

        if "away_team_stats" in game_data:
            features["away_team_wins"] = game_data["away_team_stats"].get("wins", 0)
            features["away_team_losses"] = game_data["away_team_stats"].get("losses", 0)
            features["away_team_win_pct"] = game_data["away_team_stats"].get(
                "win_percentage", 0.0
            )

        # Recent form features
        if "home_team_recent_form" in game_data:
            features["home_team_recent_form"] = game_data["home_team_recent_form"]

        if "away_team_recent_form" in game_data:
            features["away_team_recent_form"] = game_data["away_team_recent_form"]

        # Market features
        features["total_odds"] = game_data.get("total_odds", 0.0)
        features["spread"] = game_data.get("spread", 0.0)

        return features

    async def _log_learning_insights(self, insights: dict[str, Any]):
        """
        Log learning insights for monitoring and debugging.
        """
        logger.info("📊 Learning Insights Summary:")
        logger.info(f"  Total Predictions: {insights.get('total_predictions', 0)}")
        logger.info(f"  Active Patterns: {insights.get('active_patterns', 0)}")
        logger.info(f"  Recent Accuracy: {insights.get('recent_accuracy', 0.0):.2%}")

        # Log top patterns
        for i, pattern in enumerate(insights.get("top_patterns", [])[:3]):
            logger.info(
                f"  Top Pattern {i+1}: {pattern['description']} "
                f"(Success: {pattern['success_rate']:.2%}, Strength: {pattern['strength']:.2f})"
            )

        # Log model performance
        for model_name, performance in insights.get("model_performance", {}).items():
            logger.info(
                f"  Model {model_name}: {performance['accuracy']:.2%} accuracy, "
                f"${performance['total_profit']:.2f} profit"
            )

        # Log recommendations
        for recommendation in insights.get("learning_recommendations", []):
            logger.info(f"  💡 Recommendation: {recommendation}")


class EnhancedBettingSystem:
    """
    Enhanced betting system that integrates learning capabilities.
    """

    def __init__(self, learning_system: SelfLearningFeedbackSystem):
        self.learning_system = learning_system
        self.integrator = LearningSystemIntegrator(learning_system)

        # Initialize existing system components if available
        self.existing_system = None
        if UltimateIntegratedBettingSystem:
            try:
                self.existing_system = UltimateIntegratedBettingSystem(bankroll=1000.0)
                logger.info(
                    "✅ Integrated with existing UltimateIntegratedBettingSystem"
                )
            except Exception as e:
                logger.warning(f"Could not initialize existing system: {e}")

        logger.info("🚀 Enhanced Betting System initialized with learning capabilities")

    async def run_enhanced_analysis(self) -> dict[str, Any]:
        """
        Run enhanced analysis with learning integration.
        """
        logger.info("🎯 Starting enhanced analysis with learning...")

        try:
            # Run learning analysis first
            await self.integrator.run_learning_analysis()

            # Get learning recommendations
            recommendations = await self.integrator.get_learning_recommendations()

            # Run existing analysis if available
            if self.existing_system:
                base_results = await self.existing_system.run_ultimate_analysis(
                    "baseball_mlb"
                )
            else:
                base_results = {"status": "no_existing_system", "recommendations": []}

            # Enhance predictions with learning
            enhanced_recommendations = []
            for rec in base_results.get("recommendations", []):
                enhanced_rec = await self.integrator.enhance_prediction_with_learning(
                    rec,
                    rec.get("game_data", {}),
                    rec.get("model_name", "ensemble_model"),
                )
                enhanced_recommendations.append(enhanced_rec)

                # Record for learning
                await self.integrator.record_prediction_for_learning(
                    enhanced_rec, rec.get("game_data", {})
                )

            # Compile enhanced results
            enhanced_results = {
                **base_results,
                "recommendations": enhanced_recommendations,
                "learning_insights": recommendations,
                "enhancement_applied": True,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"✅ Enhanced analysis complete: {len(enhanced_recommendations)} recommendations"
            )
            return enhanced_results

        except Exception as e:
            logger.error(f"❌ Enhanced analysis failed: {e}")
            return {"error": str(e), "status": "failed"}

    async def process_game_outcomes(self, outcomes: list[dict[str, Any]]):
        """
        Process game outcomes to update learning system.
        """
        logger.info(f"📊 Processing {len(outcomes)} game outcomes...")

        for outcome in outcomes:
            prediction_id = outcome.get("prediction_id")
            actual_winner = outcome.get("actual_winner")
            actual_profit = outcome.get("actual_profit", 0.0)

            if prediction_id and actual_winner:
                await self.integrator.update_outcome_for_learning(
                    prediction_id, actual_winner, actual_profit
                )

        # Run learning analysis after processing outcomes
        await self.integrator.run_learning_analysis(force=True)
        logger.info("✅ Game outcomes processed and learning updated")


# Example usage
async def main():
    """Example of how to use the enhanced betting system with learning."""

    # Initialize learning system
    learning_system = SelfLearningFeedbackSystem()

    # Initialize enhanced betting system
    enhanced_system = EnhancedBettingSystem(learning_system)

    # Run enhanced analysis
    results = await enhanced_system.run_enhanced_analysis()

    print("Enhanced Analysis Results:")
    print(json.dumps(results, indent=2, default=str))

    # Example: Process some outcomes
    example_outcomes = [
        {
            "prediction_id": "pred_20250101_120000_123456",
            "actual_winner": "Yankees",
            "actual_profit": 85.0,
        }
    ]

    await enhanced_system.process_game_outcomes(example_outcomes)


if __name__ == "__main__":
    asyncio.run(main())
