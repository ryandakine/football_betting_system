#!/usr/bin/env python3
"""
Enhanced Learning System with Travel and Rest Analysis
======================================================
Integrates travel fatigue and pitcher rest analysis with the self-learning feedback system.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from simple_learning_integration import (
    SimpleLearningTracker,
    add_learning_to_prediction,
    record_prediction_for_learning,
    update_outcome_for_learning,
)
from travel_and_rest_analyzer import TravelAndRestAnalyzer, TravelRestLearningIntegration

logger = logging.getLogger(__name__)


class EnhancedLearningSystem:
    """
    Enhanced learning system that combines pattern learning with travel/rest analysis.
    """

    def __init__(self, learning_db_path: str = "data/enhanced_learning.db"):
        self.learning_tracker = SimpleLearningTracker(db_path=learning_db_path)
        self.travel_analyzer = TravelAndRestAnalyzer()
        self.travel_integration = TravelRestLearningIntegration(self.travel_analyzer)

        logger.info("🧠 Enhanced Learning System initialized with travel/rest analysis")

    async def enhance_prediction_comprehensive(
        self, prediction: dict[str, Any], game_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Enhance a prediction with both learning patterns and travel/rest analysis.
        """
        try:
            # Step 1: Apply basic learning enhancement
            enhanced_prediction = add_learning_to_prediction(
                prediction, game_data, self.learning_tracker
            )

            # Step 2: Apply travel and rest analysis
            final_prediction = (
                await self.travel_integration.enhance_prediction_with_travel_rest(
                    enhanced_prediction, game_data
                )
            )

            # Step 3: Combine the enhancements
            original_confidence = prediction.get("confidence", 0.5)
            learning_boost = enhanced_prediction.get("learning_boost", 0)
            travel_rest_adjustment = final_prediction.get("travel_rest_adjustment", 0)

            final_prediction["total_enhancement"] = (
                learning_boost + travel_rest_adjustment
            )
            final_prediction["enhancement_breakdown"] = {
                "learning_boost": learning_boost,
                "travel_rest_adjustment": travel_rest_adjustment,
                "total_enhancement": learning_boost + travel_rest_adjustment,
            }

            logger.info(
                f"🎯 Comprehensive enhancement: {original_confidence:.3f} → {final_prediction['confidence']:.3f} "
                f"(learning: {learning_boost:+.3f}, travel/rest: {travel_rest_adjustment:+.3f})"
            )

            return final_prediction

        except Exception as e:
            logger.error(f"Error in comprehensive enhancement: {e}")
            return prediction

    async def record_prediction_with_travel_rest(
        self, prediction: dict[str, Any], game_data: dict[str, Any]
    ) -> str:
        """Record a prediction with travel/rest analysis for learning."""
        try:
            # Enhance prediction with travel/rest analysis
            enhanced_prediction = await self.enhance_prediction_comprehensive(
                prediction, game_data
            )

            # Record for learning
            prediction_id = record_prediction_for_learning(
                enhanced_prediction, self.learning_tracker
            )

            # Record travel data if available
            if "travel_rest_analysis" in enhanced_prediction:
                analysis = enhanced_prediction["travel_rest_analysis"]

                # Record team travel if significant
                if analysis.get("away_travel_fatigue", 0) > 0.1:
                    await self.travel_analyzer.record_team_travel(
                        team_name=game_data.get("away_team", ""),
                        from_city=game_data.get("away_team_city", ""),
                        to_city=game_data.get("home_team_city", ""),
                        travel_date=game_data.get(
                            "travel_date", game_data.get("game_date", "")
                        ),
                        game_date=game_data.get("game_date", ""),
                        travel_distance=game_data.get("travel_distance", 0),
                    )

                # Record pitcher starts if available
                if game_data.get("home_pitcher"):
                    await self.travel_analyzer.record_pitcher_start(
                        pitcher_name=game_data["home_pitcher"],
                        team_name=game_data.get("home_team", ""),
                        start_date=game_data.get("game_date", ""),
                    )

                if game_data.get("away_pitcher"):
                    await self.travel_analyzer.record_pitcher_start(
                        pitcher_name=game_data["away_pitcher"],
                        team_name=game_data.get("away_team", ""),
                        start_date=game_data.get("game_date", ""),
                    )

            return prediction_id

        except Exception as e:
            logger.error(f"Error recording prediction with travel/rest: {e}")
            return None

    async def get_comprehensive_insights(self) -> dict[str, Any]:
        """Get comprehensive insights including travel/rest analysis."""
        try:
            # Get basic learning insights
            learning_insights = self.learning_tracker.get_insights()

            # Get travel/rest insights
            travel_rest_insights = await self.travel_analyzer.get_travel_rest_insights()

            # Combine insights
            comprehensive_insights = {
                "learning_system": learning_insights,
                "travel_rest_analysis": travel_rest_insights,
                "system_summary": {
                    "total_predictions": learning_insights.get("total_predictions", 0),
                    "recent_accuracy": learning_insights.get("recent_accuracy", 0.0),
                    "games_with_travel_analysis": travel_rest_insights.get(
                        "total_games_analyzed", 0
                    ),
                    "average_travel_impact": travel_rest_insights.get(
                        "average_travel_rest_impact", 0.0
                    ),
                    "home_advantage_rate": travel_rest_insights.get(
                        "home_advantage_rate", 0.0
                    ),
                    "away_advantage_rate": travel_rest_insights.get(
                        "away_advantage_rate", 0.0
                    ),
                },
            }

            return comprehensive_insights

        except Exception as e:
            logger.error(f"Error getting comprehensive insights: {e}")
            return {"error": str(e)}

    async def analyze_and_learn_enhanced(self):
        """Run enhanced learning analysis including travel/rest patterns."""
        try:
            # Run basic learning analysis
            self.learning_tracker.analyze_and_learn()

            # Get travel/rest insights
            travel_insights = await self.travel_analyzer.get_travel_rest_insights()

            logger.info("🧠 Enhanced learning analysis complete")
            logger.info(
                f"   Travel/rest games analyzed: {travel_insights.get('total_games_analyzed', 0)}"
            )
            logger.info(
                f"   Average travel impact: {travel_insights.get('average_travel_rest_impact', 0.0):.3f}"
            )

        except Exception as e:
            logger.error(f"Error in enhanced learning analysis: {e}")


# Integration with your existing bridge
class EnhancedBridgeIntegration:
    """Integrates enhanced learning with your existing bridge system."""

    def __init__(self, bridge_url: str = "http://localhost:8767"):
        self.bridge_url = bridge_url
        self.enhanced_learning = EnhancedLearningSystem()

    async def enhance_bridge_prediction(
        self, prediction: dict[str, Any], game_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Enhance a bridge prediction with comprehensive analysis."""
        return await self.enhanced_learning.enhance_prediction_comprehensive(
            prediction, game_data
        )

    async def get_enhanced_insights(self) -> dict[str, Any]:
        """Get enhanced insights from the bridge."""
        return await self.enhanced_learning.get_comprehensive_insights()


# Example usage and testing
async def test_enhanced_system():
    """Test the enhanced learning system with travel/rest analysis."""
    print("🧪 Testing Enhanced Learning System with Travel/Rest Analysis")
    print("=" * 70)

    enhanced_system = EnhancedLearningSystem()

    # Test prediction with travel/rest factors
    test_prediction = {
        "game_id": "test_game_123",
        "home_team": "New York Yankees",
        "away_team": "Los Angeles Dodgers",
        "predicted_winner": "New York Yankees",
        "confidence": 0.75,
        "stake": 100.0,
        "odds": 1.85,
        "model_name": "enhanced_ensemble",
        "features": {"test": True},
    }

    test_game_data = {
        "game_id": "test_game_123",
        "home_team": "New York Yankees",
        "away_team": "Los Angeles Dodgers",
        "game_date": "2025-07-21",
        "home_pitcher": "Gerrit Cole",
        "away_pitcher": "Clayton Kershaw",
        "away_team_city": "Los Angeles",
        "home_team_city": "New York",
        "travel_date": "2025-07-20",
        "travel_distance": 2789.0,
    }

    # Record some travel data
    await enhanced_system.travel_analyzer.record_team_travel(
        team_name="Los Angeles Dodgers",
        from_city="Los Angeles",
        to_city="New York",
        travel_date="2025-07-20",
        game_date="2025-07-21",
        travel_distance=2789.0,
    )

    # Record pitcher starts
    await enhanced_system.travel_analyzer.record_pitcher_start(
        "Clayton Kershaw", "Los Angeles Dodgers", "2025-07-18"
    )
    await enhanced_system.travel_analyzer.record_pitcher_start(
        "Gerrit Cole", "New York Yankees", "2025-07-19"
    )

    # Enhance prediction
    enhanced_prediction = await enhanced_system.enhance_prediction_comprehensive(
        test_prediction, test_game_data
    )

    print(f"\n📊 Enhanced Prediction Results:")
    print(f"   Original confidence: {test_prediction['confidence']:.3f}")
    print(f"   Enhanced confidence: {enhanced_prediction['confidence']:.3f}")
    print(
        f"   Total enhancement: {enhanced_prediction.get('total_enhancement', 0):+.3f}"
    )

    breakdown = enhanced_prediction.get("enhancement_breakdown", {})
    print(f"   Learning boost: {breakdown.get('learning_boost', 0):+.3f}")
    print(
        f"   Travel/rest adjustment: {breakdown.get('travel_rest_adjustment', 0):+.3f}"
    )

    # Show travel/rest analysis
    if "travel_rest_analysis" in enhanced_prediction:
        analysis = enhanced_prediction["travel_rest_analysis"]
        print(f"\n✈️ Travel/Rest Analysis:")
        print(
            f"   Away team travel fatigue: {analysis.get('away_travel_fatigue', 0):.3f}"
        )
        print(
            f"   Home team travel fatigue: {analysis.get('home_travel_fatigue', 0):.3f}"
        )
        print(
            f"   Away pitcher rest factor: {analysis.get('away_pitcher_rest_factor', 0):+.3f}"
        )
        print(
            f"   Home pitcher rest factor: {analysis.get('home_pitcher_rest_factor', 0):+.3f}"
        )
        print(
            f"   Overall impact: {analysis.get('overall_travel_rest_impact', 0):+.3f}"
        )
        print(f"   Recommendation: {analysis.get('recommendation', 'N/A')}")

    # Get comprehensive insights
    insights = await enhanced_system.get_comprehensive_insights()
    print(f"\n📈 System Insights:")
    print(
        f"   Total predictions: {insights.get('system_summary', {}).get('total_predictions', 0)}"
    )
    print(
        f"   Recent accuracy: {insights.get('system_summary', {}).get('recent_accuracy', 0):.2%}"
    )
    print(
        f"   Games with travel analysis: {insights.get('system_summary', {}).get('games_with_travel_analysis', 0)}"
    )
    print(
        f"   Average travel impact: {insights.get('system_summary', {}).get('average_travel_impact', 0):.3f}"
    )

    print("\n✅ Enhanced system test complete!")


if __name__ == "__main__":
    asyncio.run(test_enhanced_system())
