#!/usr/bin/env python3
"""
Integration Guide: Adding Self-Learning to Your MLB Betting System
==================================================================
This script shows exactly how to add self-learning capabilities to your existing system
with minimal code changes.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

# Import the simple learning system
from simple_learning_integration import (
    SimpleLearningTracker,
    add_learning_to_prediction,
    record_prediction_for_learning,
    update_outcome_for_learning,
)

logger = logging.getLogger(__name__)

# Global learning tracker (initialize once)
LEARNING_TRACKER = SimpleLearningTracker()


class EnhancedBettingSystem:
    """
    Enhanced version of your existing betting system with learning capabilities.
    This shows how to modify your existing system with minimal changes.
    """

    def __init__(self, bankroll: float = 1000.0):
        self.bankroll = bankroll
        self.learning_tracker = LEARNING_TRACKER

        # Your existing initialization code would go here
        logger.info("🚀 Enhanced Betting System with Learning initialized")

    async def analyze_games_with_learning(
        self, games: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Analyze games with learning enhancement.
        This replaces your existing analyze_games method.
        """
        enhanced_recommendations = []

        for game in games:
            # Your existing analysis logic here
            base_prediction = await self._analyze_single_game(game)

            if base_prediction:
                # ENHANCEMENT: Add learning to the prediction
                enhanced_prediction = add_learning_to_prediction(
                    base_prediction, game, self.learning_tracker
                )

                # ENHANCEMENT: Record for learning
                prediction_id = record_prediction_for_learning(
                    enhanced_prediction, self.learning_tracker
                )
                enhanced_prediction["learning_prediction_id"] = prediction_id

                enhanced_recommendations.append(enhanced_prediction)

        return enhanced_recommendations

    async def _analyze_single_game(self, game: dict[str, Any]) -> dict[str, Any]:
        """
        Your existing single game analysis logic.
        This is where your current AI/ML analysis happens.
        """
        # This is a placeholder - replace with your actual analysis logic
        return {
            "game_id": game.get("game_id"),
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "predicted_winner": game.get("home_team"),  # Placeholder
            "confidence": 0.7,  # Placeholder
            "stake": 50.0,  # Placeholder
            "odds": game.get("odds", 2.0),
            "model_name": "ensemble_model",
            "features": {
                "home_advantage": True,
                "recent_form": "good",
                "weather": "clear",
            },
        }

    async def process_game_outcomes(self, outcomes: list[dict[str, Any]]):
        """
        Process game outcomes to update learning system.
        Call this after games are completed.
        """
        logger.info(f"📊 Processing {len(outcomes)} game outcomes for learning...")

        for outcome in outcomes:
            prediction_id = outcome.get("learning_prediction_id")
            actual_winner = outcome.get("actual_winner")
            profit = outcome.get("profit", 0.0)

            if prediction_id and actual_winner:
                update_outcome_for_learning(
                    prediction_id, actual_winner, profit, self.learning_tracker
                )

        # Run learning analysis
        self.learning_tracker.analyze_and_learn()

        # Get and log insights
        insights = self.learning_tracker.get_insights()
        logger.info(
            f"📈 Learning Insights: {insights['recent_accuracy']:.2%} recent accuracy, "
            f"{insights['active_patterns']} active patterns"
        )

    def get_learning_recommendations(self) -> dict[str, Any]:
        """Get recommendations based on learning insights."""
        insights = self.learning_tracker.get_insights()

        recommendations = {
            "system_status": "learning_active",
            "recent_performance": insights["recent_accuracy"],
            "recommendations": [],
        }

        # Generate recommendations based on performance
        if insights["recent_accuracy"] < 0.5:
            recommendations["recommendations"].append(
                "Consider reducing confidence thresholds"
            )
        elif insights["recent_accuracy"] > 0.7:
            recommendations["recommendations"].append(
                "System performing well - consider increasing stakes"
            )

        # Model-specific recommendations
        for model_name, performance in insights["model_performance"].items():
            if performance["accuracy"] < 0.5:
                recommendations["recommendations"].append(
                    f"Reduce reliance on {model_name} model"
                )
            elif performance["accuracy"] > 0.7:
                recommendations["recommendations"].append(
                    f"Increase weight for {model_name} model"
                )

        return recommendations


# Integration examples for different parts of your system


def integrate_with_odds_fetcher():
    """
    Example: How to integrate learning with your odds fetcher.
    """

    async def enhanced_fetch_and_analyze():
        # Your existing odds fetching code
        games = await fetch_odds_from_api()  # Your existing function

        # Create enhanced system
        enhanced_system = EnhancedBettingSystem()

        # Analyze with learning
        recommendations = await enhanced_system.analyze_games_with_learning(games)

        return recommendations

    return enhanced_fetch_and_analyze


def integrate_with_daily_workflow():
    """
    Example: How to integrate learning with your daily workflow.
    """

    async def enhanced_daily_workflow():
        # Your existing daily workflow
        enhanced_system = EnhancedBettingSystem()

        # Morning: Get predictions with learning
        games = await fetch_todays_games()  # Your existing function
        recommendations = await enhanced_system.analyze_games_with_learning(games)

        # Save recommendations
        save_recommendations(recommendations)  # Your existing function

        # Evening: Process outcomes
        outcomes = await get_game_outcomes()  # Your existing function
        await enhanced_system.process_game_outcomes(outcomes)

        # Get learning insights
        insights = enhanced_system.get_learning_recommendations()
        log_insights(insights)  # Your existing logging

        return recommendations

    return enhanced_daily_workflow


def integrate_with_existing_main():
    """
    Example: How to modify your existing main function.
    """

    async def enhanced_main():
        # Initialize learning tracker (do this once at startup)
        global LEARNING_TRACKER
        LEARNING_TRACKER = SimpleLearningTracker()

        # Your existing initialization
        system = EnhancedBettingSystem(bankroll=1000.0)

        try:
            # Your existing analysis pipeline
            results = await system.analyze_games_with_learning(games)

            # Your existing output/saving
            save_results(results)

            # NEW: Get learning insights
            insights = system.get_learning_recommendations()
            print(f"Learning Insights: {insights}")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")

        return results

    return enhanced_main


# Minimal integration example - just add these lines to your existing code


def minimal_integration_example():
    """
    Minimal integration - just add these few lines to your existing code.
    """

    # 1. At the top of your main file, add:
    # from simple_learning_integration import SimpleLearningTracker, add_learning_to_prediction

    # 2. Initialize once at startup:
    # learning_tracker = SimpleLearningTracker()

    # 3. Enhance each prediction:
    # enhanced_prediction = add_learning_to_prediction(base_prediction, game_data, learning_tracker)

    # 4. Record for learning:
    # prediction_id = record_prediction_for_learning(enhanced_prediction, learning_tracker)

    # 5. After games complete, update outcomes:
    # update_outcome_for_learning(prediction_id, actual_winner, profit, learning_tracker)

    # 6. Periodically run learning analysis:
    # learning_tracker.analyze_and_learn()

    pass


# Example of how to modify your existing gold_standard_main.py


def modify_gold_standard_main():
    """
    Example modifications for your gold_standard_main.py file.
    """

    # Add these imports at the top:
    # from simple_learning_integration import SimpleLearningTracker, add_learning_to_prediction, record_prediction_for_learning, update_outcome_for_learning

    # Add this after your existing imports:
    # learning_tracker = SimpleLearningTracker()

    # Modify your analyze_opportunities_concurrently method:
    """
    async def analyze_opportunities_concurrently(self, opportunities):
        # Your existing analysis code...

        for opportunity in opportunities:
            # Your existing analysis...
            base_prediction = self._analyze_opportunity(opportunity)

            # ADD THESE LINES:
            enhanced_prediction = add_learning_to_prediction(
                base_prediction, opportunity, learning_tracker
            )
            prediction_id = record_prediction_for_learning(
                enhanced_prediction, learning_tracker
            )
            enhanced_prediction['learning_id'] = prediction_id

            # Use enhanced_prediction instead of base_prediction
            results.append(enhanced_prediction)

        return results
    """

    # Add this method to your class:
    """
    async def process_outcomes_for_learning(self, outcomes):
        for outcome in outcomes:
            prediction_id = outcome.get('learning_id')
            if prediction_id:
                update_outcome_for_learning(
                    prediction_id,
                    outcome['actual_winner'],
                    outcome['profit'],
                    learning_tracker
                )

        learning_tracker.analyze_and_learn()
    """

    pass


# Quick start function
async def quick_start_learning():
    """
    Quick start function to add learning to your system.
    """
    print("🚀 Quick Start: Adding Self-Learning to Your MLB Betting System")
    print("=" * 60)

    # Initialize learning system
    tracker = SimpleLearningTracker()

    # Example: Simulate some predictions and outcomes
    print("\n📝 Recording sample predictions...")

    sample_predictions = [
        {
            "game_id": "game_1",
            "home_team": "Yankees",
            "away_team": "Red Sox",
            "predicted_winner": "Yankees",
            "confidence": 0.75,
            "stake": 100.0,
            "odds": 1.85,
            "model_name": "ensemble_model",
            "features": {"home_advantage": True},
        },
        {
            "game_id": "game_2",
            "home_team": "Dodgers",
            "away_team": "Giants",
            "predicted_winner": "Dodgers",
            "confidence": 0.65,
            "stake": 50.0,
            "odds": 2.10,
            "model_name": "ensemble_model",
            "features": {"home_advantage": True},
        },
    ]

    prediction_ids = []
    for pred in sample_predictions:
        pred_id = record_prediction_for_learning(pred, tracker)
        prediction_ids.append(pred_id)

        # Enhance prediction
        game_data = {"game_id": pred["game_id"], "odds": pred["odds"]}
        enhanced = add_learning_to_prediction(pred, game_data, tracker)
        print(
            f"  Enhanced confidence: {pred['confidence']:.3f} → {enhanced['confidence']:.3f}"
        )

    # Simulate outcomes
    print("\n✅ Processing outcomes...")
    outcomes = [("Yankees", 85.0), ("Giants", -50.0)]  # Win  # Loss

    for pred_id, (winner, profit) in zip(prediction_ids, outcomes):
        update_outcome_for_learning(pred_id, winner, profit, tracker)

    # Run learning analysis
    print("\n🧠 Running learning analysis...")
    tracker.analyze_and_learn()

    # Get insights
    insights = tracker.get_insights()
    print(f"\n📊 Learning Insights:")
    print(f"  Recent Accuracy: {insights['recent_accuracy']:.2%}")
    print(f"  Total Predictions: {insights['total_predictions']}")
    print(f"  Active Patterns: {insights['active_patterns']}")

    print("\n🎉 Learning system is now active!")
    print("\nNext steps:")
    print("1. Add the integration code to your existing system")
    print("2. Replace your prediction generation with enhanced versions")
    print("3. Add outcome processing after games complete")
    print("4. Monitor learning insights to improve performance")


if __name__ == "__main__":
    asyncio.run(quick_start_learning())
