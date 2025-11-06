#!/usr/bin/env python3
"""
Learning System Integration
===========================
Integrates the self-learning system with n8n workflows and existing betting components.
This module provides the bridge between your n8n data collection and the learning system.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl

from self_learning_system import SelfLearningSystem

logger = logging.getLogger(__name__)


class LearningIntegration:
    """Integrates self-learning system with n8n workflows and external data sources."""

    def __init__(self, learning_system: SelfLearningSystem):
        self.learning_system = learning_system
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    async def process_n8n_data(self, n8n_output: dict[str, Any]) -> dict[str, Any]:
        """Process data from n8n workflow and feed it to the learning system."""
        logger.info("Processing n8n workflow data for learning system")

        try:
            # Extract game data from n8n output
            games_data = n8n_output.get("games", [])
            odds_data = n8n_output.get("odds", [])
            sentiment_data = n8n_output.get("sentiment", {})
            ai_analysis = n8n_output.get("ai_analysis", {})

            # Process each game
            predictions = []
            for game in games_data:
                # Prepare features for the learning system
                features = self._prepare_game_features(
                    game, odds_data, sentiment_data, ai_analysis
                )

                # Make prediction using learning system
                prediction = self.learning_system.make_prediction(features)

                # Record prediction
                self.learning_system.record_prediction(prediction)

                predictions.append(prediction)

            # Save processed data for later analysis
            self._save_processed_data(n8n_output, predictions)

            return {
                "status": "success",
                "predictions_made": len(predictions),
                "predictions": predictions,
                "learning_metrics": self.learning_system.analyze_performance(),
            }

        except Exception as e:
            logger.error(f"Error processing n8n data: {e}")
            return {"status": "error", "error": str(e)}

    def _prepare_game_features(
        self, game: dict, odds_data: list, sentiment_data: dict, ai_analysis: dict
    ) -> dict[str, Any]:
        """Prepare comprehensive features for a game."""
        game_id = game.get("game_id", "unknown")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        # Find odds for this game
        game_odds = next(
            (odds for odds in odds_data if odds.get("game_id") == game_id), {}
        )

        # Get sentiment data
        home_sentiment = sentiment_data.get(home_team, 0.5)
        away_sentiment = sentiment_data.get(away_team, 0.5)

        # Get AI analysis insights
        ai_insights = ai_analysis.get(game_id, {})

        # Build comprehensive feature set
        features = {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "date": game.get("date", datetime.now().strftime("%Y-%m-%d")),
            # Basic game info
            "home_team_win_pct": game.get("home_win_pct", 0.5),
            "away_team_win_pct": game.get("away_win_pct", 0.5),
            "home_team_runs_per_game": game.get("home_runs_per_game", 4.5),
            "away_team_runs_per_game": game.get("away_runs_per_game", 4.5),
            "home_team_era": game.get("home_era", 4.0),
            "away_team_era": game.get("away_era", 4.0),
            # Odds and betting data
            "home_odds": game_odds.get("home_odds", 2.0),
            "away_odds": game_odds.get("away_odds", 2.0),
            "home_implied_prob": 1 / game_odds.get("home_odds", 2.0),
            "away_implied_prob": 1 / game_odds.get("away_odds", 2.0),
            # Sentiment data
            "sentiment_home": home_sentiment,
            "sentiment_away": away_sentiment,
            "sentiment_difference": home_sentiment - away_sentiment,
            # AI analysis features
            "ai_confidence": ai_insights.get("confidence", 0.5),
            "ai_edge": ai_insights.get("edge", 0.0),
            "ai_recommendation": ai_insights.get("recommendation", "neutral"),
            # Calculated features
            "edge_home": 0.0,  # Will be calculated by learning system
            "edge_away": 0.0,  # Will be calculated by learning system
            # Additional context
            "weather_temp": game.get("weather_temp", 70),
            "weather_wind_speed": game.get("weather_wind_speed", 5),
            "rest_days_home": game.get("rest_days_home", 1),
            "rest_days_away": game.get("rest_days_away", 1),
            "is_playoff": game.get("is_playoff", False),
            "is_rivalry": game.get("is_rivalry", False),
        }

        return features

    def _save_processed_data(self, n8n_output: dict, predictions: list[dict]):
        """Save processed data for later analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save n8n output
        n8n_file = self.data_dir / f"n8n_output_{timestamp}.json"
        with open(n8n_file, "w") as f:
            json.dump(n8n_output, f, indent=2)

        # Save predictions
        predictions_file = self.data_dir / f"predictions_{timestamp}.json"
        with open(predictions_file, "w") as f:
            json.dump(predictions, f, indent=2)

        logger.info(f"Saved processed data: {n8n_file}, {predictions_file}")

    async def update_outcomes(self, game_results: list[dict[str, Any]]):
        """Update learning system with actual game outcomes."""
        logger.info(f"Updating outcomes for {len(game_results)} games")

        for result in game_results:
            game_id = result.get("game_id")
            actual_winner = result.get("winner")
            actual_profit = result.get("profit", 0.0)

            if game_id and actual_winner:
                # Find the prediction for this game
                conn = sqlite3.connect(self.learning_system.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT prediction_id FROM predictions
                    WHERE game_id = ? AND actual_winner IS NULL
                    ORDER BY timestamp DESC LIMIT 1
                """,
                    (game_id,),
                )

                result_row = cursor.fetchone()
                if result_row:
                    prediction_id = result_row[0]
                    self.learning_system.record_outcome(
                        prediction_id, actual_winner, actual_profit
                    )
                    logger.info(f"Updated outcome for {game_id}: {actual_winner}")

                conn.close()

        # Check if we should retrain models
        self.learning_system.retrain_with_new_data()

    def load_historical_data_for_training(self) -> pd.DataFrame:
        """Load and prepare historical data for model training."""
        logger.info("Loading historical data for training")

        # Define paths for historical data
        historical_paths = [
            "data/historical_team_data.parquet",
            "data/2024_season_data.parquet",
            "data/2025_first_half_data.parquet",
            "data/mlb_historical_games.parquet",
            "data/team_performance_history.parquet",
        ]

        # Load data using the learning system
        historical_data = self.learning_system.load_historical_data(historical_paths)

        if historical_data.empty:
            logger.warning("No historical data found, creating sample data")
            historical_data = self._create_sample_historical_data()

        return historical_data

    def _create_sample_historical_data(self) -> pd.DataFrame:
        """Create sample historical data for initial training."""
        logger.info("Creating sample historical data for training")

        import random
        from datetime import datetime, timedelta

        # Sample teams
        teams = [
            "NYY",
            "LAD",
            "BOS",
            "HOU",
            "ATL",
            "PHI",
            "SD",
            "NYM",
            "TB",
            "TOR",
            "CHC",
            "CWS",
            "CIN",
            "CLE",
            "COL",
            "DET",
            "KC",
            "LAA",
            "MIA",
            "MIL",
            "MIN",
            "OAK",
            "PIT",
            "SEA",
            "SF",
            "STL",
            "TEX",
            "WAS",
        ]

        # Generate 1000 sample games
        sample_data = []
        start_date = datetime(2024, 3, 28)

        for i in range(1000):
            game_date = start_date + timedelta(days=random.randint(0, 200))

            home_team = random.choice(teams)
            away_team = random.choice([t for t in teams if t != home_team])

            # Generate realistic features
            home_win_pct = random.uniform(0.3, 0.7)
            away_win_pct = random.uniform(0.3, 0.7)

            home_runs_per_game = random.uniform(3.5, 5.5)
            away_runs_per_game = random.uniform(3.5, 5.5)

            home_era = random.uniform(3.0, 5.0)
            away_era = random.uniform(3.0, 5.0)

            home_odds = random.uniform(1.5, 3.0)
            away_odds = random.uniform(1.5, 3.0)

            # Calculate implied probabilities
            home_implied_prob = 1 / home_odds
            away_implied_prob = 1 / away_odds

            # Generate actual outcome
            home_score = random.randint(0, 10)
            away_score = random.randint(0, 10)
            winner = home_team if home_score > away_score else away_team

            sample_data.append(
                {
                    "date": game_date.strftime("%Y-%m-%d"),
                    "game_id": f"sample_{i}",
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_team_win_pct": home_win_pct,
                    "away_team_win_pct": away_win_pct,
                    "home_team_runs_per_game": home_runs_per_game,
                    "away_team_runs_per_game": away_runs_per_game,
                    "home_team_era": home_era,
                    "away_team_era": away_era,
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "home_implied_prob": home_implied_prob,
                    "away_implied_prob": away_implied_prob,
                    "sentiment_home": random.uniform(0.3, 0.8),
                    "sentiment_away": random.uniform(0.3, 0.8),
                    "ai_confidence": random.uniform(0.4, 0.9),
                    "ai_edge": random.uniform(-0.1, 0.1),
                    "weather_temp": random.uniform(60, 85),
                    "weather_wind_speed": random.uniform(0, 15),
                    "rest_days_home": random.randint(0, 3),
                    "rest_days_away": random.randint(0, 3),
                    "is_playoff": False,
                    "is_rivalry": random.choice([True, False]),
                    "home_score": home_score,
                    "away_score": away_score,
                    "winner": winner,
                }
            )

        df = pd.DataFrame(sample_data)

        # Save sample data
        sample_file = self.data_dir / "sample_historical_data.parquet"
        df.to_parquet(sample_file, index=False)
        logger.info(f"Created and saved sample historical data: {sample_file}")

        return df

    def get_enhanced_recommendations(self, n8n_data: dict[str, Any]) -> dict[str, Any]:
        """Get enhanced betting recommendations using the learning system."""
        logger.info("Generating enhanced recommendations")

        # Process n8n data
        result = asyncio.run(self.process_n8n_data(n8n_data))

        if result["status"] == "error":
            return result

        predictions = result["predictions"]

        # Filter high-value predictions
        high_value_predictions = [
            p for p in predictions if p["edge"] > 0.02 and p["confidence"] > 0.6
        ]

        # Sort by expected value
        high_value_predictions.sort(key=lambda x: x["edge"], reverse=True)

        # Get learning insights
        learning_summary = self.learning_system.get_learning_summary()

        return {
            "status": "success",
            "total_predictions": len(predictions),
            "high_value_predictions": high_value_predictions[:5],  # Top 5
            "learning_insights": learning_summary,
            "system_performance": {
                "overall_accuracy": learning_summary["overall_metrics"]["accuracy"],
                "overall_roi": learning_summary["overall_metrics"]["roi"],
                "recent_trend": learning_summary["recent_trend"],
            },
            "recommendations": learning_summary.get("recommendations", []),
        }

    def run_comprehensive_backtest(
        self, start_date: str = "2024-03-28", end_date: str = "2024-10-01"
    ) -> dict[str, Any]:
        """Run comprehensive backtest on historical data."""
        logger.info(f"Running comprehensive backtest from {start_date} to {end_date}")

        # Run backtest
        backtest_results = self.learning_system.run_backtest(start_date, end_date)

        # Add learning system insights
        learning_summary = self.learning_system.get_learning_summary()

        return {
            "backtest_results": backtest_results,
            "learning_insights": learning_summary,
            "recommendations": learning_summary.get("recommendations", []),
            "system_evolution": {
                "total_predictions": learning_summary["overall_metrics"][
                    "total_predictions"
                ],
                "current_accuracy": learning_summary["overall_metrics"]["accuracy"],
                "current_roi": learning_summary["overall_metrics"]["roi"],
                "trend": learning_summary["recent_trend"],
            },
        }


# Example usage
if __name__ == "__main__":
    # Initialize learning system
    learning_system = SelfLearningSystem()

    # Initialize integration
    integration = LearningIntegration(learning_system)

    # Load and train on historical data
    historical_data = integration.load_historical_data_for_training()
    if not historical_data.empty:
        learning_system.train_models(historical_data)

    # Run backtest on 2024 season
    backtest_results = integration.run_comprehensive_backtest()

    print("Comprehensive Backtest Results:")
    print(f"Accuracy: {backtest_results['backtest_results'].get('accuracy', 0):.1%}")
    print(f"ROI: {backtest_results['backtest_results'].get('roi', 0):.1f}%")
    print(
        f"Total Profit: ${backtest_results['backtest_results'].get('total_profit', 0):.2f}"
    )

    print("\nLearning System Insights:")
    print(json.dumps(backtest_results["learning_insights"], indent=2))
