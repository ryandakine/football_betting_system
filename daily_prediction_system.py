#!/usr/bin/env python3
"""
Daily Prediction System for MLB Betting
=======================================
Makes predictions on EVERY MLB game each day, regardless of betting value.
This ensures the system is constantly learning and testing its reasoning,
even when there are no good betting opportunities.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from learning_integration import LearningIntegration
from self_learning_system import SelfLearningSystem

logger = logging.getLogger(__name__)


class DailyPredictionSystem:
    """System that makes predictions on every MLB game daily for continuous learning."""

    def __init__(self, learning_system: SelfLearningSystem, odds_api_key: str):
        self.learning_system = learning_system
        self.integration = LearningIntegration(learning_system)
        self.odds_api_key = odds_api_key
        self.prediction_dir = Path("predictions")
        self.prediction_dir.mkdir(exist_ok=True)

    async def get_daily_games(self, date: str = None) -> list[dict[str, Any]]:
        """Get all MLB games for a specific date."""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Fetching games for {date}")

        try:
            # Get games from The Odds API
            url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/scores"
            params = {"apiKey": self.odds_api_key, "date": date}

            response = requests.get(url, params=params)
            response.raise_for_status()

            games = response.json()
            logger.info(f"Found {len(games)} games for {date}")

            return games

        except Exception as e:
            logger.error(f"Error fetching games for {date}: {e}")
            return []

    async def get_game_odds(self, game_id: str) -> dict[str, Any]:
        """Get odds data for a specific game."""
        try:
            url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
            params = {
                "apiKey": self.odds_api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "bookmakers": "fanduel,draftkings,betmgm,caesars,pointsbet",
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            odds_data = response.json()

            # Find the specific game
            for game in odds_data:
                if game.get("id") == game_id:
                    return game

            return {}

        except Exception as e:
            logger.error(f"Error fetching odds for game {game_id}: {e}")
            return {}

    def prepare_game_features(
        self, game: dict[str, Any], odds_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare comprehensive features for a game prediction."""
        game_id = game.get("id", "unknown")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        # Find FanDuel odds
        fanduel_odds = {}
        if odds_data.get("bookmakers"):
            fanduel_book = next(
                (b for b in odds_data["bookmakers"] if b["key"] == "fanduel"), None
            )
            if fanduel_book:
                h2h_market = next(
                    (m for m in fanduel_book["markets"] if m["key"] == "h2h"), None
                )
                if h2h_market:
                    home_outcome = next(
                        (o for o in h2h_market["outcomes"] if o["name"] == home_team),
                        None,
                    )
                    away_outcome = next(
                        (o for o in h2h_market["outcomes"] if o["name"] == away_team),
                        None,
                    )

                    if home_outcome and away_outcome:
                        fanduel_odds = {
                            "home_odds": home_outcome["price"],
                            "away_odds": away_outcome["price"],
                            "home_implied_prob": 1 / home_outcome["price"],
                            "away_implied_prob": 1 / away_outcome["price"],
                        }

        # Calculate basic features (you can expand this with more data sources)
        features = {
            "game_id": game_id,
            "date": (
                game.get("commence_time", "").split("T")[0]
                if game.get("commence_time")
                else datetime.now().strftime("%Y-%m-%d")
            ),
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": game.get("commence_time", ""),
            # Basic team stats (you can enhance this with real team data)
            "home_team_win_pct": 0.5,  # Placeholder - replace with real data
            "away_team_win_pct": 0.5,  # Placeholder - replace with real data
            "home_team_runs_per_game": 4.5,  # Placeholder
            "away_team_runs_per_game": 4.5,  # Placeholder
            "home_team_era": 4.0,  # Placeholder
            "away_team_era": 4.0,  # Placeholder
            # Odds data
            "home_odds": fanduel_odds.get("home_odds", 2.0),
            "away_odds": fanduel_odds.get("away_odds", 2.0),
            "home_implied_prob": fanduel_odds.get("home_implied_prob", 0.5),
            "away_implied_prob": fanduel_odds.get("away_implied_prob", 0.5),
            # Sentiment (placeholder - you can enhance with real sentiment data)
            "sentiment_home": 0.5,
            "sentiment_away": 0.5,
            "sentiment_difference": 0.0,
            # AI analysis (placeholder)
            "ai_confidence": 0.5,
            "ai_edge": 0.0,
            # Context features
            "weather_temp": 70,  # Placeholder
            "weather_wind_speed": 5,  # Placeholder
            "rest_days_home": 1,  # Placeholder
            "rest_days_away": 1,  # Placeholder
            "is_playoff": False,
            "is_rivalry": False,  # You can add logic to detect rivalries
            # Calculated features
            "edge_home": 0.0,  # Will be calculated by learning system
            "edge_away": 0.0,  # Will be calculated by learning system
        }

        return features

    async def make_daily_predictions(self, date: str = None) -> dict[str, Any]:
        """Make predictions on every MLB game for a given date."""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Making predictions for all MLB games on {date}")

        # Get all games for the date
        games = await self.get_daily_games(date)

        if not games:
            logger.warning(f"No games found for {date}")
            return {
                "date": date,
                "total_games": 0,
                "predictions": [],
                "value_bets": [],
                "learning_opportunities": 0,
            }

        predictions = []
        value_bets = []

        for game in games:
            try:
                # Get odds data for the game
                odds_data = await self.get_game_odds(game["id"])

                # Prepare features
                features = self.prepare_game_features(game, odds_data)

                # Make prediction using learning system
                prediction = self.learning_system.make_prediction(features)

                # Record prediction for learning
                self.learning_system.record_prediction(prediction)

                # Add game context to prediction
                prediction["game_context"] = {
                    "commence_time": game.get("commence_time"),
                    "sport_key": game.get("sport_key"),
                    "sport_title": game.get("sport_title"),
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "last_update": game.get("last_update"),
                }

                predictions.append(prediction)

                # Check if this is a value bet
                if prediction["edge"] > 0.02 and prediction["confidence"] > 0.6:
                    value_bets.append(prediction)

                logger.info(
                    f"Prediction for {game['home_team']} vs {game['away_team']}: "
                    f"{prediction['predicted_winner']} (confidence: {prediction['confidence']:.2f}, "
                    f"edge: {prediction['edge']:.3f})"
                )

            except Exception as e:
                logger.error(f"Error making prediction for game {game.get('id')}: {e}")
                continue

        # Save predictions to file
        self._save_daily_predictions(date, predictions)

        # Generate summary
        summary = {
            "date": date,
            "total_games": len(games),
            "predictions_made": len(predictions),
            "value_bets_found": len(value_bets),
            "learning_opportunities": len(predictions),
            "average_confidence": (
                sum(p["confidence"] for p in predictions) / len(predictions)
                if predictions
                else 0
            ),
            "average_edge": (
                sum(p["edge"] for p in predictions) / len(predictions)
                if predictions
                else 0
            ),
            "predictions": predictions,
            "value_bets": value_bets,
        }

        logger.info(f"Daily prediction summary for {date}:")
        logger.info(f"  Total games: {summary['total_games']}")
        logger.info(f"  Predictions made: {summary['predictions_made']}")
        logger.info(f"  Value bets found: {summary['value_bets_found']}")
        logger.info(f"  Learning opportunities: {summary['learning_opportunities']}")

        return summary

    def _save_daily_predictions(self, date: str, predictions: list[dict[str, Any]]):
        """Save daily predictions to file."""
        filename = self.prediction_dir / f"predictions_{date}.json"

        with open(filename, "w") as f:
            json.dump(
                {
                    "date": date,
                    "predictions": predictions,
                    "generated_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved {len(predictions)} predictions to {filename}")

    async def record_daily_outcomes(self, date: str):
        """Record the actual outcomes of games for learning."""
        logger.info(f"Recording outcomes for {date}")

        # Load predictions for the date
        prediction_file = self.prediction_dir / f"predictions_{date}.json"

        if not prediction_file.exists():
            logger.warning(f"No predictions found for {date}")
            return

        with open(prediction_file) as f:
            data = json.load(f)

        predictions = data["predictions"]

        # Get actual results from The Odds API
        try:
            url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/scores"
            params = {"apiKey": self.odds_api_key, "date": date}

            response = requests.get(url, params=params)
            response.raise_for_status()

            results = response.json()

            # Match predictions with results
            outcomes_recorded = 0
            for prediction in predictions:
                game_id = prediction["game_id"]

                # Find matching result
                result = next((r for r in results if r.get("id") == game_id), None)

                if result and result.get("scores"):
                    home_score = result["scores"][0]
                    away_score = result["scores"][1]

                    if home_score is not None and away_score is not None:
                        # Determine winner
                        if home_score > away_score:
                            actual_winner = "home"
                        elif away_score > home_score:
                            actual_winner = "away"
                        else:
                            actual_winner = "tie"

                        # Calculate profit/loss (assuming $100 stake for learning)
                        stake = 100
                        if prediction["predicted_winner"] == actual_winner:
                            profit = stake * (prediction["odds"] - 1)
                        else:
                            profit = -stake

                        # Record outcome for learning
                        self.learning_system.record_outcome(
                            prediction["prediction_id"], actual_winner, profit
                        )

                        outcomes_recorded += 1

                        logger.info(
                            f"Recorded outcome for {prediction['home_team']} vs {prediction['away_team']}: "
                            f"Predicted: {prediction['predicted_winner']}, "
                            f"Actual: {actual_winner}, "
                            f"Profit: ${profit:.2f}"
                        )

            logger.info(f"Recorded {outcomes_recorded} outcomes for {date}")

            # Retrain models if we have enough new data
            if outcomes_recorded > 0:
                self.learning_system.retrain_with_new_data()

        except Exception as e:
            logger.error(f"Error recording outcomes for {date}: {e}")

    def get_prediction_summary(self, date: str = None) -> dict[str, Any]:
        """Get a summary of predictions for a date."""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        prediction_file = self.prediction_dir / f"predictions_{date}.json"

        if not prediction_file.exists():
            return {
                "date": date,
                "status": "no_predictions",
                "message": f"No predictions found for {date}",
            }

        with open(prediction_file) as f:
            data = json.load(f)

        predictions = data["predictions"]

        # Calculate summary statistics
        total_predictions = len(predictions)
        value_bets = [
            p for p in predictions if p["edge"] > 0.02 and p["confidence"] > 0.6
        ]

        confidence_levels = {
            "high": len([p for p in predictions if p["confidence"] >= 0.7]),
            "medium": len([p for p in predictions if 0.6 <= p["confidence"] < 0.7]),
            "low": len([p for p in predictions if p["confidence"] < 0.6]),
        }

        edge_levels = {
            "high_value": len([p for p in predictions if p["edge"] >= 0.05]),
            "medium_value": len([p for p in predictions if 0.02 <= p["edge"] < 0.05]),
            "low_value": len([p for p in predictions if p["edge"] < 0.02]),
        }

        return {
            "date": date,
            "status": "success",
            "total_predictions": total_predictions,
            "value_bets": len(value_bets),
            "confidence_distribution": confidence_levels,
            "edge_distribution": edge_levels,
            "average_confidence": sum(p["confidence"] for p in predictions)
            / total_predictions,
            "average_edge": sum(p["edge"] for p in predictions) / total_predictions,
            "top_value_bets": sorted(value_bets, key=lambda x: x["edge"], reverse=True)[
                :5
            ],
        }

    async def run_daily_cycle(self, date: str = None):
        """Run the complete daily prediction and learning cycle."""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Starting daily prediction cycle for {date}")

        # Step 1: Make predictions on all games
        prediction_summary = await self.make_daily_predictions(date)

        # Step 2: Wait for games to complete (you can adjust this timing)
        logger.info("Waiting for games to complete...")
        await asyncio.sleep(3600)  # Wait 1 hour, adjust as needed

        # Step 3: Record outcomes for learning
        await self.record_daily_outcomes(date)

        # Step 4: Get learning summary
        learning_summary = self.learning_system.get_learning_summary()

        # Step 5: Generate final report
        final_report = {
            "date": date,
            "prediction_summary": prediction_summary,
            "learning_summary": learning_summary,
            "system_improvement": {
                "total_predictions": learning_summary["overall_metrics"][
                    "total_predictions"
                ],
                "current_accuracy": learning_summary["overall_metrics"]["accuracy"],
                "current_roi": learning_summary["overall_metrics"]["roi"],
                "trend": learning_summary["recent_trend"],
            },
        }

        # Save final report
        report_file = self.prediction_dir / f"daily_report_{date}.json"
        with open(report_file, "w") as f:
            json.dump(final_report, f, indent=2)

        logger.info(f"Daily cycle completed for {date}")
        logger.info(f"Final report saved to {report_file}")

        return final_report


# Example usage
if __name__ == "__main__":
    import os

    # Initialize the system
    learning_system = SelfLearningSystem()
    odds_api_key = os.getenv("ODDS_API_KEY", "your_odds_api_key_here")

    daily_system = DailyPredictionSystem(learning_system, odds_api_key)

    # Run daily prediction cycle
    async def main():
        # Make predictions for today
        today = datetime.now().strftime("%Y-%m-%d")

        # Make predictions on all games
        summary = await daily_system.make_daily_predictions(today)

        print(f"Daily Prediction Summary for {today}:")
        print(f"Total games: {summary['total_games']}")
        print(f"Predictions made: {summary['predictions_made']}")
        print(f"Value bets found: {summary['value_bets_found']}")
        print(f"Learning opportunities: {summary['learning_opportunities']}")

        if summary["value_bets"]:
            print("\nTop Value Bets:")
            for bet in summary["value_bets"][:3]:
                print(
                    f"  {bet['home_team']} vs {bet['away_team']}: "
                    f"{bet['predicted_winner']} (confidence: {bet['confidence']:.2f}, "
                    f"edge: {bet['edge']:.3f})"
                )

        # Get prediction summary
        pred_summary = daily_system.get_prediction_summary(today)
        print(f"\nPrediction Summary: {pred_summary}")

    # Run the async function
    asyncio.run(main())
