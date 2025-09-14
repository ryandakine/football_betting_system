#!/usr/bin/env python3
"""
Automatic Outcome Tracker for MLB Betting Learning System
=========================================================
Automatically tracks game outcomes and updates the learning system.
Monitors multiple data sources to get final game results.
"""

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
import httpx
import requests

logger = logging.getLogger(__name__)


class AutomaticOutcomeTracker:
    """
    Automatically tracks game outcomes and updates the learning system.
    """

    def __init__(
        self,
        learning_db_path: str = "data/gold_standard_learning.db",
        bridge_url: str = "http://localhost:8767",
    ):
        self.learning_db_path = learning_db_path
        self.bridge_url = bridge_url
        self.tracking_active = False
        self.check_interval = 300  # Check every 5 minutes
        self.last_check = None

        # Data sources for outcomes
        self.outcome_sources = ["mlb_api", "odds_api", "manual_override"]

        logger.info("🎯 Automatic Outcome Tracker initialized")

    async def start_tracking(self):
        """Start automatic outcome tracking."""
        self.tracking_active = True
        logger.info("🚀 Starting automatic outcome tracking...")

        while self.tracking_active:
            try:
                await self.check_for_completed_games()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"❌ Error in outcome tracking loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def stop_tracking(self):
        """Stop automatic outcome tracking."""
        self.tracking_active = False
        logger.info("🛑 Stopped automatic outcome tracking")

    async def check_for_completed_games(self):
        """Check for games that have completed and need outcome updates."""
        logger.debug("🔍 Checking for completed games...")

        try:
            # Get pending predictions from learning database
            pending_predictions = await self.get_pending_predictions()

            if not pending_predictions:
                logger.debug("No pending predictions to check")
                return

            logger.info(
                f"📊 Found {len(pending_predictions)} pending predictions to check"
            )

            # Group predictions by game
            games_to_check = self.group_predictions_by_game(pending_predictions)

            # Check each game for completion
            for game_id, predictions in games_to_check.items():
                await self.check_game_outcome(game_id, predictions)

            self.last_check = datetime.now()

        except Exception as e:
            logger.error(f"❌ Error checking for completed games: {e}")

    async def get_pending_predictions(self) -> list[dict[str, Any]]:
        """Get predictions that don't have outcomes yet."""
        async with aiosqlite.connect(self.learning_db_path) as conn:
            cursor = await conn.execute(
                """
                SELECT id, game_id, home_team, away_team, predicted_winner,
                       confidence, stake, odds, model_name, timestamp
                FROM predictions
                WHERE actual_winner IS NULL
                AND timestamp >= datetime('now', '-7 days')
                ORDER BY timestamp DESC
            """
            )

            rows = await cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "game_id": row[1],
                    "home_team": row[2],
                    "away_team": row[3],
                    "predicted_winner": row[4],
                    "confidence": row[5],
                    "stake": row[6],
                    "odds": row[7],
                    "model_name": row[8],
                    "timestamp": row[9],
                }
                for row in rows
            ]

    def group_predictions_by_game(
        self, predictions: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group predictions by game ID."""
        games = {}
        for pred in predictions:
            game_id = pred["game_id"]
            if game_id not in games:
                games[game_id] = []
            games[game_id].append(pred)
        return games

    async def check_game_outcome(self, game_id: str, predictions: list[dict[str, Any]]):
        """Check if a specific game has completed and get the outcome."""
        try:
            # Try to get outcome from multiple sources
            outcome = None

            for source in self.outcome_sources:
                outcome = await self.get_outcome_from_source(game_id, source)
                if outcome:
                    logger.info(
                        f"✅ Got outcome for game {game_id} from {source}: {outcome['winner']}"
                    )
                    break

            if outcome:
                # Update all predictions for this game
                await self.update_predictions_with_outcome(predictions, outcome)

                # Send update to learning bridge
                await self.send_outcome_to_bridge(predictions, outcome)

            else:
                logger.debug(f"⏳ Game {game_id} not yet completed")

        except Exception as e:
            logger.error(f"❌ Error checking outcome for game {game_id}: {e}")

    async def get_outcome_from_source(
        self, game_id: str, source: str
    ) -> dict[str, Any] | None:
        """Get game outcome from a specific source."""
        try:
            if source == "mlb_api":
                return await self.get_outcome_from_mlb_api(game_id)
            elif source == "odds_api":
                return await self.get_outcome_from_odds_api(game_id)
            elif source == "manual_override":
                return await self.get_outcome_from_manual_override(game_id)
            else:
                return None
        except Exception as e:
            logger.debug(f"Could not get outcome from {source}: {e}")
            return None

    async def get_outcome_from_mlb_api(self, game_id: str) -> dict[str, Any] | None:
        """Get game outcome from MLB API."""
        # This is a placeholder - you would implement actual MLB API calls
        # For now, we'll simulate some outcomes based on game ID

        # Simulate checking MLB API
        await asyncio.sleep(0.1)  # Simulate API call

        # For demonstration, let's simulate some outcomes
        # In reality, you would make actual API calls to MLB stats
        simulated_outcomes = {
            "game_123": {
                "winner": "Yankees",
                "home_score": 5,
                "away_score": 3,
                "status": "final",
            },
            "game_456": {
                "winner": "Red Sox",
                "home_score": 2,
                "away_score": 4,
                "status": "final",
            },
            "game_789": {
                "winner": "Dodgers",
                "home_score": 6,
                "away_score": 1,
                "status": "final",
            },
        }

        if game_id in simulated_outcomes:
            return simulated_outcomes[game_id]

        return None

    async def get_outcome_from_odds_api(self, game_id: str) -> dict[str, Any] | None:
        """Get game outcome from odds API."""
        # This would check if the game is marked as settled in the odds API
        # For now, return None to indicate no outcome available
        return None

    async def get_outcome_from_manual_override(
        self, game_id: str
    ) -> dict[str, Any] | None:
        """Get game outcome from manual override file."""
        override_file = Path("data/manual_outcomes.json")

        if override_file.exists():
            try:
                with open(override_file) as f:
                    overrides = json.load(f)

                if game_id in overrides:
                    outcome = overrides[game_id]
                    logger.info(
                        f"📝 Using manual override for game {game_id}: {outcome['winner']}"
                    )
                    return outcome

            except Exception as e:
                logger.error(f"Error reading manual overrides: {e}")

        return None

    async def update_predictions_with_outcome(
        self, predictions: list[dict[str, Any]], outcome: dict[str, Any]
    ):
        """Update predictions in the learning database with the outcome."""
        async with aiosqlite.connect(self.learning_db_path) as conn:
            for pred in predictions:
                # Calculate profit/loss
                profit = self.calculate_profit(pred, outcome["winner"])

                await conn.execute(
                    """
                    UPDATE predictions
                    SET actual_winner = ?, was_correct = ?, profit = ?, actual_roi = ?
                    WHERE id = ?
                """,
                    (
                        outcome["winner"],
                        1 if pred["predicted_winner"] == outcome["winner"] else 0,
                        profit,
                        profit / pred["stake"] if pred["stake"] > 0 else 0,
                        pred["id"],
                    ),
                )

            await conn.commit()

        logger.info(
            f"✅ Updated {len(predictions)} predictions with outcome: {outcome['winner']}"
        )

    def calculate_profit(self, prediction: dict[str, Any], actual_winner: str) -> float:
        """Calculate profit/loss for a prediction."""
        if prediction["predicted_winner"] == actual_winner:
            # Win: calculate payout based on odds
            odds = prediction["odds"]
            stake = prediction["stake"]
            if odds > 0:
                return (odds - 1) * stake
            else:
                return stake  # Even money
        else:
            # Loss: lose the stake
            return -prediction["stake"]

    async def send_outcome_to_bridge(
        self, predictions: list[dict[str, Any]], outcome: dict[str, Any]
    ):
        """Send outcome update to the learning bridge."""
        try:
            # Prepare outcomes for the bridge
            bridge_outcomes = []
            for pred in predictions:
                profit = self.calculate_profit(pred, outcome["winner"])
                bridge_outcomes.append(
                    {
                        "learning_prediction_id": pred["id"],
                        "actual_winner": outcome["winner"],
                        "profit": profit,
                    }
                )

            # Send to bridge
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.bridge_url}/update-outcomes",
                    json=bridge_outcomes,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(
                        f"📡 Sent {len(bridge_outcomes)} outcomes to bridge: {result.get('outcomes_processed', 0)} processed"
                    )
                else:
                    logger.error(f"❌ Bridge update failed: {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Error sending outcomes to bridge: {e}")

    async def add_manual_outcome(
        self, game_id: str, winner: str, home_score: int = 0, away_score: int = 0
    ):
        """Add a manual outcome override."""
        override_file = Path("data/manual_outcomes.json")
        override_file.parent.mkdir(exist_ok=True)

        # Load existing overrides
        overrides = {}
        if override_file.exists():
            try:
                with open(override_file) as f:
                    overrides = json.load(f)
            except:
                overrides = {}

        # Add new override
        overrides[game_id] = {
            "winner": winner,
            "home_score": home_score,
            "away_score": away_score,
            "status": "final",
            "manual_override": True,
            "timestamp": datetime.now().isoformat(),
        }

        # Save overrides
        with open(override_file, "w") as f:
            json.dump(overrides, f, indent=2)

        logger.info(f"📝 Added manual outcome for game {game_id}: {winner}")

    async def get_tracking_status(self) -> dict[str, Any]:
        """Get current tracking status."""
        try:
            # Get pending predictions count
            async with aiosqlite.connect(self.learning_db_path) as conn:
                cursor = await conn.execute(
                    """
                    SELECT COUNT(*) FROM predictions
                    WHERE actual_winner IS NULL
                    AND timestamp >= datetime('now', '-7 days')
                """
                )
                pending_count = (await cursor.fetchone())[0]

                cursor = await conn.execute(
                    """
                    SELECT COUNT(*) FROM predictions
                    WHERE actual_winner IS NOT NULL
                    AND timestamp >= datetime('now', '-7 days')
                """
                )
                completed_count = (await cursor.fetchone())[0]

            return {
                "tracking_active": self.tracking_active,
                "last_check": self.last_check.isoformat() if self.last_check else None,
                "pending_predictions": pending_count,
                "completed_predictions": completed_count,
                "check_interval_seconds": self.check_interval,
                "outcome_sources": self.outcome_sources,
            }

        except Exception as e:
            logger.error(f"Error getting tracking status: {e}")
            return {"error": str(e)}


# CLI interface for manual operations
async def main():
    """CLI interface for the automatic outcome tracker."""
    import argparse

    parser = argparse.ArgumentParser(description="Automatic Outcome Tracker")
    parser.add_argument("--start", action="store_true", help="Start automatic tracking")
    parser.add_argument("--status", action="store_true", help="Show tracking status")
    parser.add_argument(
        "--add-outcome",
        nargs=3,
        metavar=("GAME_ID", "WINNER", "SCORE"),
        help="Add manual outcome (e.g., game_123 Yankees 5-3)",
    )
    parser.add_argument(
        "--check-now", action="store_true", help="Check for completed games now"
    )

    args = parser.parse_args()

    tracker = AutomaticOutcomeTracker()

    if args.status:
        status = await tracker.get_tracking_status()
        print(json.dumps(status, indent=2))

    elif args.add_outcome:
        game_id, winner, score = args.add_outcome
        home_score, away_score = map(int, score.split("-"))
        await tracker.add_manual_outcome(game_id, winner, home_score, away_score)
        print(
            f"✅ Added manual outcome: {game_id} -> {winner} ({home_score}-{away_score})"
        )

    elif args.check_now:
        await tracker.check_for_completed_games()
        print("✅ Manual check completed")

    elif args.start:
        print("🚀 Starting automatic outcome tracking...")
        print("Press Ctrl+C to stop")
        try:
            await tracker.start_tracking()
        except KeyboardInterrupt:
            await tracker.stop_tracking()
            print("\n🛑 Tracking stopped")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
