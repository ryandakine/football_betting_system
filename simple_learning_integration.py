#!/usr/bin/env python3
"""
Simple Learning Integration for MLB Betting System
==================================================
A lightweight integration that adds self-learning capabilities to your existing system.
Just import and use - minimal changes to existing code required.
"""

import asyncio
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SimpleLearningTracker:
    """
    Simple learning tracker that can be easily integrated into existing systems.
    """

    def __init__(self, db_path: str = "data/simple_learning.db"):
        self.db_path = db_path
        self._init_database()
        logger.info("🧠 Simple Learning Tracker initialized")

    def _init_database(self):
        """Initialize the learning database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Simple predictions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                game_id TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                predicted_winner TEXT NOT NULL,
                confidence REAL NOT NULL,
                stake REAL NOT NULL,
                odds REAL NOT NULL,
                model_name TEXT NOT NULL,
                features TEXT,
                actual_winner TEXT,
                was_correct INTEGER,
                profit REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Learning patterns table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                description TEXT NOT NULL,
                success_rate REAL NOT NULL,
                sample_size INTEGER NOT NULL,
                strength REAL DEFAULT 1.0,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def record_prediction(self, prediction_data: dict[str, Any]) -> str:
        """Record a prediction for learning."""
        prediction_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions
            (id, timestamp, game_id, home_team, away_team, predicted_winner,
             confidence, stake, odds, model_name, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                prediction_id,
                datetime.now().isoformat(),
                prediction_data.get("game_id", ""),
                prediction_data.get("home_team", ""),
                prediction_data.get("away_team", ""),
                prediction_data.get("predicted_winner", ""),
                prediction_data.get("confidence", 0.5),
                prediction_data.get("stake", 0.0),
                prediction_data.get("odds", 0.0),
                prediction_data.get("model_name", "unknown"),
                json.dumps(prediction_data.get("features", {})),
            ),
        )

        conn.commit()
        conn.close()

        logger.info(f"📝 Recorded prediction {prediction_id}")
        return prediction_id

    def update_outcome(self, prediction_id: str, actual_winner: str, profit: float):
        """Update a prediction with its outcome."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get the original prediction
        cursor.execute(
            "SELECT predicted_winner FROM predictions WHERE id = ?", (prediction_id,)
        )
        result = cursor.fetchone()

        if result:
            predicted_winner = result[0]
            was_correct = predicted_winner == actual_winner

            cursor.execute(
                """
                UPDATE predictions
                SET actual_winner = ?, was_correct = ?, profit = ?
                WHERE id = ?
            """,
                (actual_winner, was_correct, profit, prediction_id),
            )

            conn.commit()
            logger.info(
                f"✅ Updated outcome for {prediction_id}: {'WIN' if was_correct else 'LOSS'}"
            )

        conn.close()

    def analyze_and_learn(self):
        """Analyze recent predictions and identify patterns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent predictions (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()

        cursor.execute(
            """
            SELECT * FROM predictions
            WHERE timestamp >= ? AND was_correct IS NOT NULL
        """,
            (thirty_days_ago,),
        )

        predictions = cursor.fetchall()

        if len(predictions) < 10:
            logger.info("Not enough predictions for learning analysis")
            conn.close()
            return

        # Analyze model performance
        cursor.execute(
            """
            SELECT model_name,
                   COUNT(*) as total,
                   SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                   AVG(confidence) as avg_confidence,
                   SUM(profit) as total_profit
            FROM predictions
            WHERE timestamp >= ? AND was_correct IS NOT NULL
            GROUP BY model_name
        """,
            (thirty_days_ago,),
        )

        model_performance = cursor.fetchall()

        # Create/update patterns based on performance
        for model_name, total, wins, avg_confidence, total_profit in model_performance:
            success_rate = wins / total if total > 0 else 0

            if success_rate > 0.6:  # Good performing model
                pattern_id = f"model_{model_name}_{datetime.now().strftime('%Y%m%d')}"

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO patterns
                    (id, pattern_type, description, success_rate, sample_size, strength)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        pattern_id,
                        "model_performance",
                        f"{model_name} model performing well",
                        success_rate,
                        total,
                        1.2,  # Boost strength for good performance
                    ),
                )

        # Analyze odds ranges
        cursor.execute(
            """
            SELECT
                CASE
                    WHEN odds < 1.5 THEN 'heavy_favorite'
                    WHEN odds < 2.0 THEN 'favorite'
                    WHEN odds < 2.5 THEN 'slight_favorite'
                    WHEN odds < 3.5 THEN 'even'
                    WHEN odds < 5.0 THEN 'underdog'
                    ELSE 'heavy_underdog'
                END as odds_range,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins
            FROM predictions
            WHERE timestamp >= ? AND was_correct IS NOT NULL
            GROUP BY odds_range
        """,
            (thirty_days_ago,),
        )

        odds_performance = cursor.fetchall()

        for odds_range, total, wins in odds_performance:
            if total >= 5:  # Minimum sample size
                success_rate = wins / total

                if success_rate > 0.55:  # Good performing odds range
                    pattern_id = (
                        f"odds_{odds_range}_{datetime.now().strftime('%Y%m%d')}"
                    )

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO patterns
                        (id, pattern_type, description, success_rate, sample_size, strength)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            pattern_id,
                            "odds_range",
                            f"Good performance for {odds_range} odds",
                            success_rate,
                            total,
                            1.1,
                        ),
                    )

        conn.commit()
        conn.close()

        logger.info(
            f"🧠 Learning analysis complete: {len(predictions)} predictions analyzed"
        )

    def enhance_confidence(
        self, base_confidence: float, game_data: dict[str, Any], model_name: str
    ) -> float:
        """Enhance confidence based on learned patterns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get active patterns
        cursor.execute("SELECT * FROM patterns WHERE is_active = 1")
        patterns = cursor.fetchall()

        enhanced_confidence = base_confidence
        applied_patterns = []

        for pattern in patterns:
            (
                pattern_id,
                pattern_type,
                description,
                success_rate,
                sample_size,
                strength,
            ) = pattern

            # Check if this game matches the pattern
            if self._matches_pattern(game_data, model_name, pattern_type, description):
                # Apply confidence boost
                confidence_boost = (success_rate - 0.5) * strength * 0.1
                enhanced_confidence += confidence_boost
                applied_patterns.append(description)

        # Cap confidence
        enhanced_confidence = min(0.95, max(0.05, enhanced_confidence))

        conn.close()

        if applied_patterns:
            logger.info(
                f"🎯 Enhanced confidence: {base_confidence:.3f} → {enhanced_confidence:.3f} "
                f"(patterns: {len(applied_patterns)})"
            )

        return enhanced_confidence

    def _matches_pattern(
        self,
        game_data: dict[str, Any],
        model_name: str,
        pattern_type: str,
        description: str,
    ) -> bool:
        """Check if game data matches a pattern."""
        if pattern_type == "model_performance":
            return model_name in description

        elif pattern_type == "odds_range":
            odds = game_data.get("odds", 0)
            if "heavy_favorite" in description and odds < 1.5:
                return True
            elif "favorite" in description and 1.5 <= odds < 2.0:
                return True
            elif "slight_favorite" in description and 2.0 <= odds < 2.5:
                return True
            elif "even" in description and 2.5 <= odds < 3.5:
                return True
            elif "underdog" in description and 3.5 <= odds < 5.0:
                return True
            elif "heavy_underdog" in description and odds >= 5.0:
                return True

        return False

    def get_insights(self) -> dict[str, Any]:
        """Get learning insights."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent performance
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()

        cursor.execute(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins
            FROM predictions
            WHERE timestamp >= ? AND was_correct IS NOT NULL
        """,
            (thirty_days_ago,),
        )

        result = cursor.fetchone()
        total_predictions, wins = result if result else (0, 0)
        recent_accuracy = wins / total_predictions if total_predictions > 0 else 0

        # Get active patterns
        cursor.execute("SELECT COUNT(*) FROM patterns WHERE is_active = 1")
        active_patterns = cursor.fetchone()[0]

        # Get model performance
        cursor.execute(
            """
            SELECT model_name,
                   COUNT(*) as total,
                   SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins
            FROM predictions
            WHERE timestamp >= ? AND was_correct IS NOT NULL
            GROUP BY model_name
        """,
            (thirty_days_ago,),
        )

        model_performance = {}
        for model_name, total, wins in cursor.fetchall():
            model_performance[model_name] = {
                "accuracy": wins / total if total > 0 else 0,
                "total_predictions": total,
            }

        conn.close()

        return {
            "recent_accuracy": recent_accuracy,
            "total_predictions": total_predictions,
            "active_patterns": active_patterns,
            "model_performance": model_performance,
        }


# Simple integration functions
def add_learning_to_prediction(
    prediction: dict[str, Any],
    game_data: dict[str, Any],
    tracker: SimpleLearningTracker,
) -> dict[str, Any]:
    """Add learning enhancement to a prediction."""
    base_confidence = prediction.get("confidence", 0.5)
    model_name = prediction.get("model_name", "unknown")

    # Enhance confidence using learning
    enhanced_confidence = tracker.enhance_confidence(
        base_confidence, game_data, model_name
    )

    # Create enhanced prediction
    enhanced_prediction = prediction.copy()
    enhanced_prediction["confidence"] = enhanced_confidence
    enhanced_prediction["original_confidence"] = base_confidence
    enhanced_prediction["learning_boost"] = enhanced_confidence - base_confidence

    return enhanced_prediction


def record_prediction_for_learning(
    prediction: dict[str, Any], tracker: SimpleLearningTracker
) -> str:
    """Record a prediction for learning."""
    return tracker.record_prediction(prediction)


def update_outcome_for_learning(
    prediction_id: str,
    actual_winner: str,
    profit: float,
    tracker: SimpleLearningTracker,
):
    """Update a prediction outcome for learning."""
    tracker.update_outcome(prediction_id, actual_winner, profit)


# Example usage
async def example_usage():
    """Example of how to integrate learning into your existing system."""

    # Initialize learning tracker
    tracker = SimpleLearningTracker()

    # Example prediction
    prediction = {
        "game_id": "game_123",
        "home_team": "Yankees",
        "away_team": "Red Sox",
        "predicted_winner": "Yankees",
        "confidence": 0.75,
        "stake": 100.0,
        "odds": 1.85,
        "model_name": "ensemble_model",
        "features": {"home_advantage": True, "recent_form": "good"},
    }

    game_data = {
        "game_id": "game_123",
        "home_team": "Yankees",
        "away_team": "Red Sox",
        "odds": 1.85,
    }

    # Record prediction for learning
    prediction_id = record_prediction_for_learning(prediction, tracker)

    # Enhance prediction with learning
    enhanced_prediction = add_learning_to_prediction(prediction, game_data, tracker)

    print(f"Original confidence: {prediction['confidence']}")
    print(f"Enhanced confidence: {enhanced_prediction['confidence']}")
    print(f"Learning boost: {enhanced_prediction['learning_boost']:+.3f}")

    # Later, update with outcome
    update_outcome_for_learning(prediction_id, "Yankees", 85.0, tracker)

    # Run learning analysis
    tracker.analyze_and_learn()

    # Get insights
    insights = tracker.get_insights()
    print(f"Recent accuracy: {insights['recent_accuracy']:.2%}")
    print(f"Active patterns: {insights['active_patterns']}")


if __name__ == "__main__":
    asyncio.run(example_usage())
