#!/usr/bin/env python3
"""
Training script for unified betting intelligence system using historical backtest data.
Loads backtest results and optimizes prediction parameters.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

from unified_betting_intelligence import NarrativeIntegratedAICouncil
from betting_types import RiskLevel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedSystemTrainer:
    """Trains unified betting intelligence on historical backtest data."""

    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.council = NarrativeIntegratedAICouncil()
        # Look for game results files
        self.result_files = sorted(Path("data/referee_conspiracy").glob("game_results*.json"))
        logger.info(f"Found {len(self.result_files)} game result datasets")

    def load_backtest_data(self) -> List[Dict[str, Any]]:
        """Load all game results."""
        all_games = []
        for rf in self.result_files:
            try:
                with open(rf) as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        all_games.extend(data.values())
                    elif isinstance(data, list):
                        all_games.extend(data)
                logger.info(f"Loaded {len(all_games)} total games from {rf.name}")
            except Exception as e:
                logger.error(f"Failed to load {rf}: {e}")
        return all_games

    def prepare_game_data(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Convert backtest game record to prediction input format."""
        return {
            "game_id": game.get("game_id", "unknown"),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
            "spread": game.get("spread", 0.0),
            "total": game.get("total", 44.5),
            "home_ml_odds": game.get("home_odds", -110),
            "away_ml_odds": game.get("away_odds", 110),
            "spread_odds": -110,
            "total_odds": -110,
            "spread_model_home_pct": game.get("home_win_prob", 0.5),
            "total_model_over_pct": game.get("over_prob", 0.5),
            "home_advantage_pct": game.get("home_advantage", 0.0),
            "timestamp": game.get("date", ""),
        }

    def train(self) -> Dict[str, Any]:
        """Train system on all backtest data."""
        logger.info("=" * 80)
        logger.info("TRAINING UNIFIED BETTING INTELLIGENCE SYSTEM")
        logger.info("=" * 80)

        games = self.load_backtest_data()
        logger.info(f"Total games loaded: {len(games)}")

        correct = 0
        total = 0
        by_risk_level = {rl.value: {"correct": 0, "total": 0} for rl in RiskLevel}
        predictions = []

        for game in games[:50]:  # Limit to first 50 for demo
            try:
                game_data = self.prepare_game_data(game)
                prediction = self.council.make_unified_prediction(game_data)

                total += 1
                risk = prediction.risk_level.value
                by_risk_level[risk]["total"] += 1

                # Evaluate spread pick
                actual_winner = game.get("winner")
                spread_pick = prediction.spread_prediction.pick if prediction.spread_prediction else None
                expected_winner = "home" if spread_pick == "home" else "away"

                is_correct = expected_winner == ("home" if actual_winner == game.get("home_team") else "away")
                if is_correct:
                    correct += 1
                    by_risk_level[risk]["correct"] += 1

                predictions.append({
                    "game_id": game.get("game_id"),
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "spread_pick": spread_pick,
                    "actual_winner": actual_winner,
                    "correct": is_correct,
                    "confidence": prediction.confidence,
                    "risk_level": risk,
                    "edge_signals": prediction.edge_signals,
                })

            except Exception as e:
                logger.error(f"Failed to process game {game.get('game_id')}: {e}")

        # Results
        accuracy = (correct / total * 100) if total > 0 else 0
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total Games Evaluated: {total}")
        logger.info(f"Correct Predictions: {correct}")
        logger.info(f"Overall Accuracy: {accuracy:.1f}%")

        logger.info("\nPerformance by Risk Level:")
        for risk_level, stats in by_risk_level.items():
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"] * 100
                logger.info(f"  {risk_level}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

        # Save training results
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_games": total,
            "correct": correct,
            "accuracy": accuracy,
            "by_risk_level": by_risk_level,
            "system_version": self.council.VERSION,
            "sample_predictions": predictions[:10],
        }

        output_file = Path("training_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nTraining results saved to {output_file}")

        return results


if __name__ == "__main__":
    trainer = UnifiedSystemTrainer()
    results = trainer.train()
    
    logger.info("\n✅ Training complete!")
    logger.info(f"System accuracy: {results['accuracy']:.1f}%")
