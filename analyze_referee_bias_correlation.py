#!/usr/bin/env python3
"""
Analyze referee bias patterns and correlate with betting predictions.
Detects how referee decisions impact game outcomes and prediction accuracy.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from statistics import mean, stdev
import numpy as np

from unified_betting_intelligence import NarrativeIntegratedAICouncil
from betting_types import RiskLevel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/historical_games")


class RefereeBiasAnalyzer:
    """Analyze referee bias patterns and correlations."""

    def __init__(self):
        self.council = NarrativeIntegratedAICouncil()
        self.referee_stats = defaultdict(lambda: {
            "games": 0,
            "home_wins": 0,
            "away_wins": 0,
            "home_win_pct": 0.0,
            "total_points_per_game": [],
            "spread_results": [],
            "predictions_correct": 0,
            "predictions_total": 0,
        })

    def load_games(self) -> List[Dict[str, Any]]:
        """Load all historical games."""
        merged_file = DATA_DIR / "nfl_games_2015_2025.json"
        
        if not merged_file.exists():
            logger.error(f"File not found: {merged_file}")
            return []
        
        with open(merged_file) as f:
            return json.load(f)

    def analyze_referee_patterns(self, games: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze referee statistics and bias patterns."""
        logger.info(f"Analyzing {len(games)} games for referee bias...")
        
        for game in games:
            referee = game.get("referee", "Unknown")
            home_score = game.get("home_score", 0)
            away_score = game.get("away_score", 0)
            total = home_score + away_score
            spread_result = home_score - away_score
            
            stats = self.referee_stats[referee]
            stats["games"] += 1
            stats["total_points_per_game"].append(total)
            stats["spread_results"].append(spread_result)
            
            if home_score > away_score:
                stats["home_wins"] += 1
            else:
                stats["away_wins"] += 1
            
            stats["home_win_pct"] = stats["home_wins"] / stats["games"]
        
        return dict(self.referee_stats)

    def correlate_with_predictions(self, games: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate predictions and correlate with referee data."""
        logger.info("Generating predictions for all games...")
        
        prediction_results = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "by_referee": defaultdict(lambda: {
                "correct": 0,
                "total": 0,
                "accuracy": 0.0,
            }),
        }
        
        for i, game in enumerate(games):
            if i % 500 == 0:
                logger.info(f"Processing game {i+1}/{len(games)}...")
            
            try:
                # Prepare game data for prediction
                game_data = {
                    "game_id": game.get("game_id", ""),
                    "home_team": game.get("home_team", ""),
                    "away_team": game.get("away_team", ""),
                    "spread": 0.0,  # Unknown from historical data
                    "total": float(game.get("total_score", 44.5)),
                    "home_ml_odds": -110,
                    "away_ml_odds": 110,
                    "spread_odds": -110,
                    "total_odds": -110,
                    "spread_model_home_pct": 0.5,
                    "total_model_over_pct": 0.5,
                }
                
                prediction = self.council.make_unified_prediction(game_data)
                
                # Evaluate prediction
                actual_winner = game.get("winner", "")
                predicted_pick = prediction.spread_prediction.pick if prediction.spread_prediction else "home"
                is_correct = (predicted_pick == "home" and actual_winner == game.get("home_team")) or \
                             (predicted_pick == "away" and actual_winner == game.get("away_team"))
                
                prediction_results["total_predictions"] += 1
                if is_correct:
                    prediction_results["correct_predictions"] += 1
                
                # Track by referee
                referee = game.get("referee", "Unknown")
                ref_stats = prediction_results["by_referee"][referee]
                ref_stats["total"] += 1
                if is_correct:
                    ref_stats["correct"] += 1
                ref_stats["accuracy"] = ref_stats["correct"] / ref_stats["total"] if ref_stats["total"] > 0 else 0
                
                # Update referee overall stats with prediction performance
                self.referee_stats[referee]["predictions_correct"] += is_correct
                self.referee_stats[referee]["predictions_total"] += 1
                
            except Exception as e:
                logger.error(f"Failed to process game {game.get('game_id')}: {e}")
        
        return prediction_results

    def detect_bias_correlations(self) -> Dict[str, Any]:
        """Detect correlations between referee bias and prediction accuracy."""
        logger.info("Detecting referee bias correlations...")
        
        correlations = {
            "high_home_bias_referees": [],
            "high_away_bias_referees": [],
            "accurate_referees": [],
            "inaccurate_referees": [],
            "total_referees": len(self.referee_stats),
        }
        
        for referee, stats in sorted(self.referee_stats.items()):
            if stats["games"] < 20:  # Minimum sample size
                continue
            
            home_win_pct = stats["home_win_pct"]
            pred_accuracy = stats["predictions_correct"] / stats["predictions_total"] if stats["predictions_total"] > 0 else 0
            avg_total = mean(stats["total_points_per_game"]) if stats["total_points_per_game"] else 0
            spread_std = stdev(stats["spread_results"]) if len(stats["spread_results"]) > 1 else 0
            
            referee_profile = {
                "name": referee,
                "games": stats["games"],
                "home_win_pct": home_win_pct,
                "prediction_accuracy": pred_accuracy,
                "avg_total_points": avg_total,
                "spread_std": spread_std,
            }
            
            # Categorize
            if home_win_pct > 0.55:
                correlations["high_home_bias_referees"].append(referee_profile)
            elif home_win_pct < 0.45:
                correlations["high_away_bias_referees"].append(referee_profile)
            
            if pred_accuracy > 0.55:
                correlations["accurate_referees"].append(referee_profile)
            elif pred_accuracy < 0.45:
                correlations["inaccurate_referees"].append(referee_profile)
        
        return correlations

    def generate_report(self, games: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        logger.info("\n" + "="*80)
        logger.info("REFEREE BIAS & PREDICTION CORRELATION ANALYSIS")
        logger.info("="*80)
        
        # Step 1: Analyze referee patterns
        ref_patterns = self.analyze_referee_patterns(games)
        
        # Step 2: Correlate with predictions
        pred_results = self.correlate_with_predictions(games)
        
        # Step 3: Detect correlations
        correlations = self.detect_bias_correlations()
        
        # Print summary
        logger.info(f"\nTotal Games Analyzed: {len(games)}")
        logger.info(f"Total Unique Referees: {len(self.referee_stats)}")
        logger.info(f"\nPrediction Performance:")
        logger.info(f"  Overall Accuracy: {pred_results['correct_predictions']}/{pred_results['total_predictions']} " +
                   f"({100*pred_results['correct_predictions']/max(1, pred_results['total_predictions']):.1f}%)")
        
        logger.info(f"\nReferee Bias Patterns:")
        logger.info(f"  High Home Bias (>55% home wins): {len(correlations['high_home_bias_referees'])}")
        logger.info(f"  High Away Bias (<45% home wins): {len(correlations['high_away_bias_referees'])}")
        
        logger.info(f"\nPrediction Accuracy by Referee Bias:")
        if correlations["high_home_bias_referees"]:
            home_bias_acc = mean([r["prediction_accuracy"] for r in correlations["high_home_bias_referees"]])
            logger.info(f"  High Home Bias Referees: {home_bias_acc:.1%} accuracy")
        if correlations["high_away_bias_referees"]:
            away_bias_acc = mean([r["prediction_accuracy"] for r in correlations["high_away_bias_referees"]])
            logger.info(f"  High Away Bias Referees: {away_bias_acc:.1%} accuracy")
        
        return {
            "total_games": len(games),
            "total_referees": len(self.referee_stats),
            "prediction_accuracy": pred_results["correct_predictions"] / max(1, pred_results["total_predictions"]),
            "correlations": correlations,
            "referee_stats": dict(self.referee_stats),
        }


def main():
    analyzer = RefereeBiasAnalyzer()
    games = analyzer.load_games()
    
    if not games:
        logger.error("No games loaded!")
        return
    
    report = analyzer.generate_report(games)
    
    # Save report
    report_file = DATA_DIR / "referee_bias_correlation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\n✅ Report saved to {report_file}")


if __name__ == "__main__":
    main()
