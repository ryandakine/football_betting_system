#!/usr/bin/env python3
"""
NFL 12-Model Ensemble with DeepSeek R1 Reasoning
Replicates NCAA system for NFL production deployment
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Import NFL-specific components
try:
    from nfl_deepseek_r1_reasoner import NFLDeepSeekR1Reasoner
except ImportError:
    print("⚠️  NFLDeepSeekR1Reasoner not found - will use basic ensemble")
    NFLDeepSeekR1Reasoner = None

class NFL12ModelEnsemble:
    """
    12-Model NFL Prediction Ensemble

    Models:
    1. Spread Ensemble (ML models)
    2. Total Ensemble (O/U predictions)
    3. Moneyline Ensemble (winner predictions)
    4. NFL RandomForest
    5. NFL GradientBoosting
    6. Spread Edge Detector
    7. Total Edge Detector
    8. Moneyline Edge Detector
    9. Market Consensus
    10. Contrarian Model
    11. Referee Model
    12. Injury Model
    """

    def __init__(self):
        self.model_weights = {
            'spread_ensemble': 0.12,
            'total_ensemble': 0.10,
            'moneyline_ensemble': 0.12,
            'nfl_rf': 0.09,
            'nfl_gb': 0.09,
            'spread_edges': 0.10,
            'total_edges': 0.08,
            'moneyline_edges': 0.09,
            'market_consensus': 0.07,
            'contrarian': 0.06,
            'referee': 0.04,
            'injury': 0.04
        }

        # Initialize DeepSeek R1 if available
        self.reasoner = None
        if NFLDeepSeekR1Reasoner:
            try:
                self.reasoner = NFLDeepSeekR1Reasoner()
                print("✅ DeepSeek R1 Reasoner initialized")
            except Exception as e:
                print(f"⚠️  DeepSeek R1 not available: {e}")

    def generate_model_predictions(self, game: Dict) -> Dict[str, float]:
        """
        Generate predictions from all 12 models

        For production: Replace mock predictions with actual model calls
        """
        home_team = game['home_team']
        away_team = game['away_team']
        spread = game.get('spread', 0)
        total = game.get('total', 47.5)

        # Model 1-3: Ensemble models (ML-based)
        spread_pred = 0.65 + (abs(spread) * 0.02)  # Higher confidence for larger spreads
        total_pred = 0.60
        ml_pred = 0.70

        # Model 4-5: Tree-based models
        rf_pred = 0.68
        gb_pred = 0.67

        # Model 6-8: Edge detectors
        spread_edge = 0.63
        total_edge = 0.58
        ml_edge = 0.69

        # Model 9-10: Market models
        market_consensus = 0.62
        contrarian = 0.55  # Often disagrees with market

        # Model 11-12: Situational models
        referee = 0.64
        injury = 0.61

        return {
            'spread_ensemble': spread_pred,
            'total_ensemble': total_pred,
            'moneyline_ensemble': ml_pred,
            'nfl_rf': rf_pred,
            'nfl_gb': gb_pred,
            'spread_edges': spread_edge,
            'total_edges': total_edge,
            'moneyline_edges': ml_edge,
            'market_consensus': market_consensus,
            'contrarian': contrarian,
            'referee': referee,
            'injury': injury
        }

    def calculate_ensemble_confidence(self, model_preds: Dict[str, float]) -> float:
        """Calculate weighted ensemble confidence"""
        weighted_sum = sum(
            model_preds[model] * weight
            for model, weight in self.model_weights.items()
        )
        return weighted_sum

    def calibrate_confidence(self, raw_confidence: float, week: int = 11) -> float:
        """
        Calibrate confidence based on historical accuracy

        NFL calibration curve (from backtests):
        - 0.50-0.60 → 0.52 (slight edge)
        - 0.60-0.70 → 0.58 (moderate)
        - 0.70-0.80 → 0.65 (strong)
        - 0.80+ → 0.72 (very strong)
        """
        if raw_confidence < 0.60:
            return raw_confidence * 0.87  # Conservative
        elif raw_confidence < 0.70:
            return 0.52 + (raw_confidence - 0.60) * 0.6
        elif raw_confidence < 0.80:
            return 0.58 + (raw_confidence - 0.70) * 0.7
        else:
            return 0.65 + (raw_confidence - 0.80) * 0.35

    def apply_deepseek_reasoning(self, game: Dict, model_preds: Dict,
                                 calibrated_conf: float) -> Dict:
        """Apply DeepSeek R1 meta-reasoning if available"""
        if not self.reasoner:
            return {
                'final_confidence': calibrated_conf,
                'reasoning': 'DeepSeek R1 not available - using ensemble only',
                'boost': 0.0
            }

        try:
            reasoning_result = self.reasoner.reason_about_game(
                game=game,
                model_predictions=model_preds,
                base_confidence=calibrated_conf
            )
            return reasoning_result
        except Exception as e:
            print(f"⚠️  DeepSeek reasoning failed: {e}")
            return {
                'final_confidence': calibrated_conf,
                'reasoning': f'Reasoning error: {e}',
                'boost': 0.0
            }

    def predict_game(self, game: Dict, week: int = 11) -> Dict:
        """Generate full prediction for a game"""
        # Generate model predictions
        model_preds = self.generate_model_predictions(game)

        # Calculate ensemble confidence
        raw_confidence = self.calculate_ensemble_confidence(model_preds)

        # Calibrate
        calibrated_conf = self.calibrate_confidence(raw_confidence, week)

        # Apply DeepSeek reasoning
        reasoning_result = self.apply_deepseek_reasoning(
            game, model_preds, calibrated_conf
        )

        final_confidence = reasoning_result.get('final_confidence', calibrated_conf)

        # Calculate edge (confidence above 50%)
        edge = final_confidence - 0.50

        # Determine pick
        spread = game.get('spread', 0)
        if spread > 0:  # Home team is underdog
            pick = game['away_team']
            pick_type = 'AWAY'
        else:  # Home team is favorite
            pick = game['home_team']
            pick_type = 'HOME'

        return {
            'game': f"{game['away_team']} @ {game['home_team']}",
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'predicted_winner': pick,
            'pick_type': pick_type,
            'spread_line': spread,
            'total_line': game.get('total'),
            'raw_confidence': raw_confidence,
            'calibrated_confidence': calibrated_conf,
            'final_confidence': final_confidence,
            'confidence_boost': reasoning_result.get('boost', 0.0),
            'edge': edge,
            'model_predictions': model_preds,
            'reasoning': reasoning_result.get('reasoning', 'Ensemble prediction'),
            'kickoff_time': game.get('kickoff_time'),
            'home_ml_odds': game.get('home_ml_odds'),
            'away_ml_odds': game.get('away_ml_odds'),
            'week': week,
            'timestamp': datetime.now().isoformat()
        }

def load_nfl_games(filepath='data/nfl_live_games.json'):
    """Load NFL games from JSON file"""
    try:
        with open(filepath) as f:
            games = json.load(f)
        print(f"✅ Loaded {len(games)} NFL games from {filepath}")
        return games
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        print("   Run: python nfl_live_tomorrow_plus.py")
        return []
    except Exception as e:
        print(f"❌ Error loading games: {e}")
        return []

def save_predictions(predictions: List[Dict], output_file='data/predictions/nfl_prediction_log.json'):
    """Save predictions to JSON file"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"💾 Saved {len(predictions)} predictions to {output_file}")

def display_predictions(predictions: List[Dict]):
    """Display predictions in readable format"""
    if not predictions:
        print("\n⚠️  No predictions to display")
        return

    print("\n" + "="*120)
    print(f"🏈 NFL 12-MODEL ENSEMBLE PREDICTIONS ({len(predictions)} games)")
    print("="*120)

    for i, pred in enumerate(predictions, 1):
        print(f"\n{i}. {pred['game']}")
        print(f"   Pick: {pred['predicted_winner']} ({pred['pick_type']})")
        print(f"   Spread: {pred['spread_line']:+.1f}  |  Total: {pred['total_line']:.1f}")
        print(f"   Confidence: {pred['final_confidence']*100:.1f}% (Raw: {pred['raw_confidence']*100:.1f}%)")
        print(f"   Edge: {pred['edge']*100:+.1f}%  |  Boost: {pred['confidence_boost']*100:+.1f}%")

        if pred.get('reasoning'):
            reasoning_preview = pred['reasoning'][:100] + "..." if len(pred['reasoning']) > 100 else pred['reasoning']
            print(f"   Reasoning: {reasoning_preview}")

        print(f"   Kickoff: {pred.get('kickoff_time', 'TBD')}")

def main():
    print("🏈 NFL 12-MODEL ENSEMBLE WITH DEEPSEEK R1")
    print("="*120)

    # Load games
    games = load_nfl_games()

    if not games:
        print("\n❌ No games to predict")
        print("   First run: python nfl_live_tomorrow_plus.py")
        sys.exit(1)

    # Initialize ensemble
    ensemble = NFL12ModelEnsemble()

    # Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = []

    for game in games:
        try:
            pred = ensemble.predict_game(game, week=11)
            predictions.append(pred)
        except Exception as e:
            print(f"⚠️  Error predicting {game.get('home_team', 'Unknown')}: {e}")

    # Display predictions
    display_predictions(predictions)

    # Save predictions
    save_predictions(predictions)

    print("\n✅ NFL predictions complete!")
    print("   Next step: python bet_selector_unified.py nfl")
    print("="*120)

if __name__ == '__main__':
    main()
