"""
Base Model Ensemble for NFL and NCAA Predictions
Provides common functionality for 12-model ensembles
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.betting_config import (
    MODEL_WEIGHTS,
    get_calibration_fn,
    get_paths,
    BANKROLL_CONFIG
)


class ModelEnsembleError(Exception):
    """Custom exception for model ensemble errors"""
    pass


class ModelEnsemble(ABC):
    """
    Base class for 12-model betting ensembles

    Subclasses must implement:
    - generate_model_predictions(game) -> Dict[str, float]
    """

    def __init__(self, sport='nfl', week=None):
        """
        Initialize model ensemble

        Args:
            sport: 'nfl' or 'ncaa'
            week: Week number (for calibration adjustments)
        """
        self.sport = sport
        self.week = week or 11
        self.model_weights = MODEL_WEIGHTS.copy()
        self.calibrate_fn = get_calibration_fn(sport)
        self.paths = get_paths(sport, week)

        # Reasoner (optional - can be set by subclass)
        self.reasoner = None

    @abstractmethod
    def generate_model_predictions(self, game: Dict) -> Dict[str, float]:
        """
        Generate predictions from all 12 models

        Must return dict with keys matching MODEL_WEIGHTS:
        {
            'spread_ensemble': 0.65,
            'total_ensemble': 0.60,
            'moneyline_ensemble': 0.70,
            'rf_model': 0.68,
            'gb_model': 0.67,
            'spread_edges': 0.63,
            'total_edges': 0.58,
            'moneyline_edges': 0.69,
            'market_consensus': 0.62,
            'contrarian': 0.55,
            'referee': 0.64,
            'injury': 0.61
        }
        """
        pass

    def calculate_ensemble_confidence(self, model_preds: Dict[str, float]) -> float:
        """
        Calculate weighted ensemble confidence

        Args:
            model_preds: Dict of model predictions

        Returns:
            Weighted average confidence
        """
        weighted_sum = sum(
            model_preds.get(model, 0.50) * weight
            for model, weight in self.model_weights.items()
        )
        return weighted_sum

    def calibrate_confidence(self, raw_confidence: float) -> float:
        """
        Apply sport-specific calibration curve

        Args:
            raw_confidence: Raw ensemble confidence

        Returns:
            Calibrated confidence
        """
        return self.calibrate_fn(raw_confidence)

    def apply_reasoning(self, game: Dict, model_preds: Dict,
                       calibrated_conf: float) -> Dict:
        """
        Apply meta-reasoning if available

        Args:
            game: Game data
            model_preds: Model predictions
            calibrated_conf: Calibrated confidence

        Returns:
            Dict with final_confidence, reasoning, boost
        """
        if not self.reasoner:
            return {
                'final_confidence': calibrated_conf,
                'reasoning': 'No meta-reasoning available - using ensemble only',
                'boost': 0.0
            }

        try:
            result = self.reasoner.reason_about_game(
                game=game,
                model_predictions=model_preds,
                base_confidence=calibrated_conf
            )
            return result
        except Exception as e:
            print(f"⚠️  Reasoning failed: {e}")
            return {
                'final_confidence': calibrated_conf,
                'reasoning': f'Reasoning error: {e}',
                'boost': 0.0
            }

    def determine_pick(self, game: Dict) -> tuple:
        """
        Determine which team to pick based on spread

        Args:
            game: Game data with spread

        Returns:
            (team_name, pick_type) tuple
        """
        spread = game.get('spread', 0)

        if spread > 0:  # Home team is underdog
            return game['away_team'], 'AWAY'
        else:  # Home team is favorite
            return game['home_team'], 'HOME'

    def predict_game(self, game: Dict) -> Dict:
        """
        Generate complete prediction for a game

        Args:
            game: Game data dict

        Returns:
            Prediction dict with all fields
        """
        # Generate model predictions
        model_preds = self.generate_model_predictions(game)

        # Calculate ensemble confidence
        raw_confidence = self.calculate_ensemble_confidence(model_preds)

        # Calibrate
        calibrated_conf = self.calibrate_confidence(raw_confidence)

        # Apply reasoning
        reasoning_result = self.apply_reasoning(game, model_preds, calibrated_conf)
        final_confidence = reasoning_result.get('final_confidence', calibrated_conf)

        # Calculate edge
        edge = final_confidence - 0.50

        # Determine pick
        pick, pick_type = self.determine_pick(game)

        # Build prediction
        return {
            'game': f"{game['away_team']} @ {game['home_team']}",
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'predicted_winner': pick,
            'pick_type': pick_type,
            'spread_line': game.get('spread'),
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
            'week': self.week,
            'sport': self.sport,
            'timestamp': datetime.now().isoformat()
        }

    def load_games(self, filepath: Optional[str] = None) -> List[Dict]:
        """
        Load games from JSON file

        Args:
            filepath: Optional custom path

        Returns:
            List of game dicts
        """
        if filepath is None:
            filepath = self.paths['live_games']

        try:
            with open(filepath) as f:
                games = json.load(f)
            print(f"✅ Loaded {len(games)} {self.sport.upper()} games from {filepath}")
            return games
        except FileNotFoundError:
            raise ModelEnsembleError(
                f"Games file not found: {filepath}\n"
                f"Run game fetcher first: python core/game_fetcher.py {self.sport}"
            )
        except json.JSONDecodeError:
            raise ModelEnsembleError(f"Invalid JSON in {filepath}")

    def save_predictions(self, predictions: List[Dict],
                        output_file: Optional[str] = None):
        """
        Save predictions to JSON file

        Args:
            predictions: List of prediction dicts
            output_file: Optional custom path
        """
        if output_file is None:
            output_file = self.paths['predictions']

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)

        print(f"💾 Saved {len(predictions)} predictions to {output_file}")

    def display_predictions(self, predictions: List[Dict]):
        """Display predictions in readable format"""
        if not predictions:
            print("\n⚠️  No predictions to display")
            return

        print("\n" + "="*120)
        print(f"🏈 {self.sport.upper()} 12-MODEL ENSEMBLE PREDICTIONS ({len(predictions)} games)")
        print("="*120)

        for i, pred in enumerate(predictions, 1):
            print(f"\n{i}. {pred['game']}")
            print(f"   Pick: {pred['predicted_winner']} ({pred['pick_type']})")

            spread_str = f"{pred['spread_line']:+.1f}" if pred['spread_line'] else "N/A"
            total_str = f"{pred['total_line']:.1f}" if pred['total_line'] else "N/A"
            print(f"   Spread: {spread_str:<8} Total: {total_str}")

            print(f"   Confidence: {pred['final_confidence']*100:.1f}% "
                  f"(Raw: {pred['raw_confidence']*100:.1f}%)")
            print(f"   Edge: {pred['edge']*100:+.1f}%  |  "
                  f"Boost: {pred['confidence_boost']*100:+.1f}%")

            if pred.get('reasoning'):
                reasoning = pred['reasoning'][:100]
                if len(pred['reasoning']) > 100:
                    reasoning += "..."
                print(f"   Reasoning: {reasoning}")

            print(f"   Kickoff: {pred.get('kickoff_time', 'TBD')}")

    def filter_by_min_confidence(self, predictions: List[Dict],
                                 min_confidence: Optional[float] = None) -> List[Dict]:
        """
        Filter predictions by minimum confidence

        Args:
            predictions: List of predictions
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list
        """
        if min_confidence is None:
            min_confidence = BANKROLL_CONFIG['min_confidence']

        filtered = [
            p for p in predictions
            if p['final_confidence'] >= min_confidence
        ]

        print(f"✅ {len(filtered)}/{len(predictions)} games above {min_confidence*100:.0f}% confidence")

        return filtered

    def run_predictions(self, games: Optional[List[Dict]] = None,
                       save=True, display=True) -> List[Dict]:
        """
        Complete prediction workflow

        Args:
            games: Optional list of games (otherwise loads from file)
            save: Whether to save predictions
            display: Whether to display predictions

        Returns:
            List of predictions
        """
        # Load games if not provided
        if games is None:
            games = self.load_games()

        # Generate predictions
        print(f"\n🔮 Generating {self.sport.upper()} predictions...")
        predictions = []

        for game in games:
            try:
                pred = self.predict_game(game)
                predictions.append(pred)
            except Exception as e:
                print(f"⚠️  Error predicting {game.get('home_team', 'Unknown')}: {e}")

        # Filter by minimum confidence
        predictions = self.filter_by_min_confidence(predictions)

        # Display
        if display:
            self.display_predictions(predictions)

        # Save
        if save:
            self.save_predictions(predictions)

        return predictions
