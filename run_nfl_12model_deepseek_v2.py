#!/usr/bin/env python3
"""
NFL 12-Model Ensemble with DeepSeek R1 Reasoning (Refactored)
Clean implementation using base classes
"""
import sys
from pathlib import Path
from typing import Dict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.model_ensemble import ModelEnsemble, ModelEnsembleError

# Try to import NFL reasoner (optional)
try:
    from nfl_deepseek_r1_reasoner import NFLDeepSeekR1Reasoner
    NFL_REASONER_AVAILABLE = True
except ImportError:
    NFL_REASONER_AVAILABLE = False
    print("ℹ️  NFLDeepSeekR1Reasoner not available - will use ensemble only")


class NFL12ModelEnsemble(ModelEnsemble):
    """
    NFL-specific 12-model ensemble

    Inherits from ModelEnsemble and only needs to implement:
    - generate_model_predictions()
    """

    def __init__(self, week=11, use_reasoner=True):
        """
        Initialize NFL ensemble

        Args:
            week: NFL week number
            use_reasoner: Whether to use DeepSeek R1 reasoning
        """
        super().__init__(sport='nfl', week=week)

        # Initialize DeepSeek R1 if requested and available
        if use_reasoner and NFL_REASONER_AVAILABLE:
            try:
                self.reasoner = NFLDeepSeekR1Reasoner()
                print("✅ DeepSeek R1 Reasoner initialized")
            except Exception as e:
                print(f"⚠️  DeepSeek R1 initialization failed: {e}")
                self.reasoner = None

    def generate_model_predictions(self, game: Dict) -> Dict[str, float]:
        """
        Generate predictions from all 12 NFL models

        TODO: Replace mock predictions with actual model calls
        For production, this would:
        1. Load trained models from disk
        2. Extract features from game data
        3. Run inference on each model
        4. Return dictionary of predictions

        Args:
            game: Game data dict

        Returns:
            Dict of model predictions
        """
        home_team = game['home_team']
        away_team = game['away_team']
        spread = game.get('spread', 0)
        total = game.get('total', 47.5)

        # Mock predictions (would be replaced by actual model calls)
        # Higher confidence for larger spreads (favorites)
        spread_confidence = min(0.75, 0.60 + (abs(spread) * 0.015))

        return {
            # ML ensemble models
            'spread_ensemble': spread_confidence,
            'total_ensemble': 0.58,
            'moneyline_ensemble': 0.68,

            # Tree-based models
            'rf_model': 0.66,
            'gb_model': 0.65,

            # Edge detectors
            'spread_edges': 0.62,
            'total_edges': 0.56,
            'moneyline_edges': 0.67,

            # Market models
            'market_consensus': 0.61,
            'contrarian': 0.53,  # Often disagrees with market

            # Situational models
            'referee': 0.63,
            'injury': 0.60
        }


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='NFL 12-Model Ensemble Predictions'
    )
    parser.add_argument(
        '--week',
        type=int,
        default=11,
        help='NFL week number (default: 11)'
    )
    parser.add_argument(
        '--no-reasoner',
        action='store_true',
        help='Disable DeepSeek R1 reasoning'
    )
    parser.add_argument(
        '--games-file',
        type=str,
        help='Custom games file path (default: data/nfl_live_games.json)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Custom output file path (default: data/predictions/nfl_prediction_log.json)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Don\'t save predictions to file'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Don\'t display predictions'
    )

    args = parser.parse_args()

    print("🏈 NFL 12-MODEL ENSEMBLE WITH DEEPSEEK R1")
    print("="*120)

    try:
        # Initialize ensemble
        ensemble = NFL12ModelEnsemble(
            week=args.week,
            use_reasoner=not args.no_reasoner
        )

        # Load games
        games = ensemble.load_games(filepath=args.games_file)

        if not games:
            print("\n❌ No games to predict")
            print("   First run: python core/game_fetcher.py nfl")
            sys.exit(1)

        # Run predictions
        predictions = ensemble.run_predictions(
            games=games,
            save=not args.no_save,
            display=not args.no_display
        )

        # Save to custom output if specified
        if args.output_file and not args.no_save:
            ensemble.save_predictions(predictions, output_file=args.output_file)

        # Summary
        print(f"\n✅ NFL predictions complete!")
        print(f"   Games analyzed: {len(games)}")
        print(f"   Predictions generated: {len(predictions)}")
        print(f"   Average confidence: {sum(p['final_confidence'] for p in predictions)/len(predictions)*100:.1f}%")
        print(f"\n   Next step: python bet_selector_unified.py nfl")
        print("="*120)

    except ModelEnsembleError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
