#!/usr/bin/env python3
"""
NCAA Daily Predictions WITH Contrarian Intelligence
====================================================

PRINCIPLE: Automated Constraints (Not Manual Checking)
Contrarian intelligence runs automatically on every prediction

INTEGRATION: Step 2.5 - Contrarian Intelligence
1. Fetch games (Odds API)
2. Generate 12-model predictions
2.5. RUN CONTRARIAN ANALYSIS ← NEW!
3. Display recommendations with contrarian signals

USAGE:
    python ncaa_daily_predictions_with_contrarian.py YOUR_API_KEY

    # Disable contrarian (not recommended)
    python ncaa_daily_predictions_with_contrarian.py YOUR_API_KEY --no-contrarian
"""

import sys
from ncaa_daily_predictions import NCAADailyPredictions
from ncaa_contrarian_intelligence import NCAAContrarianIntelligence, ContrarianSignal
from typing import List, Dict


class NCAADailyPredictionsContrarian(NCAADailyPredictions):
    """
    Extends daily predictions with automatic contrarian intelligence

    WHY: System automatically detects public bias - agent can't forget
    """

    def __init__(self, odds_api_key: str, enable_contrarian: bool = True):
        super().__init__(odds_api_key)
        self.enable_contrarian = enable_contrarian

        if self.enable_contrarian:
            self.contrarian = NCAAContrarianIntelligence(odds_api_key)
            print(f"✅ Contrarian Intelligence enabled")

    def run_daily_analysis(self):
        """
        Run complete daily analysis with contrarian intelligence

        WORKFLOW:
        1. Fetch upcoming games
        2. Generate 12-model predictions
        2.5. Run contrarian analysis (NEW!)
        3. Display recommendations with alerts
        """

        print(f"\n{'='*80}")
        print(f"📊 NCAA DAILY PREDICTIONS - {self._get_today_string()}")
        print(f"{'='*80}")

        # Step 1: Fetch upcoming games
        upcoming_games = self.fetch_upcoming_games()

        if not upcoming_games:
            print(f"\n⚠️  No games in next 48 hours")
            return

        # Step 2: Generate predictions
        year = 2025  # Current season
        predictions = self.generate_predictions_for_games(upcoming_games, year)

        if not predictions:
            print(f"\n⚠️  No predictions generated")
            return

        # Step 2.5: Run contrarian intelligence
        if self.enable_contrarian:
            print(f"\n{'='*80}")
            print(f"🎯 Step 2.5: Fetching contrarian intelligence...")
            print(f"{'='*80}\n")

            predictions_with_contrarian = self._add_contrarian_signals(
                predictions,
                upcoming_games
            )
        else:
            predictions_with_contrarian = predictions

        # Step 3: Display recommendations
        self._display_recommendations_with_contrarian(predictions_with_contrarian)

        # Step 4: Save predictions
        self._save_daily_predictions(predictions_with_contrarian)

    def _add_contrarian_signals(
        self,
        predictions: List[Dict],
        upcoming_games: List[Dict]
    ) -> List[Dict]:
        """
        Add contrarian signals to each prediction

        WHY: System enriches predictions automatically
        """

        enhanced_predictions = []

        for pred in predictions:
            # Find matching game from upcoming_games
            matching_game = None
            for game in upcoming_games:
                if (pred['home_team'] == game['home_team'] and
                    pred['away_team'] == game['away_team']):
                    matching_game = game
                    break

            if not matching_game:
                enhanced_predictions.append(pred)
                continue

            # Fetch line movement
            opening, current = self.contrarian.fetch_line_movement(
                matching_game.get('game_id', '')
            )

            # Build game data for contrarian analysis
            contrarian_game = {
                'home_team': pred['home_team'],
                'away_team': pred['away_team'],
                'opening_spread': opening,
                'current_spread': matching_game['market_spread'],
                'day_of_week': matching_game['day_of_week'],
                'conference': pred.get('conference', '')
            }

            # Analyze
            signal = self.contrarian.analyze_game(
                contrarian_game,
                pred['home_team'],
                pred['away_team']
            )

            # Add signal to prediction
            pred['contrarian_signal'] = signal
            pred['contrarian_strength'] = signal.strength
            pred['contrarian_recommendation'] = signal.recommendation

            enhanced_predictions.append(pred)

        return enhanced_predictions

    def _display_recommendations_with_contrarian(self, predictions: List[Dict]):
        """
        Display predictions with contrarian alerts

        WHY: Context Embedded - shows contrarian signals inline
        """

        print(f"\n{'='*80}")
        print(f"🏈 DAILY PREDICTIONS WITH CONTRARIAN INTELLIGENCE")
        print(f"{'='*80}\n")

        # Group by contrarian strength (high to low)
        strong_contrarian = [p for p in predictions if p.get('contrarian_strength', 0) >= 3]
        regular = [p for p in predictions if p.get('contrarian_strength', 0) < 3]

        # Show strong contrarian signals first
        if strong_contrarian:
            print(f"🚨 STRONG CONTRARIAN SIGNALS (3+ stars)\n")
            for pred in sorted(strong_contrarian, key=lambda x: x.get('contrarian_strength', 0), reverse=True):
                self._print_prediction_with_contrarian(pred, highlight=True)

        # Then regular predictions
        if regular:
            print(f"\n📊 OTHER PREDICTIONS\n")
            for pred in regular:
                self._print_prediction_with_contrarian(pred, highlight=False)

        # Summary
        print(f"\n{'='*80}")
        print(f"📈 SUMMARY")
        print(f"{'='*80}\n")
        print(f"Total predictions: {len(predictions)}")
        print(f"Strong contrarian signals: {len(strong_contrarian)}")
        print(f"Regular predictions: {len(regular)}")
        print()

    def _print_prediction_with_contrarian(self, pred: Dict, highlight: bool = False):
        """Print single prediction with contrarian signal"""

        if highlight:
            print(f"{'='*80}")

        print(f"🏈 {pred['away_team']} @ {pred['home_team']}")
        print(f"   Day: {pred.get('day_of_week', 'Unknown')}")
        print(f"   Time: {pred.get('commence_time', 'TBD')}")

        print(f"\n   12-MODEL PREDICTION:")
        print(f"      Spread: {pred['predicted_spread']:+.1f} (home perspective)")
        print(f"      Market: {pred['market_spread']:+.1f}")
        print(f"      Confidence: {pred['confidence']:.0%}")
        print(f"      Edge: {pred['edge_value']:.1%}")

        # Contrarian signal
        if 'contrarian_signal' in pred:
            signal = pred['contrarian_signal']
            print(f"\n   🎯 CONTRARIAN INTELLIGENCE:")
            print(f"      Strength: {signal}")
            print(f"      Recommendation: {signal.recommendation}")

            if signal.reasons:
                print(f"      Signals:")
                for reason in signal.reasons:
                    print(f"         • {reason}")

            if signal.strength >= 4:
                print(f"\n   🚨 VERY STRONG CONTRARIAN SIGNAL!")
                print(f"      Public heavily on one side - fade opportunity!")
            elif signal.strength >= 3:
                print(f"\n   ⚠️  STRONG CONTRARIAN SIGNAL")
                print(f"      Consider fading the public")

        # Recommendation
        print(f"\n   💡 RECOMMENDED BET:")
        bet_side = pred.get('bet_side', 'N/A')
        bet_amount = pred.get('bet_amount', 0)
        print(f"      {bet_side} for ${bet_amount:.2f}")

        if highlight:
            print(f"{'='*80}")

        print()

    def _get_today_string(self) -> str:
        """Get today's date string"""
        from datetime import datetime
        return datetime.now().strftime('%A, %B %d, %Y')

    def _save_daily_predictions(self, predictions: List[Dict]):
        """Save predictions with contrarian signals"""
        from datetime import datetime
        import json

        date_str = datetime.now().strftime('%Y%m%d')
        output_file = self.predictions_dir / f"predictions_{date_str}_contrarian.json"

        # Convert contrarian signals to dict
        output_preds = []
        for pred in predictions:
            pred_copy = pred.copy()
            if 'contrarian_signal' in pred_copy:
                signal = pred_copy.pop('contrarian_signal')
                pred_copy['contrarian'] = {
                    'strength': signal.strength,
                    'recommendation': signal.recommendation,
                    'public_percentage': signal.public_percentage,
                    'line_movement': signal.line_movement,
                    'sharp_money_detected': signal.sharp_money_detected,
                    'reasons': signal.reasons
                }
            output_preds.append(pred_copy)

        with open(output_file, 'w') as f:
            json.dump(output_preds, f, indent=2, default=str)

        print(f"💾 Saved predictions to {output_file}")


def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print("Usage: python ncaa_daily_predictions_with_contrarian.py YOUR_API_KEY [--no-contrarian]")
        print()
        print("Examples:")
        print("  python ncaa_daily_predictions_with_contrarian.py abc123")
        print("  python ncaa_daily_predictions_with_contrarian.py abc123 --no-contrarian")
        return

    api_key = sys.argv[1]
    enable_contrarian = '--no-contrarian' not in sys.argv

    if not enable_contrarian:
        print("⚠️  Contrarian intelligence DISABLED")

    predictor = NCAADailyPredictionsContrarian(api_key, enable_contrarian)
    predictor.run_daily_analysis()


if __name__ == "__main__":
    main()
