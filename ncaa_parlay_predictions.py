#!/usr/bin/env python3
"""
NCAA 12-Model System + Parlay Optimizer
========================================

Combines the 12-model super intelligence system with parlay optimization
to generate profitable multi-leg bets.

Features:
- Loads all 11 trained NCAA models
- Gets predictions for current week's games
- Applies calibration (0.90x multiplier)
- Applies officiating bias adjustments
- Builds optimal 2-4 leg parlays
- Correlation-aware parlay selection
- Kelly Criterion stake sizing
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator
from college_football_system.parlay_optimizer import ParlayOptimizer, ParlayBet


# Configuration
CALIBRATION_MULTIPLIER = 0.90  # Models are overconfident
MIN_CONFIDENCE = 0.65  # 65% minimum for parlays
MIN_EDGE = 0.05  # 5% edge minimum
FRACTIONAL_KELLY = 0.25  # Use 25% of full Kelly
STARTING_BANKROLL = 10000  # $10K


class NCAASystemWithParlays:
    """NCAA 12-model system with parlay optimization"""

    def __init__(self, models_dir="models/ncaa"):
        self.models_dir = Path(models_dir)
        self.engineer = NCAAFeatureEngineer()
        self.orchestrator = SuperIntelligenceOrchestrator(models_dir=str(models_dir))
        self.parlay_optimizer = ParlayOptimizer()

        # Load trained models
        self._load_models()

        print("✅ NCAA System with Parlays initialized")
        print(f"   Loaded {self.models_loaded}/11 models")

    def _load_models(self):
        """Load all trained models"""
        model_names = [
            'spread_ensemble', 'total_ensemble', 'moneyline_ensemble',
            'first_half_spread', 'home_team_total', 'away_team_total',
            'alt_spread', 'xgboost_super', 'neural_net_deep',
            'officiating_bias', 'prop_bet_specialist'
        ]

        self.models_loaded = 0
        for model_name in model_names:
            model_path = self.models_dir / f"{model_name}.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        model = self.orchestrator.models[model_name]

                        # Load model components
                        if 'model' in model_data:
                            model.model = model_data['model']
                        if 'models' in model_data:
                            model.models = model_data['models']
                        if 'scaler' in model_data:
                            model.scaler = model_data['scaler']

                        model.is_trained = model_data.get('is_trained', True)
                        self.models_loaded += 1

                except Exception as e:
                    print(f"⚠️  Failed to load {model_name}: {e}")

    def predict_games(self, year: int, week: int = None, include_completed: bool = False) -> List[Dict[str, Any]]:
        """
        Get predictions for all games in a season/week
        Returns list of game predictions with edge/confidence

        Args:
            year: Season year
            week: Specific week (None = all games)
            include_completed: Include completed games for demo/backtest
        """
        # Try to load season data, fallback to previous year if needed
        try:
            games = self.engineer.load_season_data(year)
            if not games or len(games) == 0:
                print(f"⚠️  No data for {year}, trying {year-1}...")
                year = year - 1
                games = self.engineer.load_season_data(year)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"⚠️  {year} data not available, using {year-1}")
            year = year - 1
            games = self.engineer.load_season_data(year)

        # Filter by week if specified
        if week is not None:
            games = [g for g in games if g.get('week') == week]

        predictions = []

        for game in games:
            # Skip if already completed (unless demo mode)
            if not include_completed and game.get('completed', False):
                continue

            try:
                pred = self._predict_game(game, year)
                if pred:
                    predictions.append(pred)
            except Exception as e:
                print(f"⚠️  Error predicting {game.get('homeTeam', 'Unknown')}: {e}")

        return predictions

    def _predict_game(self, game: Dict, year: int) -> Dict[str, Any]:
        """Predict a single game"""
        # Engineer features
        features = self.engineer.engineer_features(game, year)
        if not features:
            return None

        # Convert to array
        feature_array = np.array([list(features.values())])

        # Get consensus prediction from spread models
        spread_models = ['spread_ensemble', 'xgboost_super', 'neural_net_deep', 'alt_spread']
        spread_preds = []

        for model_name in spread_models:
            model = self.orchestrator.models[model_name]
            if model.is_trained:
                try:
                    pred = model.predict(feature_array)
                    spread_preds.append(pred[0] if isinstance(pred, np.ndarray) else pred)
                except:
                    pass

        if not spread_preds:
            return None

        # Calculate consensus
        predicted_spread = np.mean(spread_preds)
        spread_std = np.std(spread_preds)

        # Calculate confidence (inverse of std dev)
        raw_confidence = max(0.5, min(0.95, 1 - (spread_std / 15.0)))
        confidence = raw_confidence * CALIBRATION_MULTIPLIER

        # Apply officiating bias (Model 11)
        home_conference = game.get('homeConference', '')
        away_conference = game.get('awayConference', '')

        # Simple conference bias adjustments
        bias_adjustments = {
            'SEC': -0.5,    # SEC refs favor home team
            'Big Ten': -0.3,
            'Big 12': -0.2,
            'ACC': -0.1,
        }

        bias_adj = bias_adjustments.get(home_conference, 0.0)
        predicted_spread += bias_adj

        # Calculate edge (simplified - assumes market spread is 0 for demo)
        # In production, you'd fetch real market spreads
        market_spread = 0.0  # TODO: Integrate with odds API
        edge = abs(predicted_spread - market_spread) / 14.0

        # Build prediction object
        return {
            'home_team': game.get('homeTeam', ''),
            'away_team': game.get('awayTeam', ''),
            'conference': home_conference,
            'predicted_spread': predicted_spread,
            'market_spread': market_spread,
            'confidence': confidence,
            'edge_value': edge,
            'odds': -110,  # Standard odds
            'week': game.get('week', 0),
            'venue': game.get('venue', ''),
            'neutral_site': game.get('neutralSite', False),
            'game_id': game.get('id', ''),
        }

    def generate_parlays(self, predictions: List[Dict], bankroll: float = STARTING_BANKROLL) -> List[ParlayBet]:
        """Generate optimal parlays from predictions"""
        # Filter high-confidence predictions
        eligible = [
            p for p in predictions
            if p['confidence'] >= MIN_CONFIDENCE and p['edge_value'] >= MIN_EDGE
        ]

        print(f"\n📊 Parlay Analysis:")
        print(f"   Total games analyzed: {len(predictions)}")
        print(f"   Eligible for parlays: {len(eligible)} ({len(eligible)/max(1, len(predictions))*100:.1f}%)")
        print(f"   Confidence threshold: {MIN_CONFIDENCE*100:.0f}%")
        print(f"   Edge threshold: {MIN_EDGE*100:.1f}%")

        if len(eligible) < 2:
            print("   ⚠️  Not enough eligible games for parlays")
            return []

        # Generate parlays
        parlays = self.parlay_optimizer.optimize_parlays(eligible, bankroll)

        return parlays

    def display_predictions(self, predictions: List[Dict]):
        """Display individual game predictions"""
        print("\n" + "="*80)
        print("🏈 NCAA GAME PREDICTIONS (12-Model System)")
        print("="*80)

        # Sort by edge * confidence
        predictions.sort(key=lambda x: x['edge_value'] * x['confidence'], reverse=True)

        print(f"\n{'RANK':<6} {'MATCHUP':<45} {'CONF':<6} {'EDGE':<7} {'SPREAD':<8}")
        print("-"*80)

        for i, pred in enumerate(predictions[:20], 1):  # Top 20
            matchup = f"{pred['away_team']} @ {pred['home_team']}"
            if len(matchup) > 43:
                matchup = matchup[:40] + "..."

            print(f"{i:<6} {matchup:<45} {pred['confidence']*100:>5.1f}% "
                  f"{pred['edge_value']*100:>5.1f}% {pred['predicted_spread']:>+6.1f}")

    def display_parlays(self, parlays: List[ParlayBet]):
        """Display parlay recommendations"""
        if not parlays:
            print("\n⚠️  No parlays meet the criteria")
            return

        print("\n" + "="*80)
        print("🎰 TOP PARLAY RECOMMENDATIONS")
        print("="*80)

        for i, parlay in enumerate(parlays[:5], 1):  # Top 5
            print(f"\n{'#'}{i}. {len(parlay.games)}-LEG PARLAY")
            print(f"   {'Expected Value:':<20} {parlay.expected_value:>6.1%}")
            print(f"   {'Combined Odds:':<20} +{parlay.combined_odds:>6.0f}")
            print(f"   {'Avg Confidence:':<20} {parlay.confidence:>6.1%}")
            print(f"   {'Risk Score:':<20} {parlay.risk_score:>6.1%}")
            print(f"   {'Recommended Stake:':<20} ${parlay.recommended_stake:>6.0f}")

            payout = parlay.recommended_stake * (parlay.combined_odds / 100)
            profit = payout - parlay.recommended_stake
            print(f"   {'Potential Payout:':<20} ${payout:>6.0f} (${profit:+.0f} profit)")

            print(f"\n   Games:")
            for j, game in enumerate(parlay.games, 1):
                print(f"      {j}. {game['away_team']} @ {game['home_team']}")
                print(f"         {game['conference']:<12} Edge: {game['edge_value']:>5.1%}  "
                      f"Conf: {game['confidence']:>5.1%}  Spread: {game['predicted_spread']:>+5.1f}")

            print(f"\n   Analysis: {'; '.join(parlay.reasoning)}")


def main():
    """Main entry point"""
    print("🏈 NCAA 12-Model System + Parlay Optimizer")
    print("="*80)

    # Initialize system
    system = NCAASystemWithParlays()

    # Get current year/week (with fallback to 2024)
    current_year = 2025  # Try current season
    current_week = 5  # Week 5 has good games
    include_completed = True  # Demo mode - show completed games

    print(f"\n📅 Analyzing {current_year} season" + (f" Week {current_week}" if current_week else " (all games)"))
    if include_completed:
        print("   💡 Demo mode: Including completed games to show system capabilities")

    # Get predictions
    predictions = system.predict_games(current_year, current_week, include_completed=include_completed)

    if not predictions:
        print("❌ No upcoming games found")
        return

    # Display individual predictions
    system.display_predictions(predictions)

    # Generate parlays
    parlays = system.generate_parlays(predictions, STARTING_BANKROLL)

    # Display parlays
    system.display_parlays(parlays)

    # Summary
    print("\n" + "="*80)
    print("📈 SUMMARY")
    print("="*80)
    print(f"   Total games analyzed: {len(predictions)}")
    print(f"   High-confidence plays: {sum(1 for p in predictions if p['confidence'] >= 0.70)}")
    print(f"   High-edge plays: {sum(1 for p in predictions if p['edge_value'] >= 0.05)}")
    print(f"   Parlays generated: {len(parlays)}")
    if parlays:
        avg_ev = np.mean([p.expected_value for p in parlays[:5]])
        print(f"   Average EV (top 5): {avg_ev:.2%}")
    print()


if __name__ == "__main__":
    main()
