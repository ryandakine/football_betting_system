#!/usr/bin/env python3
"""
NCAA Predictions - No API Required Version
===========================================

Generates predictions WITHOUT needing Odds API.
You manually compare our spreads to market (DraftKings, FanDuel, etc.)

USAGE:
    python ncaa_predictions_no_api.py 2025 5

This will generate predictions for 2025 Week 5, then you:
1. Go to DraftKings/FanDuel
2. Compare our spreads to their spreads
3. Bet when we have edge
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator


class NCAANonAPIPredictions:
    """Generate predictions without needing Odds API"""

    def __init__(self):
        self.models_dir = Path("models/ncaa")
        self.predictions_dir = Path("data/manual_predictions")
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        self.engineer = NCAAFeatureEngineer()
        self.orchestrator = SuperIntelligenceOrchestrator(models_dir=str(self.models_dir))

        self._load_models()

        print(f"✅ Prediction System initialized (NO API MODE)")
        print(f"   Models loaded: {self.models_loaded}/3")

    def _load_models(self):
        """Load trained models"""
        model_names = ['xgboost_super', 'neural_net_deep', 'alt_spread']

        self.models_loaded = 0
        for model_name in model_names:
            model_path = self.models_dir / f"{model_name}.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        model = self.orchestrator.models[model_name]

                        if 'model' in model_data:
                            if isinstance(model_data['model'], dict):
                                model.models = model_data['model']
                            else:
                                model.model = model_data['model']

                        if 'scaler' in model_data:
                            model.scaler = model_data['scaler']

                        model.is_trained = True
                        self.models_loaded += 1

                except Exception as e:
                    print(f"⚠️  Error loading {model_name}: {e}")

    def generate_predictions(self, year: int, week: int = None) -> List[Dict]:
        """Generate predictions for games"""
        print(f"\n🎯 Generating predictions for {year} Week {week or 'ALL'}")

        try:
            games = self.engineer.load_season_data(year)
        except:
            print(f"   ❌ No data for {year} season yet")
            return []

        if week:
            games = [g for g in games if g.get('week') == week and not g.get('completed')]
        else:
            games = [g for g in games if not g.get('completed')]

        if not games:
            print(f"   No upcoming games")
            return []

        predictions = []

        for game in games:
            try:
                features = self.engineer.engineer_features(game, year)
                if not features:
                    continue

                feature_array = np.array([list(features.values())])

                spread_preds = []
                for model_name in ['xgboost_super', 'neural_net_deep', 'alt_spread']:
                    model = self.orchestrator.models[model_name]
                    if model.is_trained:
                        try:
                            pred = model.predict(feature_array)
                            spread_preds.append(pred[0] if isinstance(pred, np.ndarray) else pred)
                        except:
                            pass

                if not spread_preds:
                    continue

                predicted_spread = np.mean(spread_preds)
                spread_std = np.std(spread_preds)

                raw_confidence = max(0.5, min(0.95, 1 - (spread_std / 15.0)))
                confidence = raw_confidence * 0.90

                predictions.append({
                    'game_id': game.get('id'),
                    'home_team': game.get('homeTeam', ''),
                    'away_team': game.get('awayTeam', ''),
                    'our_spread': round(predicted_spread, 1),
                    'confidence': round(confidence * 100, 0),
                    'week': game.get('week', 0),
                })

            except Exception as e:
                continue

        print(f"   ✅ Generated {len(predictions)} predictions")
        return predictions

    def save_predictions(self, predictions: List[Dict], year: int, week: int):
        """Save predictions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = self.predictions_dir / f"predictions_{year}_week{week}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump({
                'year': year,
                'week': week,
                'timestamp': timestamp,
                'predictions': predictions,
            }, f, indent=2)

        print(f"\n💾 Saved to {filename}")

    def display_predictions(self, predictions: List[Dict]):
        """Display predictions"""
        print(f"\n{'='*80}")
        print(f"📊 OUR PREDICTIONS")
        print(f"{'='*80}\n")

        for pred in predictions:
            spread = pred['our_spread']
            conf = pred['confidence']

            print(f"{pred['away_team']} @ {pred['home_team']}")
            print(f"   OUR SPREAD: {spread:+.1f} | CONFIDENCE: {conf:.0f}%")
            print()

        print(f"{'='*80}")
        print(f"📋 HOW TO USE THESE PREDICTIONS:")
        print(f"{'='*80}\n")
        print("1. Go to DraftKings, FanDuel, or BetMGM")
        print("2. Find the same games")
        print("3. Compare THEIR spread to OUR spread\n")
        print("EXAMPLE:")
        print("   Our spread: Ohio State -14.5")
        print("   Market spread: Ohio State -17.5")
        print("   → BET OHIO STATE! Market thinks they win by more, we disagree\n")
        print("EDGE CALCULATION:")
        print("   Edge = |Our spread - Market spread| / 7")
        print("   If Edge > 5%, consider betting")
        print("   If Edge > 10%, strong bet\n")
        print("BANKROLL MANAGEMENT:")
        print("   Bet 1-3% of bankroll per game")
        print("   Use Kelly Criterion for sizing")
        print(f"{'='*80}\n")


def main():
    """Main entry point"""
    print("="*80)
    print("🏈 NCAA PREDICTIONS - NO API REQUIRED")
    print("="*80)
    print()

    if len(sys.argv) < 3:
        print("Usage: python ncaa_predictions_no_api.py YEAR WEEK")
        print()
        print("Example: python ncaa_predictions_no_api.py 2025 5")
        print()
        return

    year = int(sys.argv[1])
    week = int(sys.argv[2])

    system = NCAANonAPIPredictions()

    predictions = system.generate_predictions(year, week)

    if not predictions:
        print("\n⚠️  No predictions available")
        print(f"   Either no games scheduled or season hasn't started")
        return

    system.display_predictions(predictions)
    system.save_predictions(predictions, year, week)


if __name__ == "__main__":
    main()
