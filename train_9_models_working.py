#!/usr/bin/env python3
"""
Train 9-Model NCAA System (Skip stacking learner for now)
This trains the proven, working models
"""

import sys
import logging
from pathlib import Path
from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*70)
    print("🤖 NCAA 9-MODEL TRAINING (Proven Models)")
    print("="*70)

    # Initialize
    engineer = NCAAFeatureEngineer("data/football/historical/ncaaf")
    orchestrator = SuperIntelligenceOrchestrator("models/ncaa")

    # Get seasons
    seasons_input = input("\nSeasons to train on? (e.g., 2015-2024): ").strip()
    if '-' in seasons_input:
        start, end = seasons_input.split('-')
        seasons = list(range(int(start), int(end) + 1))
    else:
        seasons = [2023, 2024]

    print(f"\nTraining on: {seasons}")
    if input("Proceed? (y/n): ").lower() != 'y':
        return

    # Train each year
    trained_count = 0
    for year in seasons:
        print(f"\n{'='*70}")
        print(f"📅 Training on {year} season data...")
        print('='*70)

        games = engineer.load_season_data(year)
        print(f"Loaded {len(games)} games")

        # Train models 1-9 (skip stacking meta-learner #10)
        models_to_train = [
            ('spread_ensemble', 'spread'),
            ('total_ensemble', 'total'),
            ('moneyline_ensemble', 'moneyline'),
            ('first_half_spread', '1h_spread'),
            ('home_team_total', 'home_total'),
            ('away_team_total', 'away_total'),
            ('alt_spread', 'spread'),
            ('xgboost_super', 'spread'),
            ('neural_net_deep', 'spread'),
        ]

        for model_name, target in models_to_train:
            X, y = engineer.create_dataset(games, year, target=target)
            if X is not None and len(X) > 0:
                print(f"\nTraining {orchestrator.models[model_name].name}...")
                print(f"  Dataset: {len(X)} samples, {len(X.columns)} features")
                orchestrator.models[model_name].train(X, y)
                trained_count += 1

    # Save all trained models
    print("\n" + "="*70)
    print("💾 Saving trained models...")
    print("="*70)

    saved_count = 0
    for model_name, model in orchestrator.models.items():
        if model.is_trained:
            try:
                model_path = Path("models/ncaa") / f"{model_name}.pkl"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(model_path)
                print(f"  ✅ Saved: {model.name}")
                saved_count += 1
            except Exception as e:
                print(f"  ⚠️  Could not save {model.name}: {e}")

    # Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Models trained: {saved_count}/9")
    print(f"📁 Saved to: models/ncaa/")

    print("\n📋 Model Status:")
    for i, (name, model) in enumerate(orchestrator.models.items(), 1):
        if name == 'stacking_meta':
            print(f"  {i:2d}. ⏭️  {model.name} (skipped - will add later)")
        else:
            status = "✅" if model.is_trained else "❌"
            print(f"  {i:2d}. {status} {model.name}")

    print("\n🎯 Next steps:")
    print("  1. Test predictions: python predict_ncaa_calibrated.py")
    print("  2. Run backtest: python backtest_ncaa_improved.py")
    print("  3. Analyze calibration: python analyze_ncaa_12_models_calibration.py")

if __name__ == "__main__":
    main()
