#!/usr/bin/env python3
"""
Train ALL 12 Models Including Stacking, Officiating Bias, and Props
"""

import sys
import logging
from pathlib import Path
from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*70)
    print("🤖 NCAA 12-MODEL TRAINING (Complete System)")
    print("="*70)
    print()
    print("Models 1-9: Core prediction models ✅ (already trained)")
    print("Model 10: Stacking Meta-Learner 🔧 (will train)")
    print("Model 11: Officiating Bias 🔧 (will train)")
    print("Model 12: Prop Bet Specialist 🔧 (will train)")
    print()

    # Initialize
    engineer = NCAAFeatureEngineer("data/football/historical/ncaaf")
    orchestrator = SuperIntelligenceOrchestrator("models/ncaa")

    # Load existing 9 models
    print("📂 Loading existing 9 trained models...")
    import pickle
    for model_name in ['spread_ensemble', 'total_ensemble', 'moneyline_ensemble',
                        'first_half_spread', 'home_team_total', 'away_team_total',
                        'alt_spread', 'xgboost_super', 'neural_net_deep']:
        model_path = Path(f"models/ncaa/{model_name}.pkl")
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                orchestrator.models[model_name].model = model_data['model']
                orchestrator.models[model_name].scaler = model_data['model']
                orchestrator.models[model_name].is_trained = model_data['is_trained']
                print(f"  ✅ {model_data['name']}")

    # Get seasons
    seasons_input = input("\nSeasons to train on? (e.g., 2015-2024): ").strip()
    if '-' in seasons_input:
        start, end = seasons_input.split('-')
        seasons = list(range(int(start), int(end) + 1))
    else:
        seasons = [2023, 2024]

    print(f"\nTraining new models on: {seasons}")
    if input("Proceed? (y/n): ").lower() != 'y':
        return

    print("\n" + "="*70)
    print("🎯 Training Model 10: Stacking Meta-Learner")
    print("="*70)

    # Train Model 10: Stacking Meta-Learner
    # Collect predictions from models 1-9 to train stacking learner
    all_X = []
    all_y = []

    for year in seasons:
        games = engineer.load_season_data(year)
        X, y = engineer.create_dataset(games, year, target='spread')

        if X is not None and len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if all_X:
        combined_X = pd.concat(all_X, ignore_index=True)
        combined_y = np.concatenate(all_y)

        print(f"Dataset: {len(combined_X)} samples")

        # Get base model predictions for meta-learning
        meta_features = []
        base_models = []

        for model_name in ['spread_ensemble', 'xgboost_super', 'neural_net_deep', 'alt_spread']:
            if orchestrator.models[model_name].is_trained:
                try:
                    preds = orchestrator.models[model_name].predict(combined_X)
                    meta_features.append(preds.reshape(-1, 1))
                    base_models.append(orchestrator.models[model_name])
                except Exception as e:
                    print(f"  ⚠️  Could not get predictions from {model_name}: {e}")

        if meta_features:
            # Train stacking learner with simplified approach
            from sklearn.linear_model import Ridge

            meta_X = np.hstack(meta_features)
            stacking_model = Ridge(alpha=1.0)
            stacking_model.fit(meta_X, combined_y)

            orchestrator.models['stacking_meta'].model = stacking_model
            orchestrator.models['stacking_meta'].base_models = base_models
            orchestrator.models['stacking_meta'].is_trained = True

            print(f"  ✅ Stacking Meta-Learner trained on {len(base_models)} base models")
        else:
            print("  ⚠️  Could not train stacking learner")

    print("\n" + "="*70)
    print("🎯 Training Model 11: Officiating Bias")
    print("="*70)

    # Model 11: Officiating Bias
    # This model adjusts predictions based on conference officiating patterns
    # For now, it uses pre-built profiles (no training needed)
    orchestrator.models['officiating_bias'].is_trained = True
    print("  ✅ Officiating Bias Model ready (uses pre-built conference profiles)")
    print("     - SEC: +1.6 pts home bias")
    print("     - Big Ten: +0.8 pts home bias")
    print("     - Pac-12: Neutral (most balanced)")

    print("\n" + "="*70)
    print("🎯 Training Model 12: Prop Bet Specialist")
    print("="*70)

    # Model 12: Prop Bet Specialist
    # Train basic team total model for now
    from sklearn.ensemble import RandomForestRegressor

    all_X = []
    all_y = []

    for year in seasons:
        games = engineer.load_season_data(year)
        X, y = engineer.create_dataset(games, year, target='home_total')

        if X is not None and len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if all_X:
        combined_X = pd.concat(all_X, ignore_index=True)
        combined_y = np.concatenate(all_y)

        print(f"Dataset: {len(combined_X)} samples")

        # Train team total model
        team_total_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        team_total_model.fit(combined_X, combined_y)

        orchestrator.models['prop_bet_specialist'].team_total_model = team_total_model
        orchestrator.models['prop_bet_specialist'].model = team_total_model  # For base class compatibility
        orchestrator.models['prop_bet_specialist'].is_trained = True

        print("  ✅ Prop Bet Specialist trained (team totals)")

    # Save all 12 models
    print("\n" + "="*70)
    print("💾 Saving all 12 models...")
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
    print("✅ 12-MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Models trained and saved: {saved_count}/12")

    print("\n📋 Model Status:")
    for i, (name, model) in enumerate(orchestrator.models.items(), 1):
        status = "✅" if model.is_trained else "❌"
        print(f"  {i:2d}. {status} {model.name}")

    print("\n🎯 Next steps:")
    print("  1. Run backtest: python backtest_high_roi.py")
    print("  2. Make live predictions: python predict_live_games.py")
    print("  3. Test officiating: python test_ncaa_officiating_bias.py")

if __name__ == "__main__":
    main()
