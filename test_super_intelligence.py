#!/usr/bin/env python3
"""
Test the 10-Model Super Intelligence System
Evaluates all models on held-out test data
"""

import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Test all models"""
    print("\n" + "="*70)
    print("🧪 TESTING 10-MODEL SUPER INTELLIGENCE")
    print("="*70)

    # Initialize
    feature_engineer = NCAAFeatureEngineer()
    orchestrator = SuperIntelligenceOrchestrator()

    # Load models
    print("\n📂 Loading trained models...")
    orchestrator.load_all()

    # Check which models are loaded
    trained_models = {name: model for name, model in orchestrator.models.items() if model.is_trained}

    if not trained_models:
        print("❌ No trained models found!")
        print("Run: python train_super_intelligence.py")
        return

    print(f"✅ Loaded {len(trained_models)}/10 models")

    # Load test data (use 2024 as test if trained on 2023)
    test_year = int(input("\nWhich season to test on? (default: 2024): ").strip() or "2024")

    print(f"\n📊 Loading {test_year} test data...")
    games = feature_engineer.load_season_data(test_year)
    print(f"✅ Loaded {len(games)} games")

    # Test each model
    print("\n" + "="*70)
    print("MODEL PERFORMANCE")
    print("="*70)

    model_targets = {
        'spread_ensemble': 'spread',
        'total_ensemble': 'total',
        'moneyline_ensemble': 'moneyline',
        'first_half_spread': '1h_spread',
        'home_team_total': 'home_total',
        'away_team_total': 'away_total',
        'alt_spread': 'spread',
        'xgboost_super': 'spread',
        'neural_net_deep': 'spread'
    }

    for model_name, target in model_targets.items():
        if model_name in trained_models:
            model = trained_models[model_name]

            # Create test dataset
            X_test, y_test = feature_engineer.create_dataset(games, test_year, target=target)

            if X_test is not None and len(X_test) > 0:
                # Make predictions
                try:
                    if model.model_type == 'classification':
                        # Classification metrics
                        y_pred = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
                        accuracy = accuracy_score(y_test, y_pred)
                        print(f"\n{model.name}:")
                        print(f"  Accuracy: {accuracy:.3f}")
                    else:
                        # Regression metrics
                        y_pred = model.predict(X_test)
                        if isinstance(y_pred, dict):
                            y_pred = y_pred['mean']

                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)

                        print(f"\n{model.name}:")
                        print(f"  MAE:  {mae:.2f}")
                        print(f"  RMSE: {rmse:.2f}")
                        print(f"  R²:   {r2:.3f}")

                except Exception as e:
                    logger.warning(f"  ⚠️  Test failed: {e}")

    # Test stacking meta-learner separately
    if 'stacking_meta' in trained_models and 'spread' in model_targets.values():
        X_test, y_test = feature_engineer.create_dataset(games, test_year, target='spread')

        if X_test is not None:
            try:
                y_pred = trained_models['stacking_meta'].predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                print(f"\nStacking Meta-Learner:")
                print(f"  MAE:  {mae:.2f}")
                print(f"  RMSE: {rmse:.2f}")
                print(f"  R²:   {r2:.3f}")
            except Exception as e:
                logger.warning(f"  ⚠️  Stacking test failed: {e}")

    # Test consensus predictions
    print("\n" + "="*70)
    print("CONSENSUS PREDICTION TEST")
    print("="*70)

    # Take a sample game and show all model predictions
    if len(games) > 0:
        sample_game = games[0]
        print(f"\nSample Game: {sample_game.get('away_team')} @ {sample_game.get('home_team')}")

        features = feature_engineer.engineer_features(sample_game, test_year)
        import pandas as pd
        features_df = pd.DataFrame([features])

        all_preds = orchestrator.predict(features_df)
        consensus = orchestrator.get_consensus_prediction(all_preds)

        print("\nModel Predictions:")
        for model_name, pred in all_preds.items():
            if not isinstance(pred, dict):
                print(f"  {model_name:25s}: {pred[0]:.2f}")

        print("\nConsensus:")
        for key, value in consensus.items():
            if hasattr(value, '__iter__'):
                print(f"  {key:25s}: {value[0]:.2f}")

    print("\n" + "="*70)
    print("✅ TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
