#!/usr/bin/env python3
"""
Quick retrain of 3 core models with proper saving
XGBoost, Neural Net, and Alt Spread - these are enough for accurate predictions
"""

import pickle
import numpy as np
from pathlib import Path
from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import XGBoostSuperEnsemble, NeuralNetworkDeepModel, AltSpreadModel

print("🔄 Quick Retrain - 3 Core Models")
print("="*80)

# Initialize
engineer = NCAAFeatureEngineer()
models_dir = Path("models/ncaa")
models_dir.mkdir(parents=True, exist_ok=True)

# Models to train
models_to_train = {
    'xgboost_super': (XGBoostSuperEnsemble(), 'spread'),
    'neural_net_deep': (NeuralNetworkDeepModel(), 'spread'),
    'alt_spread': (AltSpreadModel(), 'spread'),
}

# Train on all available years
years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

for model_name, (model, target) in models_to_train.items():
    print(f"\n📊 Training {model_name}...")

    all_X = []
    all_y = []

    # Collect data from all years
    for year in years:
        try:
            games = engineer.load_season_data(year)
            X, y = engineer.create_dataset(games, year, target=target)

            if X is not None and len(X) > 0:
                all_X.append(X)
                all_y.extend(y)
                print(f"  {year}: {len(X)} games")
        except Exception as e:
            print(f"  {year}: Failed - {e}")

    if not all_X:
        print(f"  ❌ No data for {model_name}")
        continue

    # Combine all years
    import pandas as pd
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = np.array(all_y)

    print(f"  Total: {len(X_combined)} games")

    # Train
    model.train(X_combined, y_combined)

    # Save with proper structure
    model_path = models_dir / f"{model_name}.pkl"
    model_data = {
        'name': model.name,
        'model_type': model.model_type,
        'model': model.model,  # Single model object
        'scaler': model.scaler,  # StandardScaler object
        'feature_importance': model.feature_importance,
        'is_trained': True
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"  ✅ Saved to {model_path}")

print("\n" + "="*80)
print("✅ Quick retrain complete!")
print("   3 models trained on 10 years of data")
print("   Ready for backtesting")
