#!/usr/bin/env python3
"""
Retrain Simple Ensemble Models with Correct Module Path
=======================================================
This script ensures models are pickled with the correct module path
(simple_ensemble.SimpleEnsemble, not __main__.SimpleEnsemble)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import logging
from datetime import datetime

# IMPORTANT: Import from the module, NOT define in __main__
from simple_ensemble import SimpleEnsemble

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data() -> pd.DataFrame:
    """Load historical game data"""
    logger.info("Loading training data...")

    # Try parquet first
    parquet_paths = [
        'data/referee_conspiracy/crew_game_log.parquet',
        'data/backtesting/historical_games.parquet',
        'backtesting_data/games.parquet',
    ]

    for path in parquet_paths:
        if Path(path).exists():
            df = pd.read_parquet(path)
            logger.info(f"✅ Loaded {len(df)} games from {path}")
            return df

    # Try CSV as fallback
    csv_files = list(Path('backtesting_data').glob('*.csv'))
    if csv_files:
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"✅ Loaded {len(df)} games from {len(csv_files)} CSV files")
        return df

    raise FileNotFoundError("No training data found!")


def prepare_features(df: pd.DataFrame):
    """Extract features and targets"""
    logger.info("Preparing features...")

    # Remove non-numeric columns
    exclude_cols = {
        'game_id', 'away_team', 'home_team', 'gameday', 'stadium',
        'weekday', 'gametime', 'game_type', 'season',
        'home_score', 'away_score',
        'BJ', 'DJ', 'FJ', 'LJ', 'SJ', 'U', 'referee',
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Remove columns with too many NaNs
    feature_cols = [c for c in feature_cols
                   if df[c].isna().sum() / len(df) < 0.2]

    logger.info(f"Using {len(feature_cols)} features")

    # Prepare feature matrix
    X = df[feature_cols].fillna(df[feature_cols].mean())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Create targets
    targets = {}

    # Spread: 1 if home covers
    if 'spread_line' in df.columns:
        margin = df['home_score'] - df['away_score']
        targets['spread'] = (margin > df['spread_line']).astype(int)

    # Total: 1 if OVER
    if 'total_line' in df.columns:
        total_points = df['home_score'] + df['away_score']
        targets['total'] = (total_points > df['total_line']).astype(int)

    # Moneyline: 1 if home wins
    targets['moneyline'] = (df['home_score'] > df['away_score']).astype(int)

    return X, targets, feature_cols


def train_model(X, y, model_name: str):
    """Train a single ensemble model"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name} model")
    logger.info(f"{'='*60}")

    # Create ensemble (imported from simple_ensemble module)
    model = SimpleEnsemble(name=model_name)

    # Train
    model.fit(X, y)

    # Evaluate
    accuracy = model.score(X, y)
    logger.info(f"✅ {model_name} accuracy: {accuracy:.3f}")

    return model


def save_model(model, path: str):
    """Save model with proper pickling"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"💾 Saved to: {path}")

    # Verify it can be loaded
    with open(path, 'rb') as f:
        loaded = pickle.load(f)

    logger.info(f"✅ Verified: Model can be loaded successfully")
    logger.info(f"   Module: {loaded.__class__.__module__}")
    logger.info(f"   Class: {loaded.__class__.__name__}")


def main():
    print("="*70)
    print("🏈 RETRAINING SIMPLE ENSEMBLE MODELS")
    print("="*70)
    print()
    print("This ensures models are pickled with correct module paths")
    print("(simple_ensemble.SimpleEnsemble, not __main__.SimpleEnsemble)")
    print()

    # Load data
    df = load_training_data()

    # Prepare features
    X, targets, feature_cols = prepare_features(df)

    print(f"\nTraining on {len(X)} games")
    print(f"Features: {len(feature_cols)}")
    print()

    # Train each model type
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'num_games': len(X),
        'num_features': len(feature_cols),
        'models': {}
    }

    # Train spread model
    if 'spread' in targets:
        spread_model = train_model(X, targets['spread'], 'Spread')
        save_model(spread_model, 'models/spread_ensemble.pkl')
        results['models']['spread'] = {
            'accuracy': spread_model.score(X, targets['spread']),
            'samples': len(targets['spread'])
        }

    # Train total model
    if 'total' in targets:
        total_model = train_model(X, targets['total'], 'Total')
        save_model(total_model, 'models/total_ensemble.pkl')
        results['models']['total'] = {
            'accuracy': total_model.score(X, targets['total']),
            'samples': len(targets['total'])
        }

    # Train moneyline model
    if 'moneyline' in targets:
        ml_model = train_model(X, targets['moneyline'], 'Moneyline')
        save_model(ml_model, 'models/moneyline_ensemble.pkl')
        results['models']['moneyline'] = {
            'accuracy': ml_model.score(X, targets['moneyline']),
            'samples': len(targets['moneyline'])
        }

    # Save training metadata
    with open('models/training_metadata.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\nModels saved:")
    print("  • models/spread_ensemble.pkl")
    print("  • models/total_ensemble.pkl")
    print("  • models/moneyline_ensemble.pkl")
    print("\nThese models can now be properly loaded by other scripts!")
    print("="*70)


if __name__ == "__main__":
    main()
