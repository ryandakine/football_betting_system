#!/usr/bin/env python3
"""
NFL Model Training Script
=========================
Train spread, total, and moneyline ensemble models using historical NFL data.
Mirrors the NCAA training approach but adapted for NFL.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_FILE = Path("data/football/historical/nfl_2015_2024_games.json")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


class SimpleEnsemble:
    """Simple ensemble that averages predictions from multiple models."""
    def __init__(self, models):
        self.models = models
    
    def predict_proba(self, X):
        """Average probabilities from all models"""
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(probas, axis=0)
    
    def predict(self, X):
        """Predict using averaged probabilities"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


def load_nfl_games() -> List[Dict[str, Any]]:
    """Load NFL historical game data."""
    logger.info(f"Loading NFL games from {DATA_FILE}")
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    
    games = data.get('games', [])
    logger.info(f"✅ Loaded {len(games)} NFL games from {data.get('seasons', [])} seasons")
    return games


def extract_features(game: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract features from game data.
    Uses simple, robust features available in historical data.
    """
    features = {}
    
    # Basic game info
    features['spread'] = float(game.get('spread', 0.0))
    features['total'] = float(game.get('total', 44.5))
    features['home_ml_odds'] = float(game.get('home_ml_odds', -110))
    features['away_ml_odds'] = float(game.get('away_ml_odds', -110))
    
    # Game context
    features['is_division_game'] = 1.0 if game.get('is_division_game') else 0.0
    features['is_primetime'] = 1.0 if game.get('is_primetime') else 0.0
    features['is_dome'] = 1.0 if game.get('is_dome') else 0.0
    features['week'] = float(game.get('week', 1))
    
    # Stadium/attendance
    features['attendance'] = float(game.get('attendance', 50000)) / 75000.0  # Normalize
    
    # Travel (if available)
    features['away_travel_distance'] = float(game.get('away_travel_distance', 500)) / 2000.0  # Normalize
    
    # Rest days
    features['home_rest_days'] = float(game.get('home_rest_days', 7))
    features['away_rest_days'] = float(game.get('away_rest_days', 7))
    
    # Playoff indicator
    features['is_playoff'] = 1.0 if game.get('season_type') == 'playoff' else 0.0
    
    return features


def prepare_training_data(games: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Prepare features and labels for spread, total, and moneyline predictions.
    
    Returns:
        features_df: DataFrame with all features
        spread_labels: 1 if home covers, 0 if away covers
        total_labels: 1 if over, 0 if under
        ml_labels: 1 if home wins, 0 if away wins
    """
    feature_rows = []
    spread_labels = []
    total_labels = []
    ml_labels = []
    
    skipped = 0
    for game in games:
        # Must have final scores
        home_score = game.get('home_score')
        away_score = game.get('away_score')
        spread = game.get('spread')
        total = game.get('total')
        
        if home_score is None or away_score is None or spread is None or total is None:
            skipped += 1
            continue
        
        home_score = float(home_score)
        away_score = float(away_score)
        spread = float(spread)
        total = float(total)
        
        # Extract features
        features = extract_features(game)
        feature_rows.append(features)
        
        # Spread label: home covers = 1, away covers = 0
        margin = home_score - away_score
        home_covers = 1 if margin > spread else 0
        spread_labels.append(home_covers)
        
        # Total label: over = 1, under = 0
        final_total = home_score + away_score
        is_over = 1 if final_total > total else 0
        total_labels.append(is_over)
        
        # Moneyline label: home win = 1, away win = 0
        home_wins = 1 if home_score > away_score else 0
        ml_labels.append(home_wins)
    
    logger.info(f"✅ Prepared {len(feature_rows)} games for training (skipped {skipped} incomplete games)")
    
    features_df = pd.DataFrame(feature_rows)
    return features_df, pd.Series(spread_labels), pd.Series(total_labels), pd.Series(ml_labels)


def train_ensemble(X: pd.DataFrame, y: pd.Series, model_name: str) -> Any:
    """
    Train an ensemble model combining Random Forest and Gradient Boosting.
    Returns a simple voting ensemble.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name} ensemble")
    logger.info(f"{'='*60}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} games")
    logger.info(f"Test set: {len(X_test)} games")
    
    # Train Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    logger.info(f"  RF Test Accuracy: {rf_acc:.1%}")
    
    # Train Gradient Boosting
    logger.info("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    logger.info(f"  GB Test Accuracy: {gb_acc:.1%}")
    
    # Create simple ensemble
    ensemble = SimpleEnsemble([rf, gb])
    ensemble_acc = accuracy_score(y_test, ensemble.predict(X_test))
    logger.info(f"  Ensemble Test Accuracy: {ensemble_acc:.1%}")
    
    # Feature importance (from RF)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n📊 Top 10 Feature Importances:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    return ensemble


def main():
    """Train all three NFL ensemble models."""
    logger.info("🏈 NFL Model Training Pipeline")
    logger.info("="*60)
    
    # Load games
    games = load_nfl_games()
    
    # Prepare training data
    features_df, spread_labels, total_labels, ml_labels = prepare_training_data(games)
    
    logger.info(f"\n📊 Dataset Summary:")
    logger.info(f"  Total games: {len(features_df)}")
    logger.info(f"  Features: {len(features_df.columns)}")
    logger.info(f"  Spread distribution: {spread_labels.value_counts().to_dict()}")
    logger.info(f"  Total distribution: {total_labels.value_counts().to_dict()}")
    logger.info(f"  ML distribution: {ml_labels.value_counts().to_dict()}")
    
    # Train spread model
    spread_ensemble = train_ensemble(features_df, spread_labels, "Spread")
    spread_path = MODELS_DIR / "spread_ensemble.pkl"
    with open(spread_path, 'wb') as f:
        pickle.dump(spread_ensemble, f)
    logger.info(f"✅ Saved spread model to {spread_path}")
    
    # Train total model
    total_ensemble = train_ensemble(features_df, total_labels, "Total")
    total_path = MODELS_DIR / "total_ensemble.pkl"
    with open(total_path, 'wb') as f:
        pickle.dump(total_ensemble, f)
    logger.info(f"✅ Saved total model to {total_path}")
    
    # Train moneyline model
    ml_ensemble = train_ensemble(features_df, ml_labels, "Moneyline")
    ml_path = MODELS_DIR / "moneyline_ensemble.pkl"
    with open(ml_path, 'wb') as f:
        pickle.dump(ml_ensemble, f)
    logger.info(f"✅ Saved moneyline model to {ml_path}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ All NFL models trained and saved!")
    logger.info("="*60)
    logger.info("\n📍 Next steps:")
    logger.info("  1. Re-run backtest: python unified_end_to_end_backtest_enhanced.py")
    logger.info("  2. Analyze confidence: python analyze_nfl_confidence_calibration.py")
    logger.info("  3. Create calibrated predictor: predict_nfl_calibrated.py")


if __name__ == "__main__":
    main()
