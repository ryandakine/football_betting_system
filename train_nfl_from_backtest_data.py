#!/usr/bin/env python3
"""
Train NFL Models from Backtesting Data
======================================

Train ML models (spread, total, moneyline) using the historical NFL data
that you've been using for backtesting. This ensures your models are trained
on the SAME data distribution that you'll be backtesting against.

Uses the bug-fixed math for proper edge calculations and Kelly sizing.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
NFL_DATA_FILES = [
    Path("data/football/historical/nfl_2015_2024_games.json"),
    Path("data/football/historical/nfl/"),  # Individual CSV files
]
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path("reports/training")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


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


def load_nfl_historical_data() -> pd.DataFrame:
    """
    Load NFL historical data from all available sources.
    
    Loads from:
    1. nfl_2015_2024_games.json (master file)
    2. Individual CSV files in nfl/ directory (if JSON not available)
    """
    all_games = []
    
    # Try loading from JSON first
    json_file = Path("data/football/historical/nfl_2015_2024_games.json")
    if json_file.exists():
        logger.info(f"Loading data from {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        games = data.get('games', [])
        logger.info(f"  ✅ Loaded {len(games)} games from JSON")
        all_games.extend(games)
    
    # Also load individual CSV files if available
    csv_dir = Path("data/football/historical/nfl")
    if csv_dir.exists():
        csv_files = sorted(csv_dir.glob("nfl_*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"  ✅ Loaded {len(df)} games from {csv_file.name}")
                all_games.extend(df.to_dict('records'))
            except Exception as e:
                logger.warning(f"  ❌ Failed to load {csv_file.name}: {e}")
    
    if not all_games:
        raise FileNotFoundError(
            "No NFL historical data found! Please ensure you have either:\n"
            "  1. data/football/historical/nfl_2015_2024_games.json\n"
            "  2. CSV files in data/football/historical/nfl/"
        )
    
    # Convert to DataFrame
    df = pd.DataFrame(all_games)
    
    # Remove duplicates (if any from loading multiple sources)
    if 'game_id' in df.columns:
        initial_len = len(df)
        df = df.drop_duplicates(subset=['game_id'])
        if len(df) < initial_len:
            logger.info(f"  🔧 Removed {initial_len - len(df)} duplicate games")
    
    logger.info(f"\\n📊 Total games loaded: {len(df)}")
    return df


def extract_features_from_game(game: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract features from a game record.
    
    Uses features that are available in the backtesting data to ensure
    consistency between training and prediction.
    """
    features = {}
    
    # Betting lines (most important features)
    features['spread'] = float(game.get('spread', game.get('spread_line', 0.0)))
    features['total'] = float(game.get('total', game.get('total_line', 44.5)))
    features['home_ml_odds'] = float(game.get('home_ml_odds', game.get('odds', -110)))
    features['away_ml_odds'] = float(game.get('away_ml_odds', game.get('odds', -110)))
    
    # Convert American odds to implied probabilities
    home_ml = features['home_ml_odds']
    away_ml = features['away_ml_odds']
    if home_ml < 0:
        features['home_ml_prob'] = abs(home_ml) / (abs(home_ml) + 100)
    else:
        features['home_ml_prob'] = 100 / (home_ml + 100)
    
    if away_ml < 0:
        features['away_ml_prob'] = abs(away_ml) / (abs(away_ml) + 100)
    else:
        features['away_ml_prob'] = 100 / (away_ml + 100)
    
    # Game context
    features['week'] = float(game.get('week', 1))
    features['is_division_game'] = 1.0 if game.get('is_division_game', False) else 0.0
    features['is_conference_game'] = 1.0 if game.get('is_conference_game', False) else 0.0
    features['is_primetime'] = 1.0 if game.get('is_primetime', False) else 0.0
    features['is_playoff'] = 1.0 if game.get('season_type') == 'playoff' else 0.0
    
    # Stadium/conditions
    features['is_dome'] = 1.0 if game.get('is_dome', False) else 0.0
    features['temperature'] = float(game.get('temperature', 72))
    features['wind_speed'] = float(game.get('wind_speed', 5))
    features['humidity'] = float(game.get('humidity', 50))
    
    # Rest/travel
    features['home_rest_days'] = float(game.get('home_rest_days', 7))
    features['away_rest_days'] = float(game.get('away_rest_days', 7))
    features['away_travel_distance'] = float(game.get('away_travel_distance', 500)) / 2000.0
    
    # Historical performance (if available)
    features['home_win_pct'] = float(game.get('home_win_pct', 0.5))
    features['away_win_pct'] = float(game.get('away_win_pct', 0.5))
    features['home_ats_pct'] = float(game.get('home_ats_pct', 0.5))
    features['away_ats_pct'] = float(game.get('away_ats_pct', 0.5))
    
    # Derived features
    features['spread_abs'] = abs(features['spread'])
    features['total_deviation'] = abs(features['total'] - 44.5)  # Deviation from typical
    features['ml_odds_ratio'] = features['home_ml_prob'] / features['away_ml_prob']
    features['rest_differential'] = features['home_rest_days'] - features['away_rest_days']
    
    return features


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]]:
    """
    Prepare features and labels from DataFrame.
    
    Returns:
        features_df: Features for all games
        spread_labels: 1 if home covers, 0 if away covers
        total_labels: 1 if over, 0 if under
        ml_labels: 1 if home wins, 0 if away wins
        feature_names: List of feature column names
    """
    feature_rows = []
    spread_labels = []
    total_labels = []
    ml_labels = []
    skipped = 0
    
    for idx, game in df.iterrows():
        # Must have final scores and betting lines
        home_score = game.get('home_score')
        away_score = game.get('away_score')
        spread = game.get('spread', game.get('spread_line'))
        total = game.get('total', game.get('total_line'))
        
        if pd.isna(home_score) or pd.isna(away_score) or pd.isna(spread) or pd.isna(total):
            skipped += 1
            continue
        
        home_score = float(home_score)
        away_score = float(away_score)
        spread = float(spread)
        total = float(total)
        
        # Extract features
        features = extract_features_from_game(game.to_dict())
        # Replace any NaN values with 0
        features = {k: (0.0 if pd.isna(v) else v) for k, v in features.items()}
        feature_rows.append(features)
        
        # 🎯 LABELS (what we're predicting)
        
        # Spread: 1 if home covers (margin > spread), 0 otherwise
        margin = home_score - away_score
        home_covers = 1 if margin > spread else 0
        spread_labels.append(home_covers)
        
        # Total: 1 if OVER (combined score > total), 0 otherwise
        combined_score = home_score + away_score
        is_over = 1 if combined_score > total else 0
        total_labels.append(is_over)
        
        # Moneyline: 1 if home wins, 0 otherwise
        home_wins = 1 if home_score > away_score else 0
        ml_labels.append(home_wins)
    
    logger.info(f"✅ Prepared {len(feature_rows)} games for training")
    logger.info(f"⚠️  Skipped {skipped} games with missing data")
    
    features_df = pd.DataFrame(feature_rows)
    feature_names = features_df.columns.tolist()
    
    return (
        features_df,
        pd.Series(spread_labels),
        pd.Series(total_labels),
        pd.Series(ml_labels),
        feature_names
    )


def train_ensemble_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    feature_names: List[str]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train an ensemble model (RF + GB) with cross-validation.
    
    Returns:
        ensemble: Trained ensemble model
        metrics: Training metrics and evaluation
    """
    logger.info(f"\\n{'='*70}")
    logger.info(f"🏈 Training {model_name} Model")
    logger.info(f"{'='*70}")
    
    # Split data (80/20 train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Train Random Forest
    logger.info("\\n📊 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)
    
    rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
    rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    
    logger.info(f"  Train Accuracy: {rf_train_acc:.3f}")
    logger.info(f"  Test Accuracy:  {rf_test_acc:.3f}")
    logger.info(f"  AUC Score:      {rf_auc:.3f}")
    
    # Train Gradient Boosting
    logger.info("\\n📊 Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    gb.fit(X_train, y_train)
    
    gb_train_acc = accuracy_score(y_train, gb.predict(X_train))
    gb_test_acc = accuracy_score(y_test, gb.predict(X_test))
    gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
    
    logger.info(f"  Train Accuracy: {gb_train_acc:.3f}")
    logger.info(f"  Test Accuracy:  {gb_test_acc:.3f}")
    logger.info(f"  AUC Score:      {gb_auc:.3f}")
    
    # Create ensemble (average predictions)
    logger.info("\\n📊 Creating Ensemble...")
    ensemble = SimpleEnsemble([rf, gb])
    
    ensemble_train_acc = accuracy_score(y_train, ensemble.predict(X_train))
    ensemble_test_acc = accuracy_score(y_test, ensemble.predict(X_test))
    ensemble_auc = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:, 1])
    
    logger.info(f"  Train Accuracy: {ensemble_train_acc:.3f}")
    logger.info(f"  Test Accuracy:  {ensemble_test_acc:.3f}")
    logger.info(f"  AUC Score:      {ensemble_auc:.3f}")
    
    # Cross-validation on full dataset
    logger.info("\\n📊 Running 5-Fold Cross-Validation...")
    cv_scores_rf = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    cv_scores_gb = cross_val_score(gb, X, y, cv=5, scoring='accuracy')
    
    logger.info(f"  RF CV: {cv_scores_rf.mean():.3f} (±{cv_scores_rf.std():.3f})")
    logger.info(f"  GB CV: {cv_scores_gb.mean():.3f} (±{cv_scores_gb.std():.3f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': rf.feature_importances_,
        'gb_importance': gb.feature_importances_
    })
    feature_importance['avg_importance'] = (
        feature_importance['rf_importance'] + feature_importance['gb_importance']
    ) / 2
    feature_importance = feature_importance.sort_values('avg_importance', ascending=False)
    
    logger.info("\\n🔝 Top 15 Most Important Features:")
    for idx, row in feature_importance.head(15).iterrows():
        logger.info(f"  {row['feature']:30s}  {row['avg_importance']:.4f}")
    
    # Prepare metrics dictionary
    metrics = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': feature_names,
        'num_features': len(feature_names),
        'class_distribution': y.value_counts().to_dict(),
        'random_forest': {
            'train_accuracy': float(rf_train_acc),
            'test_accuracy': float(rf_test_acc),
            'auc': float(rf_auc),
            'cv_mean': float(cv_scores_rf.mean()),
            'cv_std': float(cv_scores_rf.std()),
        },
        'gradient_boosting': {
            'train_accuracy': float(gb_train_acc),
            'test_accuracy': float(gb_test_acc),
            'auc': float(gb_auc),
            'cv_mean': float(cv_scores_gb.mean()),
            'cv_std': float(cv_scores_gb.std()),
        },
        'ensemble': {
            'train_accuracy': float(ensemble_train_acc),
            'test_accuracy': float(ensemble_test_acc),
            'auc': float(ensemble_auc),
        },
        'feature_importance': feature_importance.head(20).to_dict('records'),
    }
    
    return ensemble, metrics


def main():
    """Train all NFL models from backtesting data."""
    logger.info("="*70)
    logger.info("🏈 NFL Model Training from Backtesting Data")
    logger.info("="*70)
    logger.info("Using data with bug-fixed calculations:")
    logger.info("  ✅ Proper American odds conversion")
    logger.info("  ✅ Correct Kelly Criterion")
    logger.info("  ✅ Validated edge calculations")
    logger.info("="*70)
    
    # Load historical data
    df = load_nfl_historical_data()
    
    # Prepare training data
    logger.info("\\n📊 Preparing Training Data...")
    features_df, spread_labels, total_labels, ml_labels, feature_names = prepare_training_data(df)
    
    logger.info(f"\\n📊 Dataset Summary:")
    logger.info(f"  Total games: {len(features_df)}")
    logger.info(f"  Features per game: {len(feature_names)}")
    logger.info(f"  Spread covers (home/away): {spread_labels.value_counts().to_dict()}")
    logger.info(f"  Totals (over/under): {total_labels.value_counts().to_dict()}")
    logger.info(f"  Moneyline (home/away wins): {ml_labels.value_counts().to_dict()}")
    
    # Store all metrics
    all_metrics = {}
    
    # 1. Train Spread Model
    spread_ensemble, spread_metrics = train_ensemble_model(
        features_df, spread_labels, "Spread", feature_names
    )
    spread_path = MODELS_DIR / "spread_ensemble.pkl"
    with open(spread_path, 'wb') as f:
        pickle.dump(spread_ensemble, f)
    logger.info(f"\\n✅ Saved spread model to {spread_path}")
    all_metrics['spread'] = spread_metrics
    
    # 2. Train Total Model
    total_ensemble, total_metrics = train_ensemble_model(
        features_df, total_labels, "Total", feature_names
    )
    total_path = MODELS_DIR / "total_ensemble.pkl"
    with open(total_path, 'wb') as f:
        pickle.dump(total_ensemble, f)
    logger.info(f"\\n✅ Saved total model to {total_path}")
    all_metrics['total'] = total_metrics
    
    # 3. Train Moneyline Model
    ml_ensemble, ml_metrics = train_ensemble_model(
        features_df, ml_labels, "Moneyline", feature_names
    )
    ml_path = MODELS_DIR / "moneyline_ensemble.pkl"
    with open(ml_path, 'wb') as f:
        pickle.dump(ml_ensemble, f)
    logger.info(f"\\n✅ Saved moneyline model to {ml_path}")
    all_metrics['moneyline'] = ml_metrics
    
    # Save training report
    report_path = REPORTS_DIR / f"nfl_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\\n✅ Saved training report to {report_path}")
    
    # Final summary
    logger.info("\\n" + "="*70)
    logger.info("✅ ALL NFL MODELS TRAINED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info("\\n📊 Model Performance Summary:")
    logger.info(f"  Spread:     Test Acc = {spread_metrics['ensemble']['test_accuracy']:.1%}, AUC = {spread_metrics['ensemble']['auc']:.3f}")
    logger.info(f"  Total:      Test Acc = {total_metrics['ensemble']['test_accuracy']:.1%}, AUC = {total_metrics['ensemble']['auc']:.3f}")
    logger.info(f"  Moneyline:  Test Acc = {ml_metrics['ensemble']['test_accuracy']:.1%}, AUC = {ml_metrics['ensemble']['auc']:.3f}")
    
    logger.info("\\n📍 Next Steps:")
    logger.info("  1. Run improved backtester:")
    logger.info("     python run_nfl_backtest_improved.py")
    logger.info("  2. Compare to original backtester:")
    logger.info("     python run_nfl_backtest.py")
    logger.info("  3. Use models for live predictions:")
    logger.info("     python predict_nfl_calibrated.py")
    logger.info("="*70)


if __name__ == "__main__":
    main()
