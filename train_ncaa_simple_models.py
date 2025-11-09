#!/usr/bin/env python3
"""
NCAA Model Training Script - Simplified Version
===============================================
Train spread, total, and moneyline ensemble models using historical NCAA data.
Based on the proven NFL training approach but adapted for college football.

This is a simpler alternative to the 10-model super intelligence system,
easier to train and faster to run while still being highly effective.
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
DATA_DIR = Path("data/football/historical/ncaaf")
MODELS_DIR = Path("models/ncaa_simple")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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


def load_ncaa_games(seasons: List[int]) -> List[Dict[str, Any]]:
    """Load NCAA historical game data from multiple seasons."""
    all_games = []

    for year in seasons:
        games_file = DATA_DIR / f"ncaaf_{year}_games.json"

        if not games_file.exists():
            logger.warning(f"⚠️  Missing data for {year}: {games_file}")
            continue

        logger.info(f"Loading {year} season from {games_file.name}")

        with open(games_file, 'r') as f:
            games = json.load(f)

        # Add year to each game
        for game in games:
            game['season'] = year

        all_games.extend(games)
        logger.info(f"  ✅ Loaded {len(games)} games from {year}")

    logger.info(f"\n✅ Total games loaded: {len(all_games)} from {len(seasons)} seasons")
    return all_games


def load_sp_ratings(year: int) -> Dict[str, Dict[str, float]]:
    """Load SP+ ratings for a specific year."""
    sp_file = DATA_DIR / f"ncaaf_{year}_sp_ratings.json"

    if not sp_file.exists():
        return {}

    with open(sp_file, 'r') as f:
        sp_data = json.load(f)

    # Convert to dict keyed by team name
    sp_ratings = {}
    for team in sp_data:
        team_name = team.get('team')
        if team_name:
            sp_ratings[team_name] = {
                'rating': team.get('rating', 0),
                'offense': team.get('offense', 0),
                'defense': team.get('defense', 0),
            }

    return sp_ratings


def extract_features(game: Dict[str, Any], year: int, sp_ratings: Dict) -> Dict[str, float]:
    """
    Extract features from game data.
    Uses simple, robust features available in historical data.
    """
    features = {}

    home_team = game.get('homeTeam', '')
    away_team = game.get('awayTeam', '')

    # Get SP+ ratings
    home_sp = sp_ratings.get(home_team, {})
    away_sp = sp_ratings.get(away_team, {})

    # SP+ features (if available)
    features['home_sp_rating'] = float(home_sp.get('rating', 0))
    features['away_sp_rating'] = float(away_sp.get('rating', 0))
    features['sp_rating_diff'] = features['home_sp_rating'] - features['away_sp_rating']
    features['home_sp_offense'] = float(home_sp.get('offense', 0))
    features['away_sp_offense'] = float(away_sp.get('offense', 0))
    features['home_sp_defense'] = float(home_sp.get('defense', 0))
    features['away_sp_defense'] = float(away_sp.get('defense', 0))

    # ELO features (commonly available)
    features['home_elo'] = float(game.get('homePregameElo', 1500))
    features['away_elo'] = float(game.get('awayPregameElo', 1500))
    features['elo_diff'] = features['home_elo'] - features['away_elo']

    # Game context
    features['week'] = float(game.get('week', 1))
    features['season_type'] = 1.0 if game.get('seasonType') == 'regular' else 2.0
    features['neutral_site'] = 1.0 if game.get('neutralSite') else 0.0
    features['conference_game'] = 1.0 if game.get('conferenceGame') else 0.0

    # Venue features
    features['attendance'] = float(game.get('attendance', 40000)) / 100000.0  # Normalize
    features['indoor'] = 1.0 if game.get('indoors') else 0.0

    # Win streaks (if available)
    features['home_wins'] = float(game.get('homeWins', 0))
    features['away_wins'] = float(game.get('awayWins', 0))
    features['home_losses'] = float(game.get('homeLosses', 0))
    features['away_losses'] = float(game.get('awayLosses', 0))

    # Excitement/importance
    features['excitement_index'] = float(game.get('excitement', 0))

    return features


def prepare_training_data(games: List[Dict[str, Any]], seasons: List[int]) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Prepare features and labels for spread, total, and moneyline predictions.

    Returns:
        features_df: DataFrame with all features
        spread_labels: 1 if home covers, 0 if away covers
        total_labels: 1 if over, 0 if under (Not implemented - needs betting lines)
        ml_labels: 1 if home wins, 0 if away wins
    """
    feature_rows = []
    spread_labels = []
    total_labels = []
    ml_labels = []

    # Load SP+ ratings for all seasons
    all_sp_ratings = {}
    for year in seasons:
        all_sp_ratings[year] = load_sp_ratings(year)

    skipped = 0
    for game in games:
        # Must have final scores
        home_score = game.get('homePoints')
        away_score = game.get('awayPoints')

        if home_score is None or away_score is None:
            skipped += 1
            continue

        if not game.get('completed'):
            skipped += 1
            continue

        home_score = float(home_score)
        away_score = float(away_score)
        year = game.get('season', 2023)

        # Get SP+ ratings for this season
        sp_ratings = all_sp_ratings.get(year, {})

        # Extract features
        features = extract_features(game, year, sp_ratings)
        feature_rows.append(features)

        # Moneyline label: home win = 1, away win = 0
        home_wins = 1 if home_score > away_score else 0
        ml_labels.append(home_wins)

        # Spread label (using ELO-based spread if available)
        # Calculate implied spread from ELO difference
        elo_diff = features['elo_diff']
        implied_spread = elo_diff / 25.0  # Rough conversion: 25 ELO points ~ 1 point spread

        # Or use actual spread if available in game data
        if 'spread' in game and game['spread'] is not None:
            spread = float(game['spread'])
        else:
            spread = implied_spread

        margin = home_score - away_score
        home_covers = 1 if margin > spread else 0
        spread_labels.append(home_covers)

        # Total label (for now, use median total of ~55 for college)
        # In production, you'd use actual totals from odds
        total = game.get('total', 55.0)
        final_total = home_score + away_score
        is_over = 1 if final_total > total else 0
        total_labels.append(is_over)

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
    """Train all three NCAA ensemble models."""
    logger.info("🏈 NCAA Model Training Pipeline (Simplified)")
    logger.info("="*60)

    # Determine which seasons to train on
    seasons_input = input("\nWhich seasons to train on? (e.g., 2023,2024 or 2015-2024): ").strip()

    if not seasons_input:
        seasons = [2023, 2024]  # Default
        logger.info("Using default: 2023-2024")
    elif '-' in seasons_input:
        # Range specified
        start, end = seasons_input.split('-')
        seasons = list(range(int(start), int(end) + 1))
    elif ',' in seasons_input:
        # List specified
        seasons = [int(s.strip()) for s in seasons_input.split(',')]
    else:
        # Single season
        seasons = [int(seasons_input)]

    logger.info(f"Training on seasons: {seasons}")

    # Load games
    games = load_ncaa_games(seasons)

    if not games:
        logger.error("❌ No games loaded! Check your data directory.")
        return

    # Prepare training data
    features_df, spread_labels, total_labels, ml_labels = prepare_training_data(games, seasons)

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

    # Save feature columns for prediction
    feature_cols_path = MODELS_DIR / "feature_columns.json"
    with open(feature_cols_path, 'w') as f:
        json.dump(list(features_df.columns), f)
    logger.info(f"✅ Saved feature columns to {feature_cols_path}")

    logger.info("\n" + "="*60)
    logger.info("✅ All NCAA models trained and saved!")
    logger.info("="*60)
    logger.info("\n📍 Next steps:")
    logger.info("  1. Analyze confidence: python analyze_ncaa_confidence_calibration.py")
    logger.info("  2. Make predictions: python predict_ncaa_calibrated_v2.py")
    logger.info("  3. Run backtest: python backtest_ncaa_improved.py")


if __name__ == "__main__":
    main()
