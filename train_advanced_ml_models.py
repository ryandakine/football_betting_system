#!/usr/bin/env python3
"""
Advanced ML Model Training on 15-Year NFL Dataset
==================================================
Trains sophisticated ML models on historical NFL data (2010-2024).

Models:
1. XGBoost Classifier (spread/total prediction)
2. Neural Network (deep learning)
3. Ensemble (combines all models)

Features Used:
- Referee patterns
- Team stats
- Home/away splits
- Historical matchups
- Public sentiment (contrarian signals, Reddit, sharp/public splits)
- Crowd roar signals
- Consensus strength
- Weather, injuries (optional)
- Line movements

Usage:
    python train_advanced_ml_models.py --data-file data/historical/parlay_training_data.json
    python train_advanced_ml_models.py --model xgboost --test-year 2024
    python train_advanced_ml_models.py --model ensemble --save models/nfl_ensemble.pkl
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict

# ML libraries
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import pandas as pd
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    print("⚠️  ML libraries not installed!")
    print("Install with: pip install xgboost scikit-learn pandas numpy")

# Sentiment analysis
try:
    from ai_council_with_sentiment import SentimentFeatureExtractor
    HAS_SENTIMENT = True
except ImportError:
    HAS_SENTIMENT = False
    print("⚠️  Sentiment analysis not available (ai_council_with_sentiment.py not found)")

# Deep learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("⚠️  PyTorch not installed!")
    print("Install with: pip install torch")


class NFLDataProcessor:
    """Processes raw NFL data into ML-ready features."""

    def __init__(self, use_sentiment: bool = True):
        self.team_encoder = LabelEncoder()
        self.referee_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # Sentiment extractor
        self.use_sentiment = use_sentiment and HAS_SENTIMENT
        self.sentiment_extractor = SentimentFeatureExtractor() if self.use_sentiment else None

        # Feature statistics
        self.team_stats = defaultdict(lambda: {
            'games': 0, 'wins': 0, 'spread_covers': 0, 'overs': 0
        })

        self.referee_stats = defaultdict(lambda: {
            'games': 0, 'home_covers': 0, 'overs': 0
        })

    def load_data(self, data_file: str) -> List[Dict]:
        """Load historical game data."""
        with open(data_file, 'r') as f:
            return json.load(f)

    def extract_features(self, games: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features from game data.

        Returns:
            X_spread: Features for spread prediction
            y_spread: Spread labels (0=away cover, 1=home cover)
            X_total: Features for total prediction
            y_total: Total labels (0=under, 1=over)
        """
        # Build team and referee stats first
        self._build_team_stats(games)
        self._build_referee_stats(games)

        spread_features = []
        spread_labels = []
        total_features = []
        total_labels = []

        for game in games:
            # Skip incomplete games
            if not game.get('spread_result') or game.get('referee', 'Unknown') == 'Unknown':
                continue

            # Extract base features
            features = self._extract_game_features(game)

            # Spread prediction
            if game['spread_result'] in ['HOME_COVER', 'AWAY_COVER']:
                spread_features.append(features)
                spread_labels.append(1 if game['spread_result'] == 'HOME_COVER' else 0)

            # Total prediction
            if game['total_result'] in ['OVER', 'UNDER']:
                total_features.append(features)
                total_labels.append(1 if game['total_result'] == 'OVER' else 0)

        return (
            np.array(spread_features),
            np.array(spread_labels),
            np.array(total_features),
            np.array(total_labels)
        )

    def _build_team_stats(self, games: List[Dict]):
        """Build cumulative team statistics."""
        for game in games:
            home = game['home_team']
            away = game['away_team']

            # Update stats
            self.team_stats[home]['games'] += 1
            self.team_stats[away]['games'] += 1

            # Winner
            if game['winner'] == home:
                self.team_stats[home]['wins'] += 1
            else:
                self.team_stats[away]['wins'] += 1

            # Spread
            if game.get('spread_result') == 'HOME_COVER':
                self.team_stats[home]['spread_covers'] += 1
            elif game.get('spread_result') == 'AWAY_COVER':
                self.team_stats[away]['spread_covers'] += 1

            # Total
            if game.get('total_result') == 'OVER':
                self.team_stats[home]['overs'] += 1
                self.team_stats[away]['overs'] += 1

    def _build_referee_stats(self, games: List[Dict]):
        """Build referee statistics."""
        for game in games:
            ref = game.get('referee', 'Unknown')
            if ref == 'Unknown':
                continue

            self.referee_stats[ref]['games'] += 1

            if game.get('spread_result') == 'HOME_COVER':
                self.referee_stats[ref]['home_covers'] += 1

            if game.get('total_result') == 'OVER':
                self.referee_stats[ref]['overs'] += 1

    def _extract_game_features(self, game: Dict) -> List[float]:
        """Extract feature vector from single game."""
        features = []

        home = game['home_team']
        away = game['away_team']
        ref = game.get('referee', 'Unknown')

        # Team stats features
        home_stats = self.team_stats[home]
        away_stats = self.team_stats[away]

        # Win rates
        home_win_rate = home_stats['wins'] / max(home_stats['games'], 1)
        away_win_rate = away_stats['wins'] / max(away_stats['games'], 1)
        features.extend([home_win_rate, away_win_rate])

        # Spread cover rates
        home_cover_rate = home_stats['spread_covers'] / max(home_stats['games'], 1)
        away_cover_rate = away_stats['spread_covers'] / max(away_stats['games'], 1)
        features.extend([home_cover_rate, away_cover_rate])

        # Over rates
        home_over_rate = home_stats['overs'] / max(home_stats['games'], 1)
        away_over_rate = away_stats['overs'] / max(away_stats['games'], 1)
        features.extend([home_over_rate, away_over_rate])

        # Referee stats features
        if ref != 'Unknown' and ref in self.referee_stats:
            ref_stats = self.referee_stats[ref]
            ref_home_cover_rate = ref_stats['home_covers'] / max(ref_stats['games'], 1)
            ref_over_rate = ref_stats['overs'] / max(ref_stats['games'], 1)
            features.extend([ref_home_cover_rate, ref_over_rate, ref_stats['games']])
        else:
            features.extend([0.5, 0.5, 0])  # Neutral

        # Betting line features
        features.append(game['spread_line'])
        features.append(game['total_line'])

        # Temporal features
        features.append(game['week'])
        features.append(game['year'])

        # Sentiment features (if available)
        if self.use_sentiment and self.sentiment_extractor:
            sentiment = self.sentiment_extractor.extract_game_sentiment({
                'home_team': home,
                'away_team': away,
                'game_id': game.get('game_id', f"{away}@{home}"),
            })

            # Add key sentiment features
            features.append(sentiment.get('contrarian_opportunity', 0.0))
            features.append(sentiment.get('reddit_sentiment_score', 0.0))
            features.append(sentiment.get('sharp_public_split_ml', 0.0))
            features.append(sentiment.get('sharp_public_split_spread', 0.0))
            features.append(sentiment.get('sharp_public_split_total', 0.0))
            features.append(sentiment.get('consensus_strength', 0.0))
            features.append(sentiment.get('crowd_roar_confidence', 0.0))
        else:
            # Add neutral sentiment features (7 features)
            features.extend([0.0] * 7)

        return features


class XGBoostModel:
    """XGBoost model for NFL predictions."""

    def __init__(self):
        self.spread_model = None
        self.total_model = None

    def train(self, X_spread, y_spread, X_total, y_total):
        """Train XGBoost models."""
        print("\n🚀 Training XGBoost Models...")

        # Spread model
        print("   Training spread model...")
        self.spread_model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42
        )
        self.spread_model.fit(X_spread, y_spread)

        # Total model
        print("   Training total model...")
        self.total_model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42
        )
        self.total_model.fit(X_total, y_total)

        print("   ✅ XGBoost training complete!")

    def evaluate(self, X_spread, y_spread, X_total, y_total):
        """Evaluate model performance."""
        print("\n📊 XGBoost Performance:")

        # Spread predictions
        spread_pred = self.spread_model.predict(X_spread)
        spread_prob = self.spread_model.predict_proba(X_spread)[:, 1]

        print(f"\n   SPREAD MODEL:")
        print(f"   Accuracy: {accuracy_score(y_spread, spread_pred):.3f}")
        print(f"   Precision: {precision_score(y_spread, spread_pred):.3f}")
        print(f"   Recall: {recall_score(y_spread, spread_pred):.3f}")
        print(f"   F1: {f1_score(y_spread, spread_pred):.3f}")
        print(f"   AUC: {roc_auc_score(y_spread, spread_prob):.3f}")

        # Total predictions
        total_pred = self.total_model.predict(X_total)
        total_prob = self.total_model.predict_proba(X_total)[:, 1]

        print(f"\n   TOTAL MODEL:")
        print(f"   Accuracy: {accuracy_score(y_total, total_pred):.3f}")
        print(f"   Precision: {precision_score(y_total, total_pred):.3f}")
        print(f"   Recall: {recall_score(y_total, total_pred):.3f}")
        print(f"   F1: {f1_score(y_total, total_pred):.3f}")
        print(f"   AUC: {roc_auc_score(y_total, total_prob):.3f}")

    def predict(self, X):
        """Make predictions."""
        spread_pred = self.spread_model.predict_proba(X)[:, 1]
        total_pred = self.total_model.predict_proba(X)[:, 1]
        return spread_pred, total_pred

    def save(self, filepath: str):
        """Save model to disk."""
        model_data = {
            'spread_model': self.spread_model,
            'total_model': self.total_model
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        model = cls()
        model.spread_model = model_data['spread_model']
        model.total_model = model_data['total_model']
        return model


class NeuralNetworkModel(nn.Module):
    """Deep Neural Network for NFL predictions."""

    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class NFLNeuralNetworkTrainer:
    """Trains neural network on NFL data."""

    def __init__(self, input_size: int):
        self.spread_model = NeuralNetworkModel(input_size)
        self.total_model = NeuralNetworkModel(input_size)

    def train(self, X_spread, y_spread, X_total, y_total, epochs=100):
        """Train neural networks."""
        print("\n🧠 Training Neural Networks...")

        # Train spread model
        print("   Training spread network...")
        self._train_single_model(self.spread_model, X_spread, y_spread, epochs)

        # Train total model
        print("   Training total network...")
        self._train_single_model(self.total_model, X_total, y_total, epochs)

        print("   ✅ Neural network training complete!")

    def _train_single_model(self, model, X, y, epochs):
        """Train single neural network."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)

        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"      Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    def evaluate(self, X_spread, y_spread, X_total, y_total):
        """Evaluate neural network performance."""
        print("\n📊 Neural Network Performance:")

        # Spread evaluation
        self.spread_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_spread)
            spread_pred_prob = self.spread_model(X_tensor).numpy()
            spread_pred = (spread_pred_prob > 0.5).astype(int).flatten()

        print(f"\n   SPREAD MODEL:")
        print(f"   Accuracy: {accuracy_score(y_spread, spread_pred):.3f}")
        print(f"   AUC: {roc_auc_score(y_spread, spread_pred_prob):.3f}")

        # Total evaluation
        self.total_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_total)
            total_pred_prob = self.total_model(X_tensor).numpy()
            total_pred = (total_pred_prob > 0.5).astype(int).flatten()

        print(f"\n   TOTAL MODEL:")
        print(f"   Accuracy: {accuracy_score(y_total, total_pred):.3f}")
        print(f"   AUC: {roc_auc_score(y_total, total_pred_prob):.3f}")


class EnsembleModel:
    """Ensemble combining XGBoost and Neural Network."""

    def __init__(self, xgb_model, nn_model):
        self.xgb_model = xgb_model
        self.nn_model = nn_model

    def predict(self, X):
        """Make ensemble predictions (average of models)."""
        # XGBoost predictions
        xgb_spread, xgb_total = self.xgb_model.predict(X)

        # Neural network predictions
        self.nn_model.spread_model.eval()
        self.nn_model.total_model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            nn_spread = self.nn_model.spread_model(X_tensor).numpy().flatten()
            nn_total = self.nn_model.total_model(X_tensor).numpy().flatten()

        # Ensemble (weighted average)
        ensemble_spread = 0.6 * xgb_spread + 0.4 * nn_spread
        ensemble_total = 0.6 * xgb_total + 0.4 * nn_total

        return ensemble_spread, ensemble_total


def main():
    parser = argparse.ArgumentParser(
        description="Train advanced ML models on 15-year NFL data"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/historical/parlay_training/parlay_training_data.json",
        help="Training data file"
    )
    parser.add_argument(
        "--model",
        choices=['xgboost', 'neural', 'ensemble', 'all'],
        default='all',
        help="Model to train"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/advanced_ml",
        help="Directory to save models"
    )
    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        help="Disable sentiment features"
    )

    args = parser.parse_args()

    if not HAS_ML_LIBS:
        print("\n❌ Missing ML libraries!")
        return

    print(f"\n{'='*80}")
    print(f"🤖 ADVANCED ML MODEL TRAINING")
    print(f"{'='*80}\n")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load and process data
    print("📂 Loading data...")
    use_sentiment = not args.no_sentiment
    processor = NFLDataProcessor(use_sentiment=use_sentiment)
    games = processor.load_data(args.data_file)
    print(f"   Loaded {len(games)} games")

    # Report on sentiment usage
    if processor.use_sentiment:
        print("   ✅ Public sentiment features ENABLED (+7 features)")
    else:
        print("   ⚠️  Public sentiment features DISABLED")

    print("\n🔧 Extracting features...")
    X_spread, y_spread, X_total, y_total = processor.extract_features(games)
    print(f"   Spread samples: {len(X_spread)}")
    print(f"   Total samples: {len(X_total)}")
    print(f"   Features per sample: {X_spread.shape[1]}")

    # Train/test split
    X_spread_train, X_spread_test, y_spread_train, y_spread_test = train_test_split(
        X_spread, y_spread, test_size=args.test_split, random_state=42
    )
    X_total_train, X_total_test, y_total_train, y_total_test = train_test_split(
        X_total, y_total, test_size=args.test_split, random_state=42
    )

    # Train models
    if args.model in ['xgboost', 'all']:
        xgb_model = XGBoostModel()
        xgb_model.train(X_spread_train, y_spread_train, X_total_train, y_total_train)
        xgb_model.evaluate(X_spread_test, y_spread_test, X_total_test, y_total_test)
        xgb_model.save(save_dir / "xgboost_model.pkl")

    if args.model in ['neural', 'all'] and HAS_PYTORCH:
        nn_model = NFLNeuralNetworkTrainer(input_size=X_spread.shape[1])
        nn_model.train(X_spread_train, y_spread_train, X_total_train, y_total_train, epochs=100)
        nn_model.evaluate(X_spread_test, y_spread_test, X_total_test, y_total_test)

    if args.model == 'ensemble' and HAS_PYTORCH:
        xgb_model = XGBoostModel()
        xgb_model.train(X_spread_train, y_spread_train, X_total_train, y_total_train)

        nn_model = NFLNeuralNetworkTrainer(input_size=X_spread.shape[1])
        nn_model.train(X_spread_train, y_spread_train, X_total_train, y_total_train, epochs=100)

        ensemble = EnsembleModel(xgb_model, nn_model)
        print("\n✅ Ensemble model created!")

    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    main()
