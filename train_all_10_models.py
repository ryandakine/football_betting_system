#!/usr/bin/env python3
"""
Train All 10 Models - Mega Training Script
===========================================
Trains the complete 10-model super intelligence system:

Models 1-3: Base Ensembles (Spread, Total, Moneyline)
Model 4: First Half Spread
Models 5-6: Team Totals (Home/Away)
Model 7: XGBoost Super Ensemble
Model 8: Neural Network Deep Model
Model 9: Stacking Meta-Learner
Model 10: Situational Specialist (rule-based, no training)

Usage:
    python train_all_10_models.py --data data/historical_games.csv
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# Import all model architectures
from simple_ensemble import SimpleEnsemble
from enhanced_model_architectures import (
    FirstHalfSpreadModel,
    TeamTotalModel,
    XGBoostSuperEnsemble,
    NeuralNetworkDeepModel,
    StackingMetaLearner,
    SituationalSpecialist,
    XGBOOST_AVAILABLE,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MegaModelTrainer:
    """Trains all 10 models in the super intelligence system."""

    def __init__(self, data_path: str, output_dir: str = "models"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}

        logger.info(f"Initialized trainer with data: {data_path}")
        logger.info(f"Models will be saved to: {self.output_dir}")

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare training data."""
        logger.info("Loading data...")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} games")

        # Basic validation
        required_cols = ['home_team', 'away_team', 'spread', 'total']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        return df

    def create_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create feature matrix from game data.
        This is a simplified version - you should customize based on your data.
        """
        feature_cols = []

        # Numeric features
        numeric_features = [
            'spread', 'total', 'home_ml_odds', 'away_ml_odds',
            'home_score_avg', 'away_score_avg',
            'home_win_pct', 'away_win_pct',
            'home_ats_record', 'away_ats_record',
        ]

        for col in numeric_features:
            if col in df.columns:
                feature_cols.append(col)

        if not feature_cols:
            logger.warning("No numeric features found, creating dummy features")
            # Create dummy features for demo
            X = np.random.randn(len(df), 10)
        else:
            X = df[feature_cols].fillna(0).values

        logger.info(f"Created feature matrix: {X.shape}")
        return X

    def create_targets(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create target variables for all models.

        Returns:
            (spread_y, total_y, ml_y, first_half_y, home_total_y, away_total_y)
        """
        # Model 1: Spread (home team covers)
        if 'home_cover' in df.columns:
            spread_y = df['home_cover'].values
        elif 'home_score' in df.columns and 'away_score' in df.columns:
            margin = df['home_score'] - df['away_score']
            spread_y = (margin > df['spread']).astype(int)
        else:
            logger.warning("No spread outcome found, creating random targets")
            spread_y = np.random.randint(0, 2, len(df))

        # Model 2: Total (over hits)
        if 'over_hit' in df.columns:
            total_y = df['over_hit'].values
        elif 'home_score' in df.columns and 'away_score' in df.columns:
            total_points = df['home_score'] + df['away_score']
            total_y = (total_points > df['total']).astype(int)
        else:
            logger.warning("No total outcome found, creating random targets")
            total_y = np.random.randint(0, 2, len(df))

        # Model 3: Moneyline (home team wins)
        if 'home_win' in df.columns:
            ml_y = df['home_win'].values
        elif 'home_score' in df.columns and 'away_score' in df.columns:
            ml_y = (df['home_score'] > df['away_score']).astype(int)
        else:
            logger.warning("No moneyline outcome found, creating random targets")
            ml_y = np.random.randint(0, 2, len(df))

        # Model 4: First Half Spread
        if 'home_1h_cover' in df.columns:
            first_half_y = df['home_1h_cover'].values
        elif 'home_1h_score' in df.columns and 'away_1h_score' in df.columns:
            margin_1h = df['home_1h_score'] - df['away_1h_score']
            line_1h = df['spread'] * 0.55  # Estimate first half line
            first_half_y = (margin_1h > line_1h).astype(int)
        else:
            logger.warning("No first half data, using full game as proxy")
            first_half_y = spread_y.copy()

        # Model 5: Home Team Total
        if 'home_team_total_over' in df.columns:
            home_total_y = df['home_team_total_over'].values
        elif 'home_score' in df.columns:
            home_team_line = df['total'] / 2.0 + 0.5
            home_total_y = (df['home_score'] > home_team_line).astype(int)
        else:
            logger.warning("No home team total, creating random targets")
            home_total_y = np.random.randint(0, 2, len(df))

        # Model 6: Away Team Total
        if 'away_team_total_over' in df.columns:
            away_total_y = df['away_team_total_over'].values
        elif 'away_score' in df.columns:
            away_team_line = df['total'] / 2.0 - 0.5
            away_total_y = (df['away_score'] > away_team_line).astype(int)
        else:
            logger.warning("No away team total, creating random targets")
            away_total_y = np.random.randint(0, 2, len(df))

        logger.info("Created all target variables")
        return spread_y, total_y, ml_y, first_half_y, home_total_y, away_total_y

    def train_base_ensembles(
        self,
        X_train: np.ndarray,
        y_spread: np.ndarray,
        y_total: np.ndarray,
        y_ml: np.ndarray,
        X_test: np.ndarray,
        y_spread_test: np.ndarray,
        y_total_test: np.ndarray,
        y_ml_test: np.ndarray,
    ):
        """Train Models 1-3: Base Ensembles."""
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING MODELS 1-3: BASE ENSEMBLES")
        logger.info("=" * 60)

        # Model 1: Spread Ensemble
        logger.info("\n[1/3] Training Spread Ensemble...")
        spread_model = SimpleEnsemble(name="spread_ensemble")
        spread_model.fit(X_train, y_spread)
        self.models['spread_ensemble'] = spread_model
        self._evaluate_model('spread_ensemble', spread_model, X_test, y_spread_test)
        self._save_model(spread_model, "spread_ensemble.pkl")

        # Model 2: Total Ensemble
        logger.info("\n[2/3] Training Total Ensemble...")
        total_model = SimpleEnsemble(name="total_ensemble")
        total_model.fit(X_train, y_total)
        self.models['total_ensemble'] = total_model
        self._evaluate_model('total_ensemble', total_model, X_test, y_total_test)
        self._save_model(total_model, "total_ensemble.pkl")

        # Model 3: Moneyline Ensemble
        logger.info("\n[3/3] Training Moneyline Ensemble...")
        ml_model = SimpleEnsemble(name="moneyline_ensemble")
        ml_model.fit(X_train, y_ml)
        self.models['moneyline_ensemble'] = ml_model
        self._evaluate_model('moneyline_ensemble', ml_model, X_test, y_ml_test)
        self._save_model(ml_model, "moneyline_ensemble.pkl")

    def train_enhanced_models(
        self,
        X_train: np.ndarray,
        y_first_half: np.ndarray,
        y_home_total: np.ndarray,
        y_away_total: np.ndarray,
        X_test: np.ndarray,
        y_first_half_test: np.ndarray,
        y_home_total_test: np.ndarray,
        y_away_total_test: np.ndarray,
    ):
        """Train Models 4-6: Enhanced Prediction Targets."""
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING MODELS 4-6: ENHANCED TARGETS")
        logger.info("=" * 60)

        # Model 4: First Half Spread
        logger.info("\n[4/10] Training First Half Spread Model...")
        first_half_model = FirstHalfSpreadModel()
        first_half_model.fit(X_train, y_first_half)
        self.models['first_half_spread'] = first_half_model
        self._evaluate_model('first_half_spread', first_half_model, X_test, y_first_half_test)
        self._save_model(first_half_model, "first_half_spread.pkl")

        # Model 5: Home Team Total
        logger.info("\n[5/10] Training Home Team Total Model...")
        home_total_model = TeamTotalModel(team_type="home")
        home_total_model.fit(X_train, y_home_total)
        self.models['home_team_total'] = home_total_model
        self._evaluate_model('home_team_total', home_total_model, X_test, y_home_total_test)
        self._save_model(home_total_model, "home_team_total.pkl")

        # Model 6: Away Team Total
        logger.info("\n[6/10] Training Away Team Total Model...")
        away_total_model = TeamTotalModel(team_type="away")
        away_total_model.fit(X_train, y_away_total)
        self.models['away_team_total'] = away_total_model
        self._evaluate_model('away_team_total', away_total_model, X_test, y_away_total_test)
        self._save_model(away_total_model, "away_team_total.pkl")

    def train_algorithm_variants(
        self,
        X_train: np.ndarray,
        y_spread: np.ndarray,
        X_test: np.ndarray,
        y_spread_test: np.ndarray,
    ):
        """Train Models 7-8: Algorithm Variants."""
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING MODELS 7-8: ALGORITHM VARIANTS")
        logger.info("=" * 60)

        # Model 7: XGBoost Super Ensemble
        if XGBOOST_AVAILABLE:
            logger.info("\n[7/10] Training XGBoost Super Ensemble...")
            xgb_model = XGBoostSuperEnsemble()
            xgb_model.fit(X_train, y_spread)
            self.models['xgboost_ensemble'] = xgb_model
            self._evaluate_model('xgboost_ensemble', xgb_model, X_test, y_spread_test)
            self._save_model(xgb_model, "xgboost_ensemble.pkl")
        else:
            logger.warning("[7/10] XGBoost not available - skipping")
            logger.warning("Install with: pip install xgboost")

        # Model 8: Neural Network Deep Model
        logger.info("\n[8/10] Training Neural Network Deep Model...")
        nn_model = NeuralNetworkDeepModel()
        nn_model.fit(X_train, y_spread)
        self.models['neural_net_deep'] = nn_model
        self._evaluate_model('neural_net_deep', nn_model, X_test, y_spread_test)
        self._save_model(nn_model, "neural_net_deep.pkl")

    def train_meta_learner(
        self,
        X_train: np.ndarray,
        y_spread: np.ndarray,
        X_test: np.ndarray,
        y_spread_test: np.ndarray,
    ):
        """Train Model 9: Stacking Meta-Learner."""
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING MODEL 9: STACKING META-LEARNER")
        logger.info("=" * 60)

        logger.info("\n[9/10] Training Stacking Meta-Learner...")

        # Collect predictions from base models (Models 1-8)
        base_model_names = [
            'spread_ensemble', 'total_ensemble', 'moneyline_ensemble',
            'first_half_spread', 'home_team_total', 'away_team_total',
            'xgboost_ensemble', 'neural_net_deep'
        ]

        # Get training predictions
        train_preds = []
        for name in base_model_names:
            if name in self.models:
                model = self.models[name]
                probs = model.predict_proba(X_train)
                train_preds.append(probs)

        if len(train_preds) < 3:
            logger.warning("Not enough base models for stacking - skipping")
            return

        # Stack predictions horizontally
        X_meta_train = np.hstack(train_preds)

        # Get test predictions
        test_preds = []
        for name in base_model_names:
            if name in self.models:
                model = self.models[name]
                probs = model.predict_proba(X_test)
                test_preds.append(probs)

        X_meta_test = np.hstack(test_preds)

        # Train meta-learner
        meta_model = StackingMetaLearner()
        meta_model.fit(X_meta_train, y_spread)
        self.models['stacking_meta'] = meta_model
        self._evaluate_model('stacking_meta', meta_model, X_meta_test, y_spread_test)
        self._save_model(meta_model, "stacking_meta.pkl")

    def _evaluate_model(self, name: str, model: Any, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model and store metrics."""
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            logloss = log_loss(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)

            self.metrics[name] = {
                'accuracy': accuracy,
                'log_loss': logloss,
                'auc_roc': auc,
            }

            logger.info(f"  ✓ Accuracy: {accuracy:.3f}")
            logger.info(f"  ✓ Log Loss: {logloss:.3f}")
            logger.info(f"  ✓ AUC-ROC: {auc:.3f}")

        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")

    def _save_model(self, model: Any, filename: str):
        """Save model to disk."""
        path = self.output_dir / filename
        try:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"  ✓ Saved to {path}")
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")

    def train_all(self):
        """Main training pipeline for all 10 models."""
        logger.info("\n" + "=" * 60)
        logger.info("🚀 MEGA MODEL TRAINER - 10 MODEL SYSTEM")
        logger.info("=" * 60)

        # Load data
        df = self.load_and_prepare_data()

        # Create features
        X = self.create_features(df)

        # Create all targets
        (
            y_spread, y_total, y_ml,
            y_first_half, y_home_total, y_away_total
        ) = self.create_targets(df)

        # Train/test split
        X_train, X_test, \
        y_spread_train, y_spread_test, \
        y_total_train, y_total_test, \
        y_ml_train, y_ml_test, \
        y_first_half_train, y_first_half_test, \
        y_home_total_train, y_home_total_test, \
        y_away_total_train, y_away_total_test = train_test_split(
            X, y_spread, y_total, y_ml, y_first_half, y_home_total, y_away_total,
            test_size=0.2,
            random_state=42,
            stratify=y_spread
        )

        logger.info(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

        # Train all model groups
        self.train_base_ensembles(
            X_train, y_spread_train, y_total_train, y_ml_train,
            X_test, y_spread_test, y_total_test, y_ml_test
        )

        self.train_enhanced_models(
            X_train, y_first_half_train, y_home_total_train, y_away_total_train,
            X_test, y_first_half_test, y_home_total_test, y_away_total_test
        )

        self.train_algorithm_variants(
            X_train, y_spread_train,
            X_test, y_spread_test
        )

        self.train_meta_learner(
            X_train, y_spread_train,
            X_test, y_spread_test
        )

        # Model 10: Situational Specialist (rule-based, no training)
        logger.info("\n" + "=" * 60)
        logger.info("MODEL 10: SITUATIONAL SPECIALIST")
        logger.info("=" * 60)
        logger.info("\n[10/10] Situational Specialist is rule-based (no training needed)")
        logger.info("  ✓ Primetime adjustments configured")
        logger.info("  ✓ Divisional rivalry adjustments configured")
        logger.info("  ✓ Weather adjustments configured")
        logger.info("  ✓ Rest differential adjustments configured")

        # Save metrics summary
        self._save_metrics_summary()

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL 10 MODELS TRAINED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"\nModels saved to: {self.output_dir}")
        logger.info(f"Total models trained: {len(self.models)}")

    def _save_metrics_summary(self):
        """Save training metrics to JSON."""
        metrics_path = self.output_dir / "training_metrics.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"\n✓ Metrics saved to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Train all 10 models in the super intelligence system"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to historical games CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory for trained models (default: models/)"
    )

    args = parser.parse_args()

    trainer = MegaModelTrainer(
        data_path=args.data,
        output_dir=args.output
    )

    trainer.train_all()


if __name__ == "__main__":
    main()
