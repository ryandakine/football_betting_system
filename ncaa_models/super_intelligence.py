#!/usr/bin/env python3
"""
10-Model Super Intelligence System for NCAA Football
Combines multiple ML approaches for superior predictions
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss
import xgboost as xgb
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class BaseNCAAModel:
    """Base class for all NCAA prediction models"""

    def __init__(self, name, model_type='regression'):
        self.name = name
        self.model_type = model_type  # 'regression' or 'classification'
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.is_trained = False

    def train(self, X, y, **kwargs):
        """Train the model"""
        raise NotImplementedError

    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError(f"{self.name} not trained yet!")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Predict probabilities (for classification models)"""
        if self.model_type != 'classification':
            raise ValueError("predict_proba only for classification models")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, filepath):
        """Save model to disk"""
        model_data = {
            'name': self.name,
            'model_type': self.model_type,
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Saved {self.name} to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        instance = cls(model_data['name'], model_data['model_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_importance = model_data['feature_importance']
        instance.is_trained = model_data['is_trained']

        logger.info(f"Loaded {instance.name} from {filepath}")
        return instance


class SpreadEnsembleModel(BaseNCAAModel):
    """Model 1: Spread Prediction Ensemble"""

    def __init__(self):
        super().__init__("Spread Ensemble", "regression")

    def train(self, X, y, **kwargs):
        """Train ensemble of models for spread prediction"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train multiple base models
        rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
        ridge = Ridge(alpha=1.0)

        # Fit models
        rf.fit(X_scaled, y)
        gb.fit(X_scaled, y)
        ridge.fit(X_scaled, y)

        # Store ensemble
        self.model = {'rf': rf, 'gb': gb, 'ridge': ridge}
        self.is_trained = True

        # Feature importance (from RF)
        self.feature_importance = dict(zip(X.columns, rf.feature_importances_))

        logger.info(f"{self.name} trained on {len(X)} games")

    def predict(self, X):
        """Ensemble prediction (average of 3 models)"""
        X_scaled = self.scaler.transform(X)

        pred_rf = self.model['rf'].predict(X_scaled)
        pred_gb = self.model['gb'].predict(X_scaled)
        pred_ridge = self.model['ridge'].predict(X_scaled)

        # Weighted average (RF gets more weight)
        ensemble_pred = (pred_rf * 0.4 + pred_gb * 0.4 + pred_ridge * 0.2)
        return ensemble_pred


class TotalEnsembleModel(BaseNCAAModel):
    """Model 2: Total Points Prediction Ensemble"""

    def __init__(self):
        super().__init__("Total Ensemble", "regression")

    def train(self, X, y, **kwargs):
        """Train ensemble for total points"""
        X_scaled = self.scaler.fit_transform(X)

        # Ensemble models
        rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)

        rf.fit(X_scaled, y)
        gb.fit(X_scaled, y)

        self.model = {'rf': rf, 'gb': gb}
        self.is_trained = True
        self.feature_importance = dict(zip(X.columns, rf.feature_importances_))

        logger.info(f"{self.name} trained on {len(X)} games")

    def predict(self, X):
        """Ensemble prediction"""
        X_scaled = self.scaler.transform(X)

        pred_rf = self.model['rf'].predict(X_scaled)
        pred_gb = self.model['gb'].predict(X_scaled)

        return (pred_rf + pred_gb) / 2


class MoneylineEnsembleModel(BaseNCAAModel):
    """Model 3: Moneyline (Win Probability) Ensemble"""

    def __init__(self):
        super().__init__("Moneyline Ensemble", "classification")

    def train(self, X, y, **kwargs):
        """Train ensemble for win probability"""
        X_scaled = self.scaler.fit_transform(X)

        # Classification models
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)

        rf.fit(X_scaled, y)
        lr.fit(X_scaled, y)

        self.model = {'rf': rf, 'lr': lr}
        self.is_trained = True
        self.feature_importance = dict(zip(X.columns, rf.feature_importances_))

        logger.info(f"{self.name} trained on {len(X)} games")

    def predict_proba(self, X):
        """Ensemble probability prediction"""
        X_scaled = self.scaler.transform(X)

        prob_rf = self.model['rf'].predict_proba(X_scaled)
        prob_lr = self.model['lr'].predict_proba(X_scaled)

        # Average probabilities
        return (prob_rf + prob_lr) / 2


class FirstHalfSpreadModel(BaseNCAAModel):
    """Model 4: First Half Spread Prediction"""

    def __init__(self):
        super().__init__("1H Spread", "regression")

    def train(self, X, y, **kwargs):
        """Train for first half spreads"""
        X_scaled = self.scaler.fit_transform(X)

        # Use gradient boosting (good for temporal patterns)
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        gb.fit(X_scaled, y)

        self.model = gb
        self.is_trained = True
        self.feature_importance = dict(zip(X.columns, gb.feature_importances_))

        logger.info(f"{self.name} trained on {len(X)} games")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class HomeTeamTotalModel(BaseNCAAModel):
    """Model 5: Home Team Total Points"""

    def __init__(self):
        super().__init__("Home Team Total", "regression")

    def train(self, X, y, **kwargs):
        """Train for home team scoring"""
        X_scaled = self.scaler.fit_transform(X)

        # Random Forest good for team-specific patterns
        rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
        rf.fit(X_scaled, y)

        self.model = rf
        self.is_trained = True
        self.feature_importance = dict(zip(X.columns, rf.feature_importances_))

        logger.info(f"{self.name} trained on {len(X)} games")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class AwayTeamTotalModel(BaseNCAAModel):
    """Model 6: Away Team Total Points"""

    def __init__(self):
        super().__init__("Away Team Total", "regression")

    def train(self, X, y, **kwargs):
        """Train for away team scoring"""
        X_scaled = self.scaler.fit_transform(X)

        rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
        rf.fit(X_scaled, y)

        self.model = rf
        self.is_trained = True
        self.feature_importance = dict(zip(X.columns, rf.feature_importances_))

        logger.info(f"{self.name} trained on {len(X)} games")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class AltSpreadModel(BaseNCAAModel):
    """Model 7: Alternative Spread Lines"""

    def __init__(self):
        super().__init__("Alt Spread", "regression")

    def train(self, X, y, **kwargs):
        """Train for alternative spread predictions"""
        X_scaled = self.scaler.fit_transform(X)

        # Ensemble for spread variance
        rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)

        rf.fit(X_scaled, y)
        gb.fit(X_scaled, y)

        self.model = {'rf': rf, 'gb': gb}
        self.is_trained = True
        self.feature_importance = dict(zip(X.columns, rf.feature_importances_))

        logger.info(f"{self.name} trained on {len(X)} games")

    def predict(self, X):
        """Predict with confidence intervals"""
        X_scaled = self.scaler.transform(X)

        pred_rf = self.model['rf'].predict(X_scaled)
        pred_gb = self.model['gb'].predict(X_scaled)

        # Return mean and std for alt lines
        mean_pred = (pred_rf + pred_gb) / 2
        std_pred = np.std([pred_rf, pred_gb], axis=0)

        return {'mean': mean_pred, 'std': std_pred}


class XGBoostSuperEnsemble(BaseNCAAModel):
    """Model 8: XGBoost Super Ensemble - Gradient Boosting Beast"""

    def __init__(self):
        super().__init__("XGBoost Super", "regression")

    def train(self, X, y, **kwargs):
        """Train XGBoost with optimal hyperparameters"""
        X_scaled = self.scaler.fit_transform(X)

        # XGBoost with advanced parameters
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )

        xgb_model.fit(X_scaled, y)

        self.model = xgb_model
        self.is_trained = True
        self.feature_importance = dict(zip(X.columns, xgb_model.feature_importances_))

        logger.info(f"{self.name} trained on {len(X)} games")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class NeuralNetworkDeepModel(BaseNCAAModel):
    """Model 9: Neural Network Deep Model - Deep Learning Edge Finder"""

    def __init__(self):
        super().__init__("Neural Net Deep", "regression")

    def train(self, X, y, **kwargs):
        """Train deep neural network"""
        try:
            from sklearn.neural_network import MLPRegressor

            X_scaled = self.scaler.fit_transform(X)

            # Multi-layer perceptron
            nn = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )

            nn.fit(X_scaled, y)

            self.model = nn
            self.is_trained = True

            logger.info(f"{self.name} trained on {len(X)} games")

        except ImportError:
            logger.warning("sklearn MLPRegressor not available, using fallback")
            # Fallback to gradient boosting
            self.model = GradientBoostingRegressor(n_estimators=200, random_state=42)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class StackingMetaLearner(BaseNCAAModel):
    """Model 10: Stacking Meta-Learner - Learns from Other Models"""

    def __init__(self):
        super().__init__("Stacking Meta", "regression")
        self.base_models = []

    def train(self, X, y, base_models, **kwargs):
        """
        Train meta-learner on predictions from base models
        
        Args:
            X: Original features
            y: Targets
            base_models: List of trained base models
        """
        self.base_models = base_models

        # Generate meta-features (predictions from base models)
        meta_features = []

        for model in base_models:
            if model.is_trained:
                preds = model.predict(X)
                meta_features.append(preds.reshape(-1, 1))

        # Combine meta-features with original features (subset)
        if meta_features:
            meta_X = np.hstack(meta_features)

            # Add top original features
            X_scaled = self.scaler.fit_transform(X)
            top_features = X_scaled[:, :20]  # Top 20 original features

            combined_X = np.hstack([meta_X, top_features])

            # Train meta-model (Ridge for regularization)
            meta_model = Ridge(alpha=1.0)
            meta_model.fit(combined_X, y)

            self.model = meta_model
            self.is_trained = True

            logger.info(f"{self.name} trained on {len(base_models)} base models")

    def predict(self, X):
        """Predict using stacked models"""
        # Get predictions from base models
        meta_features = []
        for model in self.base_models:
            if model.is_trained:
                preds = model.predict(X)
                meta_features.append(preds.reshape(-1, 1))

        # Combine with top features
        if meta_features:
            meta_X = np.hstack(meta_features)
            X_scaled = self.scaler.transform(X)
            top_features = X_scaled[:, :20]

            combined_X = np.hstack([meta_X, top_features])

            return self.model.predict(combined_X)
        else:
            raise ValueError("No base models available for meta-learning")


class SuperIntelligenceOrchestrator:
    """
    Orchestrates all 12 models for comprehensive predictions

    Models 1-10: Core prediction models (spread, total, ML, etc.)
    Model 11: Officiating Bias Model (conference crew analysis)
    Model 12: Prop Bet Specialist Model (player/team props)
    """

    def __init__(self, models_dir="models/ncaa"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Import specialized models
        from .officiating_bias_model import OfficiatingBiasModel
        from .prop_bet_model import PropBetSpecialistModel

        # Initialize all 12 models
        self.models = {
            # Core prediction models (1-10)
            'spread_ensemble': SpreadEnsembleModel(),
            'total_ensemble': TotalEnsembleModel(),
            'moneyline_ensemble': MoneylineEnsembleModel(),
            'first_half_spread': FirstHalfSpreadModel(),
            'home_team_total': HomeTeamTotalModel(),
            'away_team_total': AwayTeamTotalModel(),
            'alt_spread': AltSpreadModel(),
            'xgboost_super': XGBoostSuperEnsemble(),
            'neural_net_deep': NeuralNetworkDeepModel(),
            'stacking_meta': StackingMetaLearner(),

            # Specialized models (11-12)
            'officiating_bias': OfficiatingBiasModel(),
            'prop_bet_specialist': PropBetSpecialistModel()
        }

    def train_all(self, feature_engineer, seasons=[2023, 2024]):
        """Train all 12 models on historical data"""
        logger.info("Training 12-Model Super Intelligence System...")
        logger.info("="*70)

        for year in seasons:
            logger.info(f"\nLoading {year} season data...")
            games = feature_engineer.load_season_data(year)

            # Train each model type
            model_targets = {
                'spread_ensemble': 'spread',
                'total_ensemble': 'total',
                'moneyline_ensemble': 'moneyline',
                'first_half_spread': '1h_spread',
                'home_team_total': 'home_total',
                'away_team_total': 'away_total',
                'alt_spread': 'spread',  # Same as spread
                'xgboost_super': 'spread',  # Can train on any target
                'neural_net_deep': 'spread'
            }

            datasets = {}

            # Create datasets for each target
            for model_name, target in model_targets.items():
                if model_name not in datasets or datasets[model_name] is None:
                    X, y = feature_engineer.create_dataset(games, year, target=target)
                    if X is not None and len(X) > 0:
                        datasets[model_name] = (X, y)
                        logger.info(f"  {target}: {len(X)} games")

            # Train models (except stacking meta)
            base_models_for_stacking = []

            for model_name, model in self.models.items():
                if model_name == 'stacking_meta':
                    continue  # Train last

                if model_name in datasets:
                    X, y = datasets[model_name]
                    logger.info(f"\nTraining {model.name}...")
                    model.train(X, y)
                    base_models_for_stacking.append(model)

            # Train stacking meta-learner
            if 'spread_ensemble' in datasets:
                X, y = datasets['spread_ensemble']
                logger.info(f"\nTraining Stacking Meta-Learner...")
                self.models['stacking_meta'].train(X, y, base_models=base_models_for_stacking[:5])

        logger.info("\n" + "="*70)
        logger.info("✅ All models trained successfully!")

    def predict(self, features):
        """
        Generate predictions from all models
        
        Returns dict with predictions from each model
        """
        predictions = {}

        # Get predictions from each model
        for model_name, model in self.models.items():
            if model.is_trained:
                try:
                    if model.model_type == 'classification':
                        # Get probability of home win
                        probs = model.predict_proba(features)
                        predictions[model_name] = probs[:, 1]  # Probability of class 1 (home win)
                    else:
                        preds = model.predict(features)
                        if isinstance(preds, dict):
                            predictions[model_name] = preds
                        else:
                            predictions[model_name] = preds
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")

        return predictions

    def save_all(self):
        """Save all trained models"""
        for model_name, model in self.models.items():
            if model.is_trained:
                filepath = self.models_dir / f"{model_name}.pkl"
                model.save(filepath)

    def load_all(self):
        """Load all saved models"""
        for model_name in self.models.keys():
            filepath = self.models_dir / f"{model_name}.pkl"
            if filepath.exists():
                self.models[model_name] = BaseNCAAModel.load(filepath)

    def get_consensus_prediction(self, predictions):
        """
        Generate consensus prediction from all models
        """
        consensus = {}

        # Spread consensus (average of spread models)
        spread_models = ['spread_ensemble', 'xgboost_super', 'neural_net_deep', 'stacking_meta']
        spread_preds = [predictions[m] for m in spread_models if m in predictions]
        if spread_preds:
            consensus['spread'] = np.mean(spread_preds, axis=0)
            consensus['spread_confidence'] = 1 - np.std(spread_preds, axis=0) / 10  # Lower std = higher confidence

        # Total consensus
        if 'total_ensemble' in predictions:
            # Can also derive from team totals
            if 'home_team_total' in predictions and 'away_team_total' in predictions:
                derived_total = predictions['home_team_total'] + predictions['away_team_total']
                consensus['total'] = (predictions['total_ensemble'] + derived_total) / 2
            else:
                consensus['total'] = predictions['total_ensemble']

        # Win probability
        if 'moneyline_ensemble' in predictions:
            consensus['win_probability'] = predictions['moneyline_ensemble']

        # First half
        if 'first_half_spread' in predictions:
            consensus['1h_spread'] = predictions['first_half_spread']

        # Team totals
        if 'home_team_total' in predictions:
            consensus['home_total'] = predictions['home_team_total']
        if 'away_team_total' in predictions:
            consensus['away_total'] = predictions['away_team_total']

        return consensus
