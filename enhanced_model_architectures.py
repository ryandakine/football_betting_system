#!/usr/bin/env python3
"""
Enhanced Model Architectures for NFL Betting System
====================================================
Implements 10+ model system for maximum edge detection:

1-3: Base Ensemble Models (Spread, Total, Moneyline)
4: First Half Spread Model
5-6: Team Total Models (Home/Away)
7: XGBoost Super Ensemble
8: Neural Network Deep Model
9: Stacking Meta-Learner
10: Situational Specialist Filters
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import logging

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available - install with: pip install xgboost")


# ========================================================================
# Model 4: First Half Spread Predictor
# ========================================================================
class FirstHalfSpreadModel:
    """
    Predicts first half spread outcomes.
    Teams that start fast/slow have different first half patterns.
    """

    def __init__(self, name: str = "first_half_spread"):
        self.name = name
        self.scaler = StandardScaler()
        self.model = None
        self._build_model()

    def _build_model(self):
        """Build ensemble optimized for first half patterns."""
        estimators = [
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=10,
                random_state=42
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                random_state=42
            )),
        ]
        self.model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train on first half data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict first half home team cover probability."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict first half outcomes."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# ========================================================================
# Models 5-6: Team Total Predictors
# ========================================================================
class TeamTotalModel:
    """
    Predicts individual team total (over/under team points).
    Separate models for home and away teams.
    """

    def __init__(self, team_type: str = "home", name: str = "team_total"):
        self.team_type = team_type  # "home" or "away"
        self.name = f"{name}_{team_type}"
        self.scaler = StandardScaler()
        self.model = None
        self._build_model()

    def _build_model(self):
        """Build ensemble for team scoring prediction."""
        estimators = [
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=8,
                random_state=42
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )),
        ]
        self.model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train on team total data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict team total over probability."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict team total outcomes."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# ========================================================================
# Model 7: XGBoost Super Ensemble
# ========================================================================
class XGBoostSuperEnsemble:
    """
    XGBoost-based ensemble for spread/total predictions.
    Often outperforms sklearn on structured data.
    """

    def __init__(self, name: str = "xgboost_ensemble"):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available - install with: pip install xgboost")

        self.name = name
        self.scaler = StandardScaler()
        self.model = None
        self._build_model()

    def _build_model(self):
        """Build XGBoost classifier."""
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# ========================================================================
# Model 8: Neural Network Deep Model
# ========================================================================
class NeuralNetworkDeepModel:
    """
    Deep neural network for finding non-linear patterns.
    Can capture complex interactions between features.
    """

    def __init__(self, name: str = "neural_net_deep"):
        self.name = name
        self.scaler = StandardScaler()
        self.model = None
        self._build_model()

    def _build_model(self):
        """Build deep neural network."""
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32, 16),  # 4 hidden layers
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2 regularization
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train neural network."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# ========================================================================
# Model 9: Stacking Meta-Learner
# ========================================================================
class StackingMetaLearner:
    """
    Meta-model that learns to combine predictions from other models.
    Takes predictions from Models 1-8 and learns optimal weighting.
    """

    def __init__(self, name: str = "stacking_meta"):
        self.name = name
        self.scaler = StandardScaler()
        self.meta_model = None
        self._build_model()

    def _build_model(self):
        """Build meta-learner (simple logistic regression)."""
        self.meta_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            penalty='l2',
            C=1.0,
        )

    def fit(self, base_predictions: np.ndarray, y: np.ndarray):
        """
        Train meta-learner on base model predictions.

        Args:
            base_predictions: Shape (n_samples, n_base_models * 2)
                             Each base model contributes 2 probabilities
            y: True labels
        """
        X_scaled = self.scaler.fit_transform(base_predictions)
        self.meta_model.fit(X_scaled, y)
        return self

    def predict_proba(self, base_predictions: np.ndarray) -> np.ndarray:
        """Predict using base model predictions."""
        X_scaled = self.scaler.transform(base_predictions)
        return self.meta_model.predict_proba(X_scaled)

    def predict(self, base_predictions: np.ndarray) -> np.ndarray:
        """Predict outcomes."""
        X_scaled = self.scaler.transform(base_predictions)
        return self.meta_model.predict(X_scaled)


# ========================================================================
# Model 10: Situational Specialist System
# ========================================================================
class SituationalSpecialist:
    """
    Filters and weights predictions based on game context.
    Handles primetime games, divisional matchups, weather, etc.
    """

    def __init__(self):
        self.primetime_boost = 1.15  # 15% confidence boost
        self.divisional_boost = 1.10  # 10% confidence boost
        self.weather_adjust = {
            'dome': 1.0,
            'clear': 1.0,
            'rain': 0.95,
            'snow': 0.90,
            'wind_high': 0.85,
        }

    def adjust_prediction(
        self,
        base_prob: float,
        game_context: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """
        Adjust prediction based on situational context.

        Returns:
            (adjusted_probability, reasons)
        """
        adjusted = base_prob
        reasons = []

        # Primetime game adjustment
        if game_context.get('is_primetime'):
            adjusted = min(1.0, adjusted * self.primetime_boost)
            reasons.append("PRIMETIME_BOOST")

        # Divisional rivalry adjustment
        if game_context.get('is_divisional'):
            adjusted = min(1.0, adjusted * self.divisional_boost)
            reasons.append("DIVISIONAL_BOOST")

        # Weather adjustment
        weather = game_context.get('weather', 'clear')
        if weather in self.weather_adjust:
            adjusted *= self.weather_adjust[weather]
            if weather != 'clear' and weather != 'dome':
                reasons.append(f"WEATHER_{weather.upper()}")

        # Rest advantage (short week vs long rest)
        rest_diff = game_context.get('rest_differential', 0)
        if abs(rest_diff) >= 3:
            rest_boost = 1.0 + (rest_diff * 0.02)
            adjusted = min(1.0, max(0.0, adjusted * rest_boost))
            reasons.append(f"REST_ADVANTAGE_{'+' if rest_diff > 0 else ''}{rest_diff}")

        return adjusted, reasons


# ========================================================================
# Model Registry and Manager
# ========================================================================
class EnhancedModelRegistry:
    """
    Central registry for all 10 models.
    Handles loading, saving, and coordinated predictions.
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.specialist = SituationalSpecialist()

    def register_model(self, name: str, model: Any):
        """Add model to registry."""
        self.models[name] = model
        logger.info(f"Registered model: {name}")

    def get_model(self, name: str) -> Optional[Any]:
        """Get model by name."""
        return self.models.get(name)

    def get_all_predictions(
        self,
        X: np.ndarray,
        game_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from all models.

        Returns:
            Dict mapping model name to probabilities
        """
        predictions = {}

        for name, model in self.models.items():
            if model is None:
                logger.warning(f"Model {name} not loaded")
                continue

            try:
                probs = model.predict_proba(X)
                predictions[name] = probs
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")

        return predictions

    def get_ensemble_prediction(
        self,
        X: np.ndarray,
        game_context: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Get weighted ensemble prediction from all models.

        Args:
            X: Feature array
            game_context: Situational context for specialist
            weights: Optional model weights (default: equal weight)

        Returns:
            (final_probability, metadata)
        """
        predictions = self.get_all_predictions(X, game_context)

        if not predictions:
            return 0.5, {"error": "No models available"}

        # Default to equal weights
        if weights is None:
            weights = {name: 1.0 for name in predictions.keys()}

        # Weighted average
        total_weight = sum(weights.get(name, 0) for name in predictions.keys())
        weighted_sum = 0.0

        for name, probs in predictions.items():
            weight = weights.get(name, 1.0)
            # Take probability of positive class (index 1)
            weighted_sum += probs[0][1] * weight

        ensemble_prob = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Apply situational adjustments
        if game_context:
            adjusted_prob, reasons = self.specialist.adjust_prediction(
                ensemble_prob, game_context
            )
        else:
            adjusted_prob = ensemble_prob
            reasons = []

        metadata = {
            "base_prob": ensemble_prob,
            "adjusted_prob": adjusted_prob,
            "model_count": len(predictions),
            "adjustments": reasons,
            "individual_predictions": {
                name: float(probs[0][1]) for name, probs in predictions.items()
            }
        }

        return adjusted_prob, metadata


if __name__ == "__main__":
    print("Enhanced Model Architectures - 10 Model System")
    print("=" * 60)
    print("\n✅ Available Models:")
    print("1. Simple Ensemble (Spread)")
    print("2. Simple Ensemble (Total)")
    print("3. Simple Ensemble (Moneyline)")
    print("4. First Half Spread")
    print("5. Home Team Total")
    print("6. Away Team Total")
    print("7. XGBoost Super Ensemble")
    print("8. Neural Network Deep Model")
    print("9. Stacking Meta-Learner")
    print("10. Situational Specialist System")
    print("\n🚀 Ready to train the beast!")
