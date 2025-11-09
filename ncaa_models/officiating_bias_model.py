#!/usr/bin/env python3
"""
Model 11: Officiating Bias Model
Integrates conference crew analysis into predictions
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from .officiating_analysis.officiating_bias_detector import OfficiatingBiasDetector


class OfficiatingBiasModel:
    """
    Model 11: Officiating Bias Prediction Model

    Uses conference crew tendencies to adjust predictions:
    - Home bias by conference
    - Conference protection patterns
    - Rivalry game adjustments
    - Critical call tendencies

    Combines officiating profiles with ML to predict impact.
    """

    def __init__(self):
        self.detector = OfficiatingBiasDetector()
        self.rf_model = None
        self.gb_model = None
        self.is_trained = False

    def extract_officiating_features(
        self,
        home_team: str,
        away_team: str,
        home_conference: str,
        away_conference: str,
        officiating_conference: str,
        is_rivalry: bool = False,
        game_importance: float = 0.5
    ) -> Dict[str, float]:
        """
        Extract officiating-specific features.

        Returns:
            Dictionary of officiating features for ML models
        """
        # Get bias profile
        profile = self.detector.get_bias_adjustment(
            home_team=home_team,
            away_team=away_team,
            home_conference=home_conference,
            away_conference=away_conference,
            officiating_conference=officiating_conference,
            is_rivalry=is_rivalry
        )

        features = {
            # Direct bias metrics
            'off_spread_adjustment': profile['spread_adjustment'],
            'off_confidence_penalty': profile['confidence_penalty'],
            'off_risk_score': profile['risk_score'],

            # Conference profiles
            'off_home_bias': self.detector.profiles.get(officiating_conference,
                           self.detector.profiles['DEFAULT']).home_bias_score,
            'off_protection_score': self.detector.profiles.get(officiating_conference,
                                   self.detector.profiles['DEFAULT']).protection_score,
            'off_penalty_strictness': self.detector.profiles.get(officiating_conference,
                                     self.detector.profiles['DEFAULT']).penalty_strictness,
            'off_critical_call_bias': self.detector.profiles.get(officiating_conference,
                                     self.detector.profiles['DEFAULT']).critical_call_bias,

            # Game context
            'off_is_rivalry': 1.0 if is_rivalry else 0.0,
            'off_same_conference': 1.0 if home_conference == away_conference else 0.0,
            'off_protecting_home': 1.0 if officiating_conference == home_conference and \
                                         home_conference != away_conference else 0.0,
            'off_protecting_away': 1.0 if officiating_conference == away_conference and \
                                         home_conference != away_conference else 0.0,
            'off_game_importance': game_importance,

            # Derived features
            'off_bias_magnitude': abs(profile['spread_adjustment']),
            'off_total_adjustment': profile['spread_adjustment'] * (1 - profile['confidence_penalty']),
        }

        return features

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the officiating bias models.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (1 = home covers with bias adjustment, 0 = away covers)
        """
        # Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=15,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X, y)

        # Train Gradient Boosting
        self.gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        self.gb_model.fit(X, y)

        self.is_trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability with officiating bias adjustment.

        Returns:
            Array of shape (n_samples, 2) with probabilities for [away_covers, home_covers]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Get predictions from both models
        rf_proba = self.rf_model.predict_proba(X)
        gb_proba = self.gb_model.predict_proba(X)

        # Ensemble: 60% RF, 40% GB (RF better for this task)
        ensemble_proba = 0.6 * rf_proba + 0.4 * gb_proba

        return ensemble_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary outcome (1 = home covers, 0 = away covers)"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def get_spread_adjustment(
        self,
        home_team: str,
        away_team: str,
        home_conference: str,
        away_conference: str,
        officiating_conference: str,
        is_rivalry: bool = False
    ) -> float:
        """
        Get spread adjustment for a specific game.

        Returns:
            Spread adjustment in points (positive favors home team)
        """
        profile = self.detector.get_bias_adjustment(
            home_team=home_team,
            away_team=away_team,
            home_conference=home_conference,
            away_conference=away_conference,
            officiating_conference=officiating_conference,
            is_rivalry=is_rivalry
        )

        return profile['spread_adjustment']

    def get_confidence_adjustment(
        self,
        home_team: str,
        away_team: str,
        home_conference: str,
        away_conference: str,
        officiating_conference: str,
        is_rivalry: bool = False
    ) -> float:
        """
        Get confidence penalty for a specific game.

        Returns:
            Confidence penalty (0.0 = no penalty, 0.1 = reduce confidence by 10%)
        """
        profile = self.detector.get_bias_adjustment(
            home_team=home_team,
            away_team=away_team,
            home_conference=home_conference,
            away_conference=away_conference,
            officiating_conference=officiating_conference,
            is_rivalry=is_rivalry
        )

        return profile['confidence_penalty']

    def analyze_game(
        self,
        home_team: str,
        away_team: str,
        home_conference: str,
        away_conference: str,
        officiating_conference: str,
        is_rivalry: bool = False
    ) -> str:
        """
        Generate detailed officiating analysis for a game.

        Returns:
            Formatted analysis string
        """
        return self.detector.analyze_game(
            home_team=home_team,
            away_team=away_team,
            home_conference=home_conference,
            away_conference=away_conference,
            officiating_conference=officiating_conference,
            is_rivalry=is_rivalry
        )
