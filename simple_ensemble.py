#!/usr/bin/env python3
"""
Simple Ensemble Model
====================
A simple voting ensemble that can be properly pickled/unpickled.
This module ensures pickle can find the class when loading models.
"""

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class SimpleEnsemble:
    """
    Simple voting ensemble for binary classification.
    Uses logistic regression, random forest, and gradient boosting.

    This class is defined in a module (not __main__) so pickle works correctly.
    """

    def __init__(self, name: str = "ensemble"):
        self.name = name
        self.scaler = StandardScaler()
        self.model = None
        self._build_ensemble()

    def _build_ensemble(self):
        """Build the voting ensemble"""
        estimators = [
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42))
        ]

        self.model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability averaging
            n_jobs=-1
        )

    def fit(self, X, y):
        """Train the ensemble"""
        logger.info(f"Training {self.name} on {len(X)} samples")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train ensemble
        self.model.fit(X_scaled, y)

        # Training accuracy
        train_acc = self.model.score(X_scaled, y)
        logger.info(f"{self.name} training accuracy: {train_acc:.3f}")

        return self

    def predict(self, X):
        """Predict class labels"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Predict class probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def score(self, X, y):
        """Calculate accuracy score"""
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)


# For backwards compatibility with old pickled models
# If models were pickled as __main__.SimpleEnsemble, this helps
def load_legacy_model(pickle_file):
    """
    Load models that were pickled with __main__.SimpleEnsemble
    """
    import sys
    import pickle

    # Temporarily add this module to __main__ for unpickling
    sys.modules['__main__'].SimpleEnsemble = SimpleEnsemble

    try:
        with open(pickle_file, 'rb') as f:
            model = pickle.load(f)
        return model
    finally:
        # Clean up
        if hasattr(sys.modules['__main__'], 'SimpleEnsemble'):
            delattr(sys.modules['__main__'], 'SimpleEnsemble')
