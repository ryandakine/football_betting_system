#!/usr/bin/env python3
"""
Simple Model Predictor
======================
Loads existing ensemble models and makes predictions.
Much simpler than GGUF - just uses sklearn models.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import sys

# Import SimpleEnsemble so pickle can find it when loading models
from simple_ensemble import SimpleEnsemble

logger = logging.getLogger(__name__)

class SimpleModelPredictor:
    """Load and use existing trained ensemble models"""
    
    def __init__(self):
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load spread/total/moneyline ensemble models"""
        models_dir = Path("models")
        model_files = {
            'spread': 'spread_ensemble.pkl',
            'total': 'total_ensemble.pkl',
            'moneyline': 'moneyline_ensemble.pkl'
        }

        for name, filename in model_files.items():
            path = models_dir / filename
            if path.exists():
                try:
                    # Add __main__.SimpleEnsemble redirect for legacy models
                    sys.modules['__main__'].SimpleEnsemble = SimpleEnsemble

                    with open(path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    logger.info(f"✅ Loaded {name} model")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load {name} model: {e}")
                finally:
                    # Clean up
                    if hasattr(sys.modules['__main__'], 'SimpleEnsemble'):
                        delattr(sys.modules['__main__'], 'SimpleEnsemble')

        if not self.models:
            logger.warning("⚠️  No models loaded - predictions will be random")
    
    def extract_features(self, game: Dict[str, Any]) -> np.ndarray:
        """
        Extract features that the ensemble models expect.
        Based on training features from train_and_optimize_system.py
        """
        features = []
        
        # Use simple features available in game data
        features.append(float(game.get('spread', 0.0)))
        features.append(float(game.get('total', 44.5)))
        features.append(float(game.get('is_division_game', 0)))
        features.append(float(game.get('is_primetime', 0)))
        features.append(float(game.get('is_dome', 0)))
        features.append(float(game.get('attendance', 50000)) / 75000.0)  # Normalize
        
        return np.array([features])
    
    def get_spread_prediction(self, game: Dict[str, Any]) -> float:
        """Get home team win probability for spread"""
        if 'spread' not in self.models:
            return 0.5
        
        try:
            features = self.extract_features(game)
            proba = self.models['spread'].predict_proba(features)[0]
            # Return probability of home team covering
            return float(proba[1]) if len(proba) > 1 else 0.5
        except Exception as e:
            logger.debug(f"Spread prediction failed: {e}")
            return 0.5
    
    def get_total_prediction(self, game: Dict[str, Any]) -> float:
        """Get over probability"""
        if 'total' not in self.models:
            return 0.5
        
        try:
            features = self.extract_features(game)
            proba = self.models['total'].predict_proba(features)[0]
            # Return probability of over
            return float(proba[1]) if len(proba) > 1 else 0.5
        except Exception as e:
            logger.debug(f"Total prediction failed: {e}")
            return 0.5
    
    def get_moneyline_prediction(self, game: Dict[str, Any]) -> float:
        """Get home team win probability"""
        if 'moneyline' not in self.models:
            return 0.5
        
        try:
            features = self.extract_features(game)
            proba = self.models['moneyline'].predict_proba(features)[0]
            # Return probability of home team winning
            return float(proba[1]) if len(proba) > 1 else 0.5
        except Exception as e:
            logger.debug(f"Moneyline prediction failed: {e}")
            return 0.5
