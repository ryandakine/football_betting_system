#!/usr/bin/env python3
"""
Interaction World Model for NCAA Betting System
================================================
Discovers and learns 2-way and 3-way interactions between your 12 models.
Boosts predictions when known good interactions align.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from itertools import combinations
import pickle

logger = logging.getLogger(__name__)


@dataclass
class InteractionEffect:
    """Represents a learned interaction between models"""
    models: Tuple[str, ...]
    interaction_strength: float  # 0-1 scale
    confidence: float  # How sure we are about this interaction
    prediction_boost: float  # % boost when interaction fires
    frequency: int  # How often we see this interaction
    avg_model_agreement: float  # Average agreement between models in interaction


class InteractionWorldModel:
    """
    Learns and applies interaction effects between NCAA prediction models.
    
    Key insight: Individual models aren't independent.
    - When Model 1 and Model 7 BOTH agree → confidence should boost
    - When Model 3, Model 8, Model 11 all align → even stronger signal
    - Some interactions predict edge cases better than base models
    """
    
    def __init__(self, cache_path: Optional[str] = None):
        self.cache_path = cache_path or "models/interaction_world_model.pkl"
        self.interactions_2way: Dict[Tuple[str, str], InteractionEffect] = {}
        self.interactions_3way: Dict[Tuple[str, str, str], InteractionEffect] = {}
        self.model_names = [
            'spread_ensemble', 'total_ensemble', 'moneyline_ensemble',
            'ncaa_rf', 'ncaa_gb',
            'spread_edges', 'total_edges', 'moneyline_edges',
            'Market Consensus', 'Contrarian Model',
            'Referee Model', 'Injury Model'  # Will be added by system
        ]
        self.interaction_history: List[Dict] = []
        self._load_cache()
    
    def _load_cache(self):
        """Load cached interactions from disk"""
        cache_file = Path(self.cache_path)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.interactions_2way = data.get('interactions_2way', {})
                    self.interactions_3way = data.get('interactions_3way', {})
                    self.interaction_history = data.get('history', [])
                    logger.info(f"✅ Loaded {len(self.interactions_2way)} 2-way interactions")
                    logger.info(f"✅ Loaded {len(self.interactions_3way)} 3-way interactions")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load interaction cache: {e}")
    
    def _save_cache(self):
        """Save learned interactions to disk"""
        try:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'interactions_2way': self.interactions_2way,
                    'interactions_3way': self.interactions_3way,
                    'history': self.interaction_history[-1000:]  # Keep last 1000
                }, f)
        except Exception as e:
            logger.warning(f"⚠️  Failed to save interaction cache: {e}")
    
    def record_prediction_batch(self, 
                               game_id: str,
                               model_predictions: Dict[str, float],  # {model_name: confidence}
                               actual_result: Optional[float] = None):
        """
        Record a batch of model predictions for a single game.
        This trains the interaction model over time.
        
        Args:
            game_id: Unique game identifier
            model_predictions: {model_name: confidence_score}
            actual_result: If known, actual game result for feedback
        """
        self.interaction_history.append({
            'game_id': game_id,
            'predictions': model_predictions,
            'result': actual_result,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })
        
        # Learn from patterns when we have results
        if actual_result is not None and len(self.interaction_history) % 10 == 0:
            self._learn_from_history()
    
    def _learn_from_history(self, lookback: int = 100):
        """
        Analyze recent predictions to discover interaction patterns.
        Called periodically to update interaction weights.
        """
        recent = self.interaction_history[-lookback:]
        if len(recent) < 10:
            return
        
        # Find 2-way interactions
        for model1, model2 in combinations(self.model_names, 2):
            agreement_score = self._calc_model_agreement(recent, [model1, model2])
            if agreement_score > 0.65:  # Strong agreement
                key = tuple(sorted([model1, model2]))
                if key not in self.interactions_2way:
                    self.interactions_2way[key] = InteractionEffect(
                        models=key,
                        interaction_strength=min(agreement_score, 1.0),
                        confidence=0.3,  # Start low
                        prediction_boost=0.05,  # 5% boost
                        frequency=1,
                        avg_model_agreement=agreement_score
                    )
                else:
                    # Update existing
                    existing = self.interactions_2way[key]
                    existing.frequency += 1
                    existing.avg_model_agreement = (
                        (existing.avg_model_agreement * (existing.frequency - 1) + agreement_score) 
                        / existing.frequency
                    )
                    existing.confidence = min(existing.frequency / 20, 1.0)
        
        # Find 3-way interactions
        for model1, model2, model3 in combinations(self.model_names, 3):
            agreement_score = self._calc_model_agreement(recent, [model1, model2, model3])
            if agreement_score > 0.70:  # Very strong agreement
                key = tuple(sorted([model1, model2, model3]))
                if key not in self.interactions_3way:
                    self.interactions_3way[key] = InteractionEffect(
                        models=key,
                        interaction_strength=min(agreement_score, 1.0),
                        confidence=0.3,
                        prediction_boost=0.08,  # 8% boost for 3-way
                        frequency=1,
                        avg_model_agreement=agreement_score
                    )
                else:
                    existing = self.interactions_3way[key]
                    existing.frequency += 1
                    existing.avg_model_agreement = (
                        (existing.avg_model_agreement * (existing.frequency - 1) + agreement_score) 
                        / existing.frequency
                    )
                    existing.confidence = min(existing.frequency / 15, 1.0)
    
    def _calc_model_agreement(self, history: List[Dict], models: List[str]) -> float:
        """
        Calculate how well models agree with each other.
        Uses pairwise correlation of their confidence scores.
        """
        if len(models) < 2:
            return 0.0
        
        scores = {model: [] for model in models}
        
        for record in history:
            preds = record.get('predictions', {})
            for model in models:
                if model in preds:
                    scores[model].append(float(preds[model]))
        
        # All models must have sufficient data
        if any(len(scores[m]) < len(history) * 0.5 for m in models):
            return 0.0
        
        # Calculate average pairwise correlation
        correlations = []
        for model1, model2 in combinations(models, 2):
            if scores[model1] and scores[model2]:
                corr = np.corrcoef(scores[model1], scores[model2])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return float(np.mean(correlations)) if correlations else 0.0
    
    def boost_prediction(self, 
                        base_confidence: float,
                        model_predictions: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """
        Apply interaction-based boosting to a base prediction.
        
        Returns:
            (boosted_confidence, boost_details)
        """
        boosted = base_confidence
        boost_details = {
            'base_confidence': base_confidence,
            'boosts_applied': [],
            'total_boost': 0.0,
            'interaction_count': 0
        }
        
        # Check 3-way interactions (higher priority)
        for models_tuple, interaction in self.interactions_3way.items():
            if self._check_interaction_active(models_tuple, model_predictions):
                boost_amount = interaction.prediction_boost * interaction.confidence
                boosted *= (1 + boost_amount)
                boost_details['boosts_applied'].append({
                    'type': '3-way',
                    'models': models_tuple,
                    'boost': boost_amount,
                    'strength': interaction.interaction_strength
                })
                boost_details['total_boost'] += boost_amount
                boost_details['interaction_count'] += 1
        
        # Check 2-way interactions
        for models_tuple, interaction in self.interactions_2way.items():
            if self._check_interaction_active(models_tuple, model_predictions):
                boost_amount = interaction.prediction_boost * interaction.confidence
                boosted *= (1 + boost_amount * 0.7)  # 70% of weight for 2-way
                boost_details['boosts_applied'].append({
                    'type': '2-way',
                    'models': models_tuple,
                    'boost': boost_amount * 0.7,
                    'strength': interaction.interaction_strength
                })
                boost_details['total_boost'] += boost_amount * 0.7
                boost_details['interaction_count'] += 1
        
        # Cap at 98% confidence
        boosted = min(boosted, 0.98)
        boost_details['final_confidence'] = boosted
        
        return boosted, boost_details
    
    def _check_interaction_active(self, 
                                 models_tuple: Tuple[str, ...],
                                 model_predictions: Dict[str, float]) -> bool:
        """
        Check if an interaction is "active" (all models in the interaction agree).
        Agreement = confidence within 15% of each other.
        """
        confidences = []
        
        for model_name in models_tuple:
            if model_name in model_predictions:
                confidences.append(float(model_predictions[model_name]))
        
        if len(confidences) != len(models_tuple):
            return False  # Missing data
        
        # Check if all confidences are within 15% range
        if len(confidences) < 2:
            return False
        
        conf_range = max(confidences) - min(confidences)
        return conf_range < 0.15  # Within 15% of each other
    
    def get_active_interactions(self, 
                               model_predictions: Dict[str, float]) -> Dict[str, List[Dict]]:
        """Get all currently active interactions for a prediction set"""
        active = {'2way': [], '3way': []}
        
        for models_tuple, interaction in self.interactions_2way.items():
            if self._check_interaction_active(models_tuple, model_predictions):
                active['2way'].append({
                    'models': models_tuple,
                    'strength': interaction.interaction_strength,
                    'boost': interaction.prediction_boost,
                    'frequency': interaction.frequency
                })
        
        for models_tuple, interaction in self.interactions_3way.items():
            if self._check_interaction_active(models_tuple, model_predictions):
                active['3way'].append({
                    'models': models_tuple,
                    'strength': interaction.interaction_strength,
                    'boost': interaction.prediction_boost,
                    'frequency': interaction.frequency
                })
        
        return active
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for logging/analysis"""
        return {
            'interactions_2way_count': len(self.interactions_2way),
            'interactions_3way_count': len(self.interactions_3way),
            'history_size': len(self.interaction_history),
            'top_2way': [
                {
                    'models': k,
                    'strength': v.interaction_strength,
                    'frequency': v.frequency
                }
                for k, v in sorted(
                    self.interactions_2way.items(),
                    key=lambda x: x[1].frequency,
                    reverse=True
                )[:5]
            ],
            'top_3way': [
                {
                    'models': k,
                    'strength': v.interaction_strength,
                    'frequency': v.frequency
                }
                for k, v in sorted(
                    self.interactions_3way.items(),
                    key=lambda x: x[1].frequency,
                    reverse=True
                )[:5]
            ]
        }
