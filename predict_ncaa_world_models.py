#!/usr/bin/env python3
"""
NCAA Production Prediction System with World Models
====================================================
Combines 12-model consensus with InteractionWorldModel and Causal Discovery
for maximum prediction accuracy.

Confidence Boost Chain:
  R1 Base → Calibration → Interaction Boost → Causal Adjustment → Final
"""

import json
import pickle
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Direct imports to avoid __init__.py dependencies
sys.path.insert(0, str(Path(__file__).parent / 'college_football_system' / 'core'))
sys.path.insert(0, str(Path(__file__).parent / 'college_football_system' / 'causal_discovery'))
from interaction_world_model import InteractionWorldModel
from ncaa_causal_learner import NCAACousalLearner

print("🏈 NCAA WORLD MODELS PREDICTION SYSTEM")
print("=" * 80)
print()

# Configuration
MIN_CONFIDENCE = 0.60
MAX_CONFIDENCE = 0.98
MODELS_DIR = Path("models/ncaa_12_model")


class NCAAWorldModelPredictor:
    """Production NCAA predictor with world model boosting."""

    def __init__(self):
        self.interaction_model = InteractionWorldModel()
        self.causal_model = NCAACousalLearner()
        self.models = {}
        self.model_names = [
            'spread_ensemble', 'total_ensemble', 'moneyline_ensemble',
            'ncaa_rf', 'ncaa_gb',
            'spread_edges', 'total_edges', 'moneyline_edges',
            'Market Consensus', 'Contrarian Model',
            'Referee Model', 'Injury Model'
        ]

        print("✅ World Models initialized")
        print(f"   - InteractionWorldModel: {self.interaction_model.to_dict()['interactions_2way_count']} 2-way interactions")
        print(f"   - InteractionWorldModel: {self.interaction_model.to_dict()['interactions_3way_count']} 3-way interactions")
        print(f"   - CausalDiscovery: {self.causal_model.to_dict()['causal_edges']} causal edges")
        print()

    def predict_game(
        self,
        game_data: Dict[str, Any],
        model_predictions: Dict[str, float],
        base_confidence: float
    ) -> Dict[str, Any]:
        """
        Full prediction with world model boosting.

        Args:
            game_data: Game context (teams, weather, injuries, etc.)
            model_predictions: Raw predictions from all 12 models {model_name: confidence}
            base_confidence: R1/meta-learner base confidence

        Returns:
            Complete prediction with boost details
        """
        game_id = game_data.get('game_id', 'unknown')

        print(f"🎯 Analyzing: {game_id}")
        print(f"   Base confidence: {base_confidence:.1%}")

        # Step 1: Calibration boost (for high confidence)
        calibrated = self._apply_calibration(base_confidence)
        calibration_boost = calibrated - base_confidence
        print(f"   After calibration: {calibrated:.1%} (+{calibration_boost:.1%})")

        # Step 2: Interaction boost
        interaction_conf, interaction_details = self.interaction_model.boost_prediction(
            calibrated,
            model_predictions
        )
        interaction_boost = interaction_conf - calibrated
        print(f"   After interaction: {interaction_conf:.1%} (+{interaction_boost:.1%})")
        print(f"      → {interaction_details['interaction_count']} interactions active")

        # Step 3: Causal adjustment
        causal_context = self._extract_causal_features(game_data)
        final_conf, causal_details = self.causal_model.apply_causal_adjustments(
            interaction_conf,
            causal_context
        )
        causal_boost = final_conf - interaction_conf
        print(f"   Final confidence: {final_conf:.1%} (+{causal_boost:.1%})")

        # Total boost
        total_boost = final_conf - base_confidence
        print(f"   ⚡ Total boost: +{total_boost:.1%}")
        print()

        # Record for learning
        self.interaction_model.record_prediction_batch(
            game_id=game_id,
            model_predictions=model_predictions,
            actual_result=None  # Will be updated when result is known
        )

        # Build result
        return {
            'game_id': game_id,
            'game_data': game_data,
            'timestamp': datetime.now().isoformat(),

            # Confidence chain
            'r1_base_confidence': base_confidence,
            'calibrated_confidence': calibrated,
            'interaction_confidence': interaction_conf,
            'final_confidence': final_conf,

            # Boosts
            'calibration_boost': calibration_boost,
            'interaction_boost': interaction_boost,
            'causal_boost': causal_boost,
            'total_boost': total_boost,

            # Details
            'interaction_details': interaction_details,
            'causal_details': causal_details,
            'active_interactions': self.interaction_model.get_active_interactions(model_predictions),
            'model_predictions': model_predictions,

            # Recommendation
            'recommendation': self._make_recommendation(final_conf, interaction_details, causal_details)
        }

    def _apply_calibration(self, confidence: float) -> float:
        """Apply calibration adjustment to base confidence."""
        if confidence > 0.80:
            return min(confidence * 1.03, MAX_CONFIDENCE)  # 3% boost for very high confidence
        elif confidence > 0.75:
            return min(confidence * 1.05, MAX_CONFIDENCE)  # 5% boost for high confidence
        else:
            return confidence

    def _extract_causal_features(self, game_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract and standardize causal variables from game data.

        Causal variables (standardized to mean 0, std 1):
        - temperature: Weather impact
        - wind_speed: Wind conditions
        - key_injury: Injury severity
        - rest_days: Rest differential
        - referee_penalty_rate: Expected penalty rate
        """
        # Standardize features (zero mean, unit variance)
        def standardize(value, mean, std):
            return (value - mean) / std if std > 0 else 0

        return {
            'temperature': standardize(
                game_data.get('temperature', 65),
                mean=65,
                std=15
            ),
            'wind_speed': standardize(
                game_data.get('wind_mph', 5),
                mean=5,
                std=10
            ),
            'key_injury': game_data.get('injury_impact', 0),  # Already 0-1
            'rest_days': standardize(
                game_data.get('days_rest', 7),
                mean=7,
                std=3
            ),
            'referee_penalty_rate': game_data.get('ref_penalty_rate', 0.5),  # Already 0-1
        }

    def _make_recommendation(
        self,
        confidence: float,
        interaction_details: Dict,
        causal_details: Dict
    ) -> Dict[str, Any]:
        """Generate betting recommendation based on confidence and signals."""

        # Determine bet strength
        if confidence >= 0.78:
            strength = "STRONG BET"
            stake_pct = 0.05  # 5% of bankroll
        elif confidence >= 0.75:
            strength = "GOOD BET"
            stake_pct = 0.03
        elif confidence >= 0.70:
            strength = "MODERATE BET"
            stake_pct = 0.02
        else:
            strength = "PASS"
            stake_pct = 0.0

        # Gather signals
        signals = []

        if interaction_details['interaction_count'] >= 3:
            signals.append("STRONG MODEL ALIGNMENT")
        elif interaction_details['interaction_count'] >= 1:
            signals.append("Model agreement detected")

        if causal_details.get('total_adjustment', 0) > 0.05:
            signals.append("SIGNIFICANT CAUSAL FACTORS")
        elif causal_details.get('total_adjustment', 0) > 0.02:
            signals.append("Causal factors present")

        return {
            'action': strength,
            'confidence': confidence,
            'stake_percentage': stake_pct,
            'signals': signals,
            'interaction_count': interaction_details['interaction_count'],
            'causal_adjustment': causal_details.get('total_adjustment', 0)
        }

    def update_result(self, game_id: str, actual_result: float):
        """
        Update prediction with actual result for learning.

        Args:
            game_id: Game identifier
            actual_result: 1 for win, 0 for loss
        """
        # This triggers learning in InteractionWorldModel
        self.interaction_model.record_prediction_batch(
            game_id=game_id,
            model_predictions={},  # Not needed for result updates
            actual_result=actual_result
        )
        print(f"✅ Updated result for {game_id}: {actual_result}")


def demo_prediction():
    """Demo prediction with world models."""

    print("📊 DEMO: World Models Prediction")
    print("=" * 80)
    print()

    predictor = NCAAWorldModelPredictor()

    # Example game data
    game_data = {
        'game_id': 'OKST_vs_TCU_2024_W12',
        'home_team': 'Oklahoma State',
        'away_team': 'TCU',
        'spread': -7.5,
        'total': 58.5,
        'temperature': 45,  # Cold
        'wind_mph': 15,  # Windy
        'injury_impact': 0.3,  # Some injuries
        'days_rest': 7,
        'ref_penalty_rate': 0.65  # High penalty rate expected
    }

    # Example 12-model predictions (normally from your models)
    model_predictions = {
        'spread_ensemble': 0.72,
        'total_ensemble': 0.68,
        'moneyline_ensemble': 0.75,
        'ncaa_rf': 0.70,
        'ncaa_gb': 0.71,
        'spread_edges': 0.73,
        'total_edges': 0.67,
        'moneyline_edges': 0.74,
        'Market Consensus': 0.69,
        'Contrarian Model': 0.76,
        'Referee Model': 0.65,
        'Injury Model': 0.68
    }

    # Base confidence (from R1/meta-learner)
    base_confidence = 0.70

    # Get prediction
    result = predictor.predict_game(game_data, model_predictions, base_confidence)

    # Display recommendation
    rec = result['recommendation']
    print("🎯 RECOMMENDATION")
    print("=" * 80)
    print(f"Action: {rec['action']}")
    print(f"Final Confidence: {rec['confidence']:.1%}")
    print(f"Stake: {rec['stake_percentage']:.1%} of bankroll")
    print(f"\nSignals:")
    for signal in rec['signals']:
        print(f"  ✓ {signal}")
    print()

    # Show boost breakdown
    print("⚡ BOOST BREAKDOWN")
    print("=" * 80)
    print(f"R1 Base:        {result['r1_base_confidence']:.1%}")
    print(f"+ Calibration:  {result['calibration_boost']:+.1%}")
    print(f"+ Interaction:  {result['interaction_boost']:+.1%}")
    print(f"+ Causal:       {result['causal_boost']:+.1%}")
    print(f"{'='*30}")
    print(f"Final:          {result['final_confidence']:.1%} ({result['total_boost']:+.1%} total)")
    print()

    # Active interactions
    active = result['active_interactions']
    if active['3way']:
        print(f"🔗 3-Way Interactions ({len(active['3way'])})")
        for interaction in active['3way']:
            print(f"   {interaction['models']}")
            print(f"      Strength: {interaction['strength']:.2f}, Boost: {interaction['boost']:.3f}")

    if active['2way']:
        print(f"\n🔗 2-Way Interactions ({len(active['2way'])})")
        for interaction in active['2way'][:3]:  # Top 3
            print(f"   {interaction['models']}")
            print(f"      Strength: {interaction['strength']:.2f}, Boost: {interaction['boost']:.3f}")

    return result


if __name__ == "__main__":
    # Run demo
    result = demo_prediction()

    # Save demo result
    output_file = Path("data/predictions/world_model_demo.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✅ Demo complete! Result saved to {output_file}")
