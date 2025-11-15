#!/usr/bin/env python3
"""
NCAA World Models Prediction Script
====================================
Standalone script to run NCAA predictions with world models.
Can be used directly or imported as a module.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from college_football_system.core.interaction_world_model import InteractionWorldModel
from college_football_system.causal_discovery.ncaa_causal_learner import NCAACousalLearner


class NCAAWorldModelPredictor:
    """Production predictor with world models."""
    
    def __init__(self):
        self.interaction_model = InteractionWorldModel()
        self.causal_model = NCAACousalLearner()
    
    def predict_game(self, game_id, game_info, model_predictions, base_confidence):
        """
        Predict a game with world model boosting.
        
        Args:
            game_id: Unique game identifier
            game_info: Dict with game details (teams, venue, etc)
            model_predictions: Dict of model_name -> confidence (0-1)
            base_confidence: Base prediction confidence (0-1)
        
        Returns:
            Prediction record with all boosts applied
        """
        # Apply interaction boost
        interaction_boosted, interaction_details = self.interaction_model.boost_prediction(
            base_confidence,
            model_predictions
        )
        
        # Apply causal adjustment
        causal_context = {
            'weather_impact': game_info.get('weather_impact', 0.0),
            'injury_impact': game_info.get('injury_count', 0) * 0.1,
            'trap_score': game_info.get('trap_score', 0) / 100
        }
        causal_adjusted, causal_details = self.causal_model.apply_causal_adjustments(
            interaction_boosted,
            causal_context
        )
        
        # Build prediction record
        record = {
            'timestamp': datetime.now().isoformat(),
            'game_id': game_id,
            'game_info': game_info,
            'base_confidence': base_confidence,
            'confidence_breakdown': {
                'base': int(base_confidence * 100),
                'after_interactions': int(interaction_boosted * 100),
                'final': int(causal_adjusted * 100)
            },
            'interaction_boost': {
                'applied': interaction_details.get('interaction_count', 0) > 0,
                'count': interaction_details.get('interaction_count', 0),
                'boost_amount': interaction_boosted - base_confidence
            },
            'causal_adjustment': {
                'applied': len(causal_details.get('adjustments', [])) > 0,
                'count': len(causal_details.get('adjustments', [])),
                'adjustment_amount': causal_adjusted - interaction_boosted
            },
            'final_recommendation': 'BET' if causal_adjusted > 0.65 else 'PASS'
        }
        
        return record
    
    def record_result(self, game_id, model_predictions, actual_result):
        """Record game result for model learning."""
        self.interaction_model.record_prediction_batch(
            game_id=game_id,
            model_predictions=model_predictions,
            actual_result=actual_result
        )


def run_demo():
    """Run a demo prediction."""
    print("\n" + "="*70)
    print("🏈 NCAA WORLD MODELS PREDICTION SYSTEM")
    print("="*70 + "\n")
    
    predictor = NCAAWorldModelPredictor()
    
    # Sample game
    game = {
        'game_id': 'clemson_louisville_20241115',
        'teams': 'Clemson @ Louisville',
        'weather_impact': 0.3,
        'injury_count': 2,
        'trap_score': 25
    }
    
    # Sample model predictions
    models = {
        'spread_ensemble': 0.75,
        'total_ensemble': 0.72,
        'moneyline_ensemble': 0.73,
        'ncaa_rf': 0.70,
        'ncaa_gb': 0.71,
        'market_consensus': 0.68
    }
    
    # Get prediction
    result = predictor.predict_game(
        game_id=game['game_id'],
        game_info=game,
        model_predictions=models,
        base_confidence=0.70
    )
    
    # Display results
    print(f"Game: {game['teams']}")
    print(f"\nConfidence Chain:")
    print(f"  Base: {result['confidence_breakdown']['base']}%")
    print(f"  → After Interactions: {result['confidence_breakdown']['after_interactions']}%")
    print(f"  → Final: {result['confidence_breakdown']['final']}%")
    
    if result['interaction_boost']['applied']:
        print(f"\n✨ Interactions Detected: {result['interaction_boost']['count']} active")
        print(f"   Boost: +{int(result['interaction_boost']['boost_amount']*100)}%")
    
    if result['causal_adjustment']['applied']:
        print(f"\n📊 Causal Adjustments: {result['causal_adjustment']['count']} factors")
        print(f"   Adjustment: +{int(result['causal_adjustment']['adjustment_amount']*100)}%")
    
    print(f"\n🎯 Recommendation: {result['final_recommendation']}")
    
    # Save to file
    output_file = Path('data/predictions/world_model_demo.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Result saved to {output_file}\n")


if __name__ == '__main__':
    run_demo()
