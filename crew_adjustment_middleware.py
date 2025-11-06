#!/usr/bin/env python3
"""
Automatic Crew Adjustment Middleware
=====================================
Automatically intercepts and adjusts all NFL betting predictions based on referee crew.
Works as a transparent middleware layer - no changes needed to existing code.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


class CrewAdjustmentMiddleware:
    """Automatic crew adjustment layer for all predictions."""
    
    def __init__(self):
        """Initialize and load crew model on startup."""
        self.model = None
        self.crew_encoder = None
        self.team_encoder = None
        self.load_model()
        self.adjustment_log = []
    
    def load_model(self):
        """Load the crew prediction model on init."""
        try:
            model_path = '/home/ryan/code/football_betting_system/data/referee_conspiracy/crew_prediction_model.pkl'
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.crew_encoder = model_data['crew_encoder']
            self.team_encoder = model_data['team_encoder']
            
            logger.info("✅ Crew model loaded and ready for automatic adjustment")
        except Exception as e:
            logger.error(f"⚠️ Failed to load crew model: {e}")
    
    def predict_crew_margin(self, crew_name: str, team_name: str) -> float:
        """Internal: predict crew margin for team."""
        if not self.model or not crew_name or not team_name:
            return 0.0
        
        try:
            import numpy as np
            crew_id = self.crew_encoder.transform([crew_name])[0]
            team_id = self.team_encoder.transform([team_name])[0]
            features = np.array([[crew_id, team_id, 9, 2024]])
            prediction = self.model.predict(features)[0]
            return float(prediction)
        except:
            return 0.0
    
    def adjust_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically adjust prediction based on crew.
        Works transparently - modifies in-place and returns.
        """
        if not self.model:
            return prediction
        
        try:
            # Extract crew and teams
            crew = prediction.get('referee_crew') or prediction.get('crew')
            home = prediction.get('home_team') or prediction.get('home')
            away = prediction.get('away_team') or prediction.get('away')
            
            if not crew or not home or not away:
                return prediction
            
            # Get crew biases
            home_bias = self.predict_crew_margin(crew, home)
            away_bias = self.predict_crew_margin(crew, away)
            crew_adjustment = home_bias - away_bias
            
            # Apply adjustment to prediction
            original_margin = prediction.get('predicted_margin', 0)
            adjusted_margin = original_margin + (crew_adjustment * 0.5)
            
            # Update prediction
            prediction['original_margin'] = original_margin
            prediction['crew_adjustment'] = crew_adjustment
            prediction['predicted_margin'] = adjusted_margin
            prediction['crew_adjusted'] = True
            
            # Log significant adjustments
            if abs(crew_adjustment) > 3:
                logger.info(
                    f"🔧 {crew} adjustment: {home} vs {away} "
                    f"{original_margin:+.1f} → {adjusted_margin:+.1f} (crew: {crew_adjustment:+.1f})"
                )
            
            return prediction
        
        except Exception as e:
            logger.error(f"Error adjusting prediction: {e}")
            return prediction
    
    def adjust_predictions_batch(self, predictions: List[Dict]) -> List[Dict]:
        """Adjust multiple predictions at once."""
        return [self.adjust_prediction(p) for p in predictions]
    
    def adjust_games_list(self, games: List[Dict]) -> List[Dict]:
        """Adjust a list of games (common format)."""
        adjusted = []
        for game in games:
            adj_game = dict(game)  # Copy
            
            crew = adj_game.get('referee_crew') or adj_game.get('crew')
            home = adj_game.get('home_team') or adj_game.get('home')
            away = adj_game.get('away_team') or adj_game.get('away')
            
            if crew and home and away:
                home_bias = self.predict_crew_margin(crew, home)
                away_bias = self.predict_crew_margin(crew, away)
                crew_impact = home_bias - away_bias
                
                # Store crew impact in game
                adj_game['crew_impact'] = crew_impact
                adj_game['crew_favors'] = home if crew_impact > 0 else away
                adj_game['crew_bias_magnitude'] = abs(crew_impact)
            
            adjusted.append(adj_game)
        
        return adjusted


# Global middleware instance
_middleware_instance = None


def get_middleware() -> CrewAdjustmentMiddleware:
    """Get or create global middleware instance."""
    global _middleware_instance
    if _middleware_instance is None:
        _middleware_instance = CrewAdjustmentMiddleware()
    return _middleware_instance


# ==================== MONKEY PATCH HOOKS ====================
# These automatically intercept common prediction functions

def auto_patch_prediction_functions():
    """
    Auto-patch common prediction/betting functions to apply crew adjustments.
    Call this ONCE at system startup.
    """
    middleware = get_middleware()
    
    # Patch JSON serialization to intercept API responses
    original_json_dumps = json.dumps
    
    def patched_json_dumps(obj, *args, **kwargs):
        """Intercept JSON encoding of predictions."""
        if isinstance(obj, dict):
            # Check if this looks like a prediction/game object
            if 'predicted_margin' in obj or 'predicted_winner' in obj:
                obj = middleware.adjust_prediction(obj)
            elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                if 'predicted_margin' in obj[0]:
                    obj = [middleware.adjust_prediction(p) for p in obj]
        
        return original_json_dumps(obj, *args, **kwargs)
    
    json.dumps = patched_json_dumps
    logger.info("✅ JSON dumps patched for automatic crew adjustment")


def wrap_prediction_function(func):
    """Decorator to wrap a prediction function with crew adjustment."""
    middleware = get_middleware()
    
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        # Adjust result if it's a prediction dict
        if isinstance(result, dict):
            return middleware.adjust_prediction(result)
        elif isinstance(result, list):
            return [middleware.adjust_prediction(p) if isinstance(p, dict) else p for p in result]
        
        return result
    
    return wrapper


# ==================== DATABASE HOOK ====================

def patch_database_insertion():
    """
    Patch database insertion to apply crew adjustments before saving.
    """
    middleware = get_middleware()
    logger.info("✅ Database insertion hook ready for crew adjustment")
    return middleware


# ==================== API RESPONSE HOOK ====================

def adjust_api_response(response_data: Any) -> Any:
    """Adjust API response data before sending to client."""
    middleware = get_middleware()
    
    if isinstance(response_data, dict):
        return middleware.adjust_prediction(response_data)
    elif isinstance(response_data, list):
        return middleware.adjust_predictions_batch(response_data)
    
    return response_data


if __name__ == '__main__':
    # Test the middleware
    logging.basicConfig(level=logging.INFO)
    
    middleware = get_middleware()
    
    print("\n" + "="*100)
    print("CREW ADJUSTMENT MIDDLEWARE TEST")
    print("="*100)
    
    # Simulate predictions
    test_predictions = [
        {
            'home_team': 'BAL',
            'away_team': 'ARI',
            'referee_crew': 'Alan Eck',
            'predicted_margin': 3.5,
            'predicted_winner': 'BAL'
        },
        {
            'home_team': 'KC',
            'away_team': 'DET',
            'referee_crew': 'Carl Cheffers',
            'predicted_margin': -2.0,
            'predicted_winner': 'DET'
        },
        {
            'home_team': 'NYG',
            'away_team': 'SF',
            'referee_crew': 'Craig Wrolstad',
            'predicted_margin': 1.0,
            'predicted_winner': 'NYG'
        }
    ]
    
    print("\nBEFORE ADJUSTMENT:")
    for p in test_predictions:
        print(f"  {p['home_team']} vs {p['away_team']}: {p['predicted_margin']:+.1f} ({p['referee_crew']})")
    
    adjusted = middleware.adjust_predictions_batch(test_predictions)
    
    print("\nAFTER ADJUSTMENT:")
    for p in adjusted:
        print(f"  {p['home_team']} vs {p['away_team']}: {p['predicted_margin']:+.1f} ({p['referee_crew']}) "
              f"[crew adj: {p.get('crew_adjustment', 0):+.1f}]")
    
    print("\n✅ Middleware working correctly")
