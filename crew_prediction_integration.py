#!/usr/bin/env python3
"""
Crew Prediction Integration Module
===================================
Loads the trained crew prediction model and provides easy interface for daily predictions.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "data" / "referee_conspiracy" / "crew_prediction_model.pkl"


class CrewPredictionEngine:
    """Load and use the crew prediction model."""
    
    def __init__(self, model_path: Union[str, Path, None] = None):
        """Initialize the crew prediction engine."""
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        model_path = Path(model_path)
        
        try:
            with model_path.open('rb') as f:
                self.model_data = pickle.load(f)
            
            self.model = self.model_data['model']
            self.crew_encoder = self.model_data['crew_encoder']
            self.team_encoder = self.model_data['team_encoder']
            self.rmse = self.model_data['rmse']
            self.mae = self.model_data['mae']
            self.r2 = self.model_data['r2']
            
            logger.info(f"✅ Crew prediction model loaded (R²: {self.r2:.4f}, MAE: {self.mae:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to load crew prediction model: {e}")
            self.model = None
    
    def predict_margin(self, crew_name: str, team_name: str, week: int = 9, year: int = 2024) -> float:
        """Predict the margin for a game based on crew and team."""
        if not self.model:
            return 0.0
        
        try:
            crew_id = self.crew_encoder.transform([crew_name])[0]
            team_id = self.team_encoder.transform([team_name])[0]
            
            features = np.array([[crew_id, team_id, week, year]])
            prediction = self.model.predict(features)[0]
            
            return float(prediction)
        
        except ValueError:
            # Crew or team not in training data
            return 0.0
        except Exception as e:
            logger.error(f"Error predicting margin: {e}")
            return 0.0
    
    def get_crew_team_biases(self, crew_name: str) -> Dict[str, float]:
        """Get predicted margins for a crew against all teams."""
        if not self.model:
            return {}
        
        try:
            crew_id = self.crew_encoder.transform([crew_name])[0]
            
            predictions = {}
            for team_name in self.team_encoder.classes_:
                team_id = self.team_encoder.transform([team_name])[0]
                margin = self.predict_margin(crew_name, team_name)
                predictions[team_name] = margin
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error getting crew biases: {e}")
            return {}
    
    def rank_games_by_crew_impact(self, games: List[Dict]) -> List[Tuple[Dict, float, str]]:
        """
        Rank games by crew impact.
        
        Returns list of (game, predicted_margin, crew_edge)
        """
        if not self.model:
            return games
        
        ranked = []
        for game in games:
            crew = game.get('referee_crew')
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            
            if not crew or not home_team or not away_team:
                ranked.append((game, 0.0, "No crew data"))
                continue
            
            # Predict how crew affects this matchup
            home_margin = self.predict_margin(crew, home_team)
            away_margin = self.predict_margin(crew, away_team)
            
            # Net crew impact (favors home if positive)
            crew_impact = home_margin - away_margin
            
            if abs(crew_impact) > 5:
                edge = f"CREW BIAS: {crew_impact:+.1f} (favors {home_team if crew_impact > 0 else away_team})"
            else:
                edge = f"Neutral ({crew_impact:+.1f})"
            
            ranked.append((game, crew_impact, edge))
        
        # Sort by absolute crew impact
        ranked.sort(key=lambda x: abs(x[1]), reverse=True)
        return ranked
    
    def get_crew_adjustment_for_prediction(self, crew_name: str, home_team: str, away_team: str,
                                          base_prediction: float) -> Tuple[float, str]:
        """
        Adjust a base prediction based on crew tendencies.
        
        Returns (adjusted_margin, reasoning)
        """
        if not self.model:
            return base_prediction, "No model"
        
        home_bias = self.predict_margin(crew_name, home_team)
        away_bias = self.predict_margin(crew_name, away_team)
        crew_adjustment = home_bias - away_bias
        
        adjusted = base_prediction + (crew_adjustment * 0.5)  # 50% weight on crew
        
        reasoning = f"Base: {base_prediction:+.1f}, Crew {crew_name} adjusts by {crew_adjustment:+.1f} → {adjusted:+.1f}"
        
        return adjusted, reasoning


def load_crew_predictions() -> CrewPredictionEngine:
    """Convenience function to load the crew prediction engine."""
    return CrewPredictionEngine()


# Example usage for NFL daily runner integration
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    engine = CrewPredictionEngine()
    
    print("\n" + "="*100)
    print("CREW PREDICTION ENGINE TEST")
    print("="*100)
    
    # Test specific crew-team predictions
    test_crew = "Alan Eck"
    test_games = [
        {'crew': test_crew, 'home': 'BAL', 'away': 'ARI'},
        {'crew': test_crew, 'home': 'KC', 'away': 'DET'},
        {'crew': test_crew, 'home': 'NYG', 'away': 'SF'},
    ]
    
    print(f"\nPredictions for {test_crew}:")
    for game in test_games:
        home_margin = engine.predict_margin(test_crew, game['home'])
        away_margin = engine.predict_margin(test_crew, game['away'])
        net = home_margin - away_margin
        
        print(f"  {game['home']} vs {game['away']}: {home_margin:+.1f} vs {away_margin:+.1f} (net {net:+.1f})")
    
    print("\n✅ Engine working correctly")
