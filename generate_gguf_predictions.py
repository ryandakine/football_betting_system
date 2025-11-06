#!/usr/bin/env python3
"""
Generate GGUF Predictions (Run Once)
====================================
Runs the 5 Hugging Face GGUF models ONCE to generate predictions
for all games, then saves them to be used in backtests.

This avoids loading 20GB of models repeatedly.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from practical_gguf_ensemble import PracticalGGUFEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTIONS_DIR = Path("data/gguf_predictions")
PREDICTIONS_DIR.mkdir(exist_ok=True)


def generate_predictions_for_season(season: int) -> Dict[str, Dict]:
    """Generate GGUF predictions for all games in a season"""
    logger.info(f"🤖 Generating GGUF predictions for {season} season...")
    
    # Load historical games
    hist_file = Path(f"data/football/historical/nfl_{season}_{season}_games.json")
    if not hist_file.exists():
        logger.error(f"Historical data not found: {hist_file}")
        return {}
    
    with open(hist_file) as f:
        data = json.load(f)
    
    games = data.get('games', [])
    logger.info(f"Found {len(games)} games")
    
    # Initialize GGUF ensemble
    try:
        ensemble = PracticalGGUFEnsemble()
    except Exception as e:
        logger.error(f"Failed to initialize GGUF ensemble: {e}")
        return {}
    
    # Generate predictions
    predictions = {}
    
    for game in tqdm(games, desc=f"Processing {season}"):
        game_id = game.get('game_id')
        if not game_id:
            continue
        
        try:
            # Get ensemble prediction
            pred = ensemble.get_ensemble_prediction(game, num_models=3)
            if pred:
                predictions[game_id] = {
                    'probability': pred.get('probability', 0.5),
                    'confidence': pred.get('confidence', 0.5),
                    'models_used': pred.get('models_used', []),
                    'analysis': pred.get('analysis', ''),
                }
        except Exception as e:
            logger.warning(f"Failed to predict {game_id}: {e}")
            predictions[game_id] = {
                'probability': 0.5,
                'confidence': 0.5,
                'error': str(e)
            }
    
    # Save predictions
    output_file = PREDICTIONS_DIR / f"gguf_predictions_{season}.json"
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"✅ Saved {len(predictions)} predictions to {output_file}")
    
    return predictions


def main():
    seasons = [2020, 2021, 2022, 2023]
    
    for season in seasons:
        try:
            generate_predictions_for_season(season)
        except Exception as e:
            logger.error(f"Failed to process {season}: {e}")
    
    logger.info("🎉 All predictions generated!")


if __name__ == '__main__':
    main()
