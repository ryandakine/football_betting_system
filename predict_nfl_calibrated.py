#!/usr/bin/env python3
"""
Calibrated NFL Prediction System with Logging
- Applies confidence calibration formula
- Logs all predictions for tracking
- Filters out low-confidence bets (50-60%)
- Uses trained ensemble models
"""

import json
import os
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests

print("🏈 CALIBRATED NFL PREDICTION SYSTEM")
print("=" * 80)

# SimpleEnsemble class (needed for unpickling models)
class SimpleEnsemble:
    """Simple ensemble that averages predictions from multiple models."""
    def __init__(self, models):
        self.models = models
    
    def predict_proba(self, X):
        """Average probabilities from all models"""
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(probas, axis=0)
    
    def predict(self, X):
        """Predict using averaged probabilities"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

# Configuration
MIN_CONFIDENCE = 0.60  # Skip anything below 60%
MAX_CONFIDENCE = 0.85  # Cap at 85% after calibration

def calibrate_confidence(raw_confidence):
    """
    Apply calibration formula based on analysis.
    Current data shows underconfidence (+31% gap), so we boost confidence.
    
    Calibration strategy:
    - 80%+ predictions: multiply by 1.15 (boost high confidence)
    - 60-80%: multiply by 1.10 (modest boost)
    - Below 60%: skip entirely (don't bet)
    """
    if raw_confidence < MIN_CONFIDENCE:
        return None  # Don't bet on these
    
    if raw_confidence > 0.80:
        calibrated = raw_confidence * 1.15
    elif raw_confidence > 0.60:
        calibrated = raw_confidence * 1.10
    else:
        return None
    
    # Cap at 85%
    return min(calibrated, MAX_CONFIDENCE)

def log_prediction(prediction):
    """Log prediction to JSON file for tracking."""
    log_file = Path("data/predictions/nfl_prediction_log.json")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing logs
    if log_file.exists():
        with open(log_file) as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Add timestamp
    prediction['timestamp'] = datetime.now().isoformat()
    
    # Append new prediction
    logs.append(prediction)
    
    # Save
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"📝 Logged to: {log_file}")

# Load trained models
print("\n🤖 Loading trained ensemble models...")
models_dir = Path("models")

try:
    with open(models_dir / "spread_ensemble.pkl", "rb") as f:
        spread_model = pickle.load(f)
    with open(models_dir / "total_ensemble.pkl", "rb") as f:
        total_model = pickle.load(f)
    with open(models_dir / "moneyline_ensemble.pkl", "rb") as f:
        ml_model = pickle.load(f)
    print("✅ Models loaded (spread, total, moneyline)")
except FileNotFoundError as e:
    print(f"❌ Model files not found: {e}")
    print("   Run: python train_nfl_models.py")
    exit(1)

# Get this week's games
print("\n" + "=" * 80)
print("🏈 FETCHING THIS WEEK'S NFL GAMES")
print("=" * 80)

api_key = os.getenv("ODDS_API_KEY", "e84d496405014d166f5dce95094ea024")
url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={api_key}&regions=us&markets=spreads,totals,h2h"

print("\n📊 Fetching from Odds API...")
try:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    games = resp.json()
except Exception as e:
    print(f"❌ Failed to fetch odds: {e}")
    exit(1)

# Filter for upcoming games
now = datetime.utcnow()
upcoming_games = []

for g in games:
    game_time = datetime.fromisoformat(g['commence_time'].replace('Z', '+00:00'))
    hours_away = (game_time - now.replace(tzinfo=game_time.tzinfo)).total_seconds() / 3600
    
    if 0 <= hours_away <= 168:  # Next 7 days
        upcoming_games.append((g, hours_away))

upcoming_games.sort(key=lambda x: x[1])

print(f"✅ Found {len(upcoming_games)} upcoming games\n")

if not upcoming_games:
    print("⚠️  No games in next 7 days")
    exit(0)

# Make predictions
print("=" * 80)
print("🎯 CALIBRATED PREDICTIONS")
print("=" * 80)
print()

predictions = []
filtered_count = 0

for g, hours in upcoming_games:
    # Extract odds
    spread = None
    total = None
    home_ml = None
    away_ml = None
    
    for book in g.get('bookmakers', [])[:1]:
        for market in book.get('markets', []):
            if market['key'] == 'spreads':
                for outcome in market['outcomes']:
                    if outcome['name'] == g['home_team']:
                        spread = outcome['point']
            elif market['key'] == 'totals':
                for outcome in market['outcomes']:
                    if outcome['name'] == 'Over':
                        total = outcome['point']
            elif market['key'] == 'h2h':
                for outcome in market['outcomes']:
                    if outcome['name'] == g['home_team']:
                        home_ml = outcome['price']
                    else:
                        away_ml = outcome['price']
    
    if spread is None or total is None:
        continue
    
    # Prepare features matching training data
    game_features = {
        'spread': float(spread),
        'total': float(total),
        'home_ml_odds': float(home_ml or -110),
        'away_ml_odds': float(away_ml or -110),
        'is_division_game': 0.0,  # Don't know this from API
        'is_primetime': 0.0,
        'is_dome': 0.0,
        'week': 10.0,  # Current week estimate
        'attendance': 0.67,  # Average normalized
        'away_travel_distance': 0.25,  # Average normalized
        'home_rest_days': 7.0,
        'away_rest_days': 7.0,
        'is_playoff': 0.0,
    }
    
    pred_df = pd.DataFrame([game_features])
    
    # Get predictions from all three models
    spread_prob = spread_model.predict_proba(pred_df)[0, 1]  # Home covers
    total_prob = total_model.predict_proba(pred_df)[0, 1]    # Over
    ml_prob = ml_model.predict_proba(pred_df)[0, 1]          # Home wins
    
    # Compute confidence (distance from 50%)
    spread_confidence = max(spread_prob, 1 - spread_prob)
    total_confidence = max(total_prob, 1 - total_prob)
    ml_confidence = max(ml_prob, 1 - ml_prob)
    
    # Apply calibration
    spread_calibrated = calibrate_confidence(spread_confidence)
    total_calibrated = calibrate_confidence(total_confidence)
    ml_calibrated = calibrate_confidence(ml_confidence)
    
    # Filter: at least one market must have calibrated confidence
    if not any([spread_calibrated, total_calibrated, ml_calibrated]):
        filtered_count += 1
        continue
    
    # Determine picks
    spread_pick = g['home_team'] if spread_prob > 0.5 else g['away_team']
    spread_pick_type = "HOME" if spread_prob > 0.5 else "AWAY"
    total_pick = "OVER" if total_prob > 0.5 else "UNDER"
    ml_pick = g['home_team'] if ml_prob > 0.5 else g['away_team']
    
    prediction = {
        'game': f"{g['away_team']} @ {g['home_team']}",
        'home_team': g['home_team'],
        'away_team': g['away_team'],
        'hours_until_game': float(hours),
        
        # Spread prediction
        'spread_line': spread,
        'spread_pick': spread_pick,
        'spread_pick_type': spread_pick_type,
        'spread_raw_confidence': float(spread_confidence),
        'spread_calibrated_confidence': float(spread_calibrated) if spread_calibrated else None,
        
        # Total prediction
        'total_line': total,
        'total_pick': total_pick,
        'total_raw_confidence': float(total_confidence),
        'total_calibrated_confidence': float(total_calibrated) if total_calibrated else None,
        
        # Moneyline prediction
        'ml_pick': ml_pick,
        'ml_raw_confidence': float(ml_confidence),
        'ml_calibrated_confidence': float(ml_calibrated) if ml_calibrated else None,
        'home_ml_odds': home_ml,
        'away_ml_odds': away_ml,
        
        # Metadata
        'actual_result': None  # Fill in after game completes
    }
    
    predictions.append(prediction)
    
    # Log prediction
    log_prediction(prediction.copy())

# Sort by average calibrated confidence
def avg_confidence(pred):
    confs = [
        pred['spread_calibrated_confidence'],
        pred['total_calibrated_confidence'],
        pred['ml_calibrated_confidence']
    ]
    valid = [c for c in confs if c is not None]
    return sum(valid) / len(valid) if valid else 0

predictions.sort(key=avg_confidence, reverse=True)

print(f"⚠️  Filtered out {filtered_count} low-confidence games (below 60%)")
print(f"✅ {len(predictions)} HIGH-QUALITY predictions\n")

print("=" * 80)
print("🎯 TOP PICKS (After Calibration)")
print("=" * 80)
print()

for i, pred in enumerate(predictions[:10], 1):
    print(f"\n#{i}: {pred['game']}")
    print(f"   🕒 In {pred['hours_until_game']:.1f} hours")
    
    if pred['spread_calibrated_confidence']:
        raw = pred['spread_raw_confidence'] * 100
        cal = pred['spread_calibrated_confidence'] * 100
        adj = cal - raw
        print(f"   SPREAD: {pred['spread_pick']} ({pred['spread_line']:+.1f})")
        print(f"           {raw:.1f}% → {cal:.1f}% ({adj:+.1f}%)")
    
    if pred['total_calibrated_confidence']:
        raw = pred['total_raw_confidence'] * 100
        cal = pred['total_calibrated_confidence'] * 100
        adj = cal - raw
        print(f"   TOTAL:  {pred['total_pick']} {pred['total_line']}")
        print(f"           {raw:.1f}% → {cal:.1f}% ({adj:+.1f}%)")
    
    if pred['ml_calibrated_confidence']:
        raw = pred['ml_raw_confidence'] * 100
        cal = pred['ml_calibrated_confidence'] * 100
        adj = cal - raw
        odds = pred['home_ml_odds'] if pred['ml_pick'] == pred['home_team'] else pred['away_ml_odds']
        print(f"   ML:     {pred['ml_pick']} ({odds:+.0f})")
        print(f"           {raw:.1f}% → {cal:.1f}% ({adj:+.1f}%)")

print("\n" + "=" * 80)
print(f"✅ All predictions logged to: data/predictions/nfl_prediction_log.json")
print("=" * 80)
