#!/usr/bin/env python3
"""
Calibrated NCAA Prediction System with Logging
- Applies confidence calibration formula
- Logs all predictions for tracking
- Filters out low-confidence bets (50-60%)
- Uses trained models with proper adjustments
"""

import json
import os
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests

print("🏈 CALIBRATED NCAA PREDICTION SYSTEM")
print("=" * 80)

# Configuration
MIN_CONFIDENCE = 0.60  # Skip anything below 60%
MAX_CONFIDENCE = 0.75  # Cap at 75% after calibration

def calibrate_confidence(raw_confidence):
    """
    Apply calibration formula based on analysis:
    - 80%+ predictions: multiply by 0.85
    - 60-80%: multiply by 0.90
    - Below 60%: skip entirely (don't bet)
    """
    if raw_confidence < MIN_CONFIDENCE:
        return None  # Don't bet on these
    
    if raw_confidence > 0.80:
        calibrated = raw_confidence * 0.85
    elif raw_confidence > 0.60:
        calibrated = raw_confidence * 0.90
    else:
        return None
    
    # Cap at 75%
    return min(calibrated, MAX_CONFIDENCE)

def log_prediction(prediction):
    """Log prediction to JSON file for tracking."""
    log_file = Path("data/predictions/prediction_log.json")
    
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
print("\n🤖 Loading trained models...")
models_dir = Path("models")

with open(models_dir / "ncaa_rf_model.pkl", "rb") as f:
    rf = pickle.load(f)
with open(models_dir / "ncaa_gb_model.pkl", "rb") as f:
    gb = pickle.load(f)

print("✅ Models loaded")

# Get this weekend's games
print("\n" + "=" * 80)
print("🏈 FETCHING THIS WEEKEND'S GAMES")
print("=" * 80)

api_key = os.getenv("ODDS_API_KEY", "e84d496405014d166f5dce95094ea024")
url = f"https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds/?apiKey={api_key}&regions=us&markets=spreads,h2h"

print("\n📊 Fetching from Odds API...")
resp = requests.get(url, timeout=10)
games = resp.json()

# Filter for this weekend
now = datetime.utcnow()
weekend_games = []

for g in games:
    game_time = datetime.fromisoformat(g['commence_time'].replace('Z', '+00:00'))
    hours_away = (game_time - now.replace(tzinfo=game_time.tzinfo)).total_seconds() / 3600
    
    if 0 <= hours_away <= 72:
        weekend_games.append((g, hours_away))

weekend_games.sort(key=lambda x: x[1])

print(f"✅ Found {len(weekend_games)} games\n")

if not weekend_games:
    print("⚠️  No games in next 72 hours")
    exit(0)

# Make predictions
print("=" * 80)
print("🎯 CALIBRATED PREDICTIONS")
print("=" * 80)
print()

predictions = []
filtered_count = 0

for g, hours in weekend_games:
    # Prepare features
    pred_features = {
        'home_conference_game': 0,
        'neutral_site': 0,
        'season_type': 0,
        'week': 11,
        'home_team_encoded': hash(g['home_team']) % 1000,
        'away_team_encoded': hash(g['away_team']) % 1000,
    }
    
    pred_df = pd.DataFrame([pred_features])
    
    # Get raw predictions
    rf_prob = rf.predict_proba(pred_df)[0, 1]
    gb_prob = gb.predict_proba(pred_df)[0, 1]
    raw_prob = (rf_prob + gb_prob) / 2
    raw_confidence = max(raw_prob, 1 - raw_prob)
    
    # Apply calibration
    calibrated_confidence = calibrate_confidence(raw_confidence)
    
    if calibrated_confidence is None:
        filtered_count += 1
        continue  # Skip low confidence bets
    
    # Get odds info
    spread = None
    home_ml = None
    away_ml = None
    
    for book in g.get('bookmakers', [])[:1]:
        for market in book.get('markets', []):
            if market['key'] == 'spreads':
                for outcome in market['outcomes']:
                    if outcome['name'] == g['home_team']:
                        spread = outcome['point']
            elif market['key'] == 'h2h':
                for outcome in market['outcomes']:
                    if outcome['name'] == g['home_team']:
                        home_ml = outcome['price']
                    else:
                        away_ml = outcome['price']
    
    pick = g['home_team'] if raw_prob > 0.5 else g['away_team']
    pick_type = "HOME" if raw_prob > 0.5 else "AWAY"
    
    prediction = {
        'game': f"{g['away_team']} @ {g['home_team']}",
        'predicted_winner': pick,
        'pick_type': pick_type,
        'raw_confidence': float(raw_confidence),
        'calibrated_confidence': float(calibrated_confidence),
        'confidence_adjustment': float(calibrated_confidence - raw_confidence),
        'spread': spread,
        'home_ml': home_ml,
        'away_ml': away_ml,
        'hours_until_game': float(hours),
        'edge': float(abs(raw_prob - 0.5)),
        'actual_result': None  # Fill in after game completes
    }
    
    predictions.append(prediction)
    
    # Log prediction
    log_prediction(prediction.copy())

# Sort by calibrated confidence
predictions.sort(key=lambda x: x['calibrated_confidence'], reverse=True)

print(f"⚠️  Filtered out {filtered_count} low-confidence games (below 60%)")
print(f"✅ {len(predictions)} HIGH-QUALITY predictions\n")

print("=" * 80)
print("🎯 TOP PICKS (After Calibration)")
print("=" * 80)
print()

for i, pred in enumerate(predictions[:10], 1):
    raw = pred['raw_confidence'] * 100
    calibrated = pred['calibrated_confidence'] * 100
    adjustment = pred['confidence_adjustment'] * 100
    
    print(f"{i:2d}. {pred['game']}")
    print(f"    Pick: {pred['predicted_winner']} ({pred['pick_type']})")
    print(f"    Raw Confidence: {raw:.1f}% → Calibrated: {calibrated:.1f}% ({adjustment:+.1f}%)")
    print(f"    Edge: {pred['edge']*100:.1f}% | Spread: {pred['spread']}")
    
    # Kelly sizing recommendation
    if calibrated >= 70:
        print(f"    💰 BET SIZE: 2-3 units (high confidence)")
    elif calibrated >= 65:
        print(f"    💰 BET SIZE: 1.5-2 units (medium confidence)")
    else:
        print(f"    💰 BET SIZE: 1 unit (lower confidence)")
    
    print(f"    ⏰ Starts in: {pred['hours_until_game']:.1f} hours")
    print()

print("=" * 80)
print("📊 BETTING SUMMARY")
print("=" * 80)
print(f"Total Games Analyzed: {len(weekend_games)}")
print(f"Filtered Out (< 60% conf): {filtered_count}")
print(f"Recommended Bets: {len(predictions)}")
print(f"High Confidence (70%+): {len([p for p in predictions if p['calibrated_confidence'] >= 0.70])}")
print(f"Medium Confidence (65-70%): {len([p for p in predictions if 0.65 <= p['calibrated_confidence'] < 0.70])}")
print(f"Lower Confidence (60-65%): {len([p for p in predictions if 0.60 <= p['calibrated_confidence'] < 0.65])}")
print()

print("=" * 80)
print("⚠️  IMPORTANT REMINDERS")
print("=" * 80)
print("1. ✅ All predictions logged to: data/predictions/prediction_log.json")
print("2. ✅ Confidence scores calibrated (reduced by 10-15%)")
print("3. ✅ Low confidence bets (< 60%) filtered out")
print("4. ⚠️  Use Kelly Criterion with 0.25 fraction (quarter-Kelly)")
print("5. ⚠️  Never bet more than 3% of bankroll on single game")
print("6. 📝 Update actual_result in log after games complete")
print()

print("🔧 TO UPDATE RESULTS AFTER GAMES:")
print("-" * 80)
print("1. Edit: data/predictions/prediction_log.json")
print("2. Change 'actual_result': null → 'WIN' or 'LOSS'")
print("3. Run: python analyze_confidence_calibration.py")
print("4. Adjust calibration formula monthly based on results")
print()
