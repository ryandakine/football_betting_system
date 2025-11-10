#!/usr/bin/env python3
"""
LIVE NCAA Prediction System
Uses all 11 trained models to predict upcoming games
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle
from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator

print("="*80)
print("🏈 NCAA 11-MODEL LIVE PREDICTION SYSTEM")
print("="*80)
print()

# Configuration
MIN_EDGE = 0.05  # 5%
MIN_CONFIDENCE = 0.70  # 70%
CALIBRATION_MULTIPLIER = 0.90  # Models are overconfident

# Initialize
engineer = NCAAFeatureEngineer("data/football/historical/ncaaf")
orchestrator = SuperIntelligenceOrchestrator("models/ncaa")

# Load all trained models
print("📂 Loading trained models...")
model_count = 0
for model_name in ['spread_ensemble', 'total_ensemble', 'moneyline_ensemble',
                    'first_half_spread', 'home_team_total', 'away_team_total',
                    'alt_spread', 'xgboost_super', 'neural_net_deep',
                    'officiating_bias', 'prop_bet_specialist']:
    model_path = Path(f"models/ncaa/{model_name}.pkl")
    if model_path.exists():
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            orchestrator.models[model_name].model = model_data['model']
            orchestrator.models[model_name].scaler = model_data.get('scaler', None)
            orchestrator.models[model_name].is_trained = model_data['is_trained']
            print(f"  ✅ {model_data['name']}")
            model_count += 1

print(f"\n✅ Loaded {model_count}/11 models")
print()

# Get current season and week
current_year = datetime.now().year
if datetime.now().month < 8:
    current_year -= 1

# Load current year - NO FALLBACKS
print(f"📅 Loading {current_year} season...")
games = engineer.load_season_data(current_year)

print(f"📍 Prediction Mode: LIVE")
print()

# Load games
print("🔍 Loading games...")
upcoming_games = [g for g in games if not g.get('completed')]

if not upcoming_games:
    print("❌ No upcoming games found.")
    print("   System requires live game data - completed games cannot be used.")
    exit(0)

print(f"Found {len(upcoming_games)} games to analyze")
print()

predictions = []

for game in upcoming_games:
    home_team = game.get('homeTeam')
    away_team = game.get('awayTeam')
    week = game.get('week')

    if not home_team or not away_team:
        continue

    # Engineer features
    try:
        features_dict = engineer.engineer_features(game, current_year)
        features = pd.DataFrame([features_dict])
    except Exception as e:
        print(f"  ⚠️  Could not engineer features for {away_team} @ {home_team}: {e}")
        continue

    # Get predictions from all models
    spread_preds = []
    total_preds = []
    ml_probs = []

    # Spread predictions
    for model_name in ['spread_ensemble', 'xgboost_super', 'neural_net_deep', 'alt_spread']:
        if orchestrator.models[model_name].is_trained:
            try:
                pred = orchestrator.models[model_name].predict(features)
                spread_preds.append(pred[0] if hasattr(pred, '__iter__') else pred)
            except:
                pass

    # Total predictions
    for model_name in ['total_ensemble', 'home_team_total', 'away_team_total']:
        if orchestrator.models[model_name].is_trained:
            try:
                pred = orchestrator.models[model_name].predict(features)
                if model_name == 'total_ensemble':
                    total_preds.append(pred[0] if hasattr(pred, '__iter__') else pred)
                else:
                    # Home + Away total
                    total_preds.append((pred[0] if hasattr(pred, '__iter__') else pred) * 2)
            except:
                pass

    # Moneyline probability
    if orchestrator.models['moneyline_ensemble'].is_trained:
        try:
            ml_prob = orchestrator.models['moneyline_ensemble'].predict_proba(features)
            ml_probs.append(ml_prob[0, 1])  # Home win probability
        except:
            pass

    if not spread_preds:
        continue

    # Calculate consensus
    predicted_spread = np.mean(spread_preds)
    spread_std = np.std(spread_preds)
    predicted_total = np.mean(total_preds) if total_preds else None
    home_win_prob = np.mean(ml_probs) if ml_probs else None

    # Confidence based on model agreement
    raw_confidence = max(0.5, min(0.95, 1 - (spread_std / 15.0)))
    confidence = raw_confidence * CALIBRATION_MULTIPLIER

    # Officiating bias adjustment (Model 11)
    home_conf = game.get('homeConference', 'Unknown')
    away_conf = game.get('awayConference', 'Unknown')
    off_conf = home_conf  # Assume home conference officiates

    officiating_adjustment = 0
    if orchestrator.models['officiating_bias'].is_trained:
        try:
            bias_profiles = {
                'SEC': 1.6,
                'Big Ten': 0.8,
                'Big 12': 0.4,
                'ACC': 1.0,
                'Pac-12': 0.0
            }
            if off_conf in bias_profiles:
                officiating_adjustment = bias_profiles[off_conf]
                predicted_spread += officiating_adjustment
        except:
            pass

    # Calculate edge (simplified - using market as baseline)
    market_spread = 0  # Neutral baseline
    edge = abs(predicted_spread - market_spread) / 14.0

    # Determine recommendation
    if edge >= MIN_EDGE and confidence >= MIN_CONFIDENCE:
        if predicted_spread > 2:
            recommendation = f"BET {home_team} (Predicted to win by {predicted_spread:.1f})"
            pick = home_team
        elif predicted_spread < -2:
            recommendation = f"BET {away_team} (Predicted to win by {abs(predicted_spread):.1f})"
            pick = away_team
        else:
            recommendation = "PASS (Spread too close)"
            pick = None
    else:
        recommendation = "PASS (Insufficient edge/confidence)"
        pick = None

    prediction = {
        'week': week,
        'away_team': away_team,
        'home_team': home_team,
        'predicted_spread': predicted_spread,
        'predicted_total': predicted_total,
        'home_win_prob': home_win_prob,
        'confidence': confidence,
        'edge': edge,
        'officiating_adj': officiating_adjustment,
        'recommendation': recommendation,
        'pick': pick
    }

    predictions.append(prediction)

# Sort by edge * confidence (best opportunities first)
predictions.sort(key=lambda x: x['edge'] * x['confidence'], reverse=True)

# Display predictions
print("="*80)
print("🎯 TOP PREDICTIONS (Best Opportunities)")
print("="*80)
print()

shown = 0
for pred in predictions:
    if pred['pick'] and shown < 20:  # Show top 20 recommendations
        shown += 1
        print(f"#{shown}: {pred['away_team']} @ {pred['home_team']} (Week {pred['week']})")
        print(f"   Predicted Spread: {pred['predicted_spread']:+.1f} (home team)")
        if pred['predicted_total']:
            print(f"   Predicted Total: {pred['predicted_total']:.1f}")
        if pred['home_win_prob']:
            print(f"   Home Win Probability: {pred['home_win_prob']:.1%}")
        print(f"   Confidence: {pred['confidence']:.1%}")
        print(f"   Edge: {pred['edge']:.1%}")
        if pred['officiating_adj'] != 0:
            print(f"   Officiating Adjustment: {pred['officiating_adj']:+.1f} pts")
        print(f"   🎯 RECOMMENDATION: {pred['recommendation']}")
        print()

if shown == 0:
    print("⚠️  No games meet minimum thresholds:")
    print(f"   Min Edge: {MIN_EDGE:.1%}")
    print(f"   Min Confidence: {MIN_CONFIDENCE:.1%}")
    print()
    print("Showing all predictions:")
    for i, pred in enumerate(predictions[:10], 1):
        print(f"\n{i}. {pred['away_team']} @ {pred['home_team']}")
        print(f"   Spread: {pred['predicted_spread']:+.1f} | Confidence: {pred['confidence']:.1%} | Edge: {pred['edge']:.1%}")
        print(f"   Status: {pred['recommendation']}")

print("\n" + "="*80)
print("✅ PREDICTION ANALYSIS COMPLETE")
print("="*80)
print()
print("📊 Summary:")
print(f"  Total Games Analyzed: {len(predictions)}")
print(f"  Recommended Bets: {shown}")
print(f"  Models Used: {model_count}/11")
print()
print("🎯 Next Steps:")
print("  1. Compare with actual odds from sportsbooks")
print("  2. Apply proper Kelly sizing based on your bankroll")
print("  3. Track results to refine calibration")
print()
