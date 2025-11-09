#!/usr/bin/env python3
"""
Train NCAA models on 10 years of data and predict this weekend's games.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
import pickle

print("🏈 NCAA TRAINING & PREDICTION SYSTEM")
print("=" * 80)

# Load historical data
data_dir = Path("data/football/historical/ncaaf")
all_games = []

print("\n📊 Loading 10 years of NCAA data...")
for year in range(2015, 2025):
    games_file = data_dir / f"ncaaf_{year}_games.json"
    stats_file = data_dir / f"ncaaf_{year}_stats.json"
    sp_file = data_dir / f"ncaaf_{year}_sp_ratings.json"
    
    if games_file.exists():
        with open(games_file) as f:
            games = json.load(f)
            print(f"  ✅ {year}: {len(games)} games")
            all_games.extend(games)

print(f"\n✅ Total games loaded: {len(all_games)}")

# Prepare features
print("\n🔧 Engineering features...")
features_list = []
labels = []

for game in all_games:
    if not game.get('completed') or game.get('homePoints') is None:
        continue
    
    # Basic features
    home_points = game.get('homePoints', 0)
    away_points = game.get('awayPoints', 0)
    
    # Home team won?
    home_won = 1 if home_points > away_points else 0
    
    # Features
    features = {
        'home_conference_game': 1 if game.get('conferenceGame') else 0,
        'neutral_site': 1 if game.get('neutralSite') else 0,
        'season_type': 1 if game.get('seasonType') == 'postseason' else 0,
        'week': int(game.get('week', 1)),
        'home_team_encoded': hash(game.get('homeTeam', '')) % 1000,
        'away_team_encoded': hash(game.get('awayTeam', '')) % 1000,
    }
    
    features_list.append(features)
    labels.append(home_won)

# Convert to DataFrame
df = pd.DataFrame(features_list)
y = np.array(labels)

print(f"✅ Prepared {len(df)} training samples")
print(f"   Home win rate: {y.mean():.1%}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🤖 Training models...")
print(f"   Training set: {len(X_train)} games")
print(f"   Test set: {len(X_test)} games")

# Train Random Forest
print("\n   Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)
print(f"   ✅ Random Forest Accuracy: {rf_acc:.1%}")

# Train Gradient Boosting
print("   Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
gb_acc = gb.score(X_test, y_test)
print(f"   ✅ Gradient Boosting Accuracy: {gb_acc:.1%}")

# Ensemble
print("\n🎯 Creating ensemble...")
rf_preds = rf.predict_proba(X_test)[:, 1]
gb_preds = gb.predict_proba(X_test)[:, 1]
ensemble_preds = (rf_preds + gb_preds) / 2
ensemble_acc = ((ensemble_preds > 0.5) == y_test).mean()
print(f"✅ Ensemble Accuracy: {ensemble_acc:.1%}")

# Save models
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

with open(models_dir / "ncaa_rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)
with open(models_dir / "ncaa_gb_model.pkl", "wb") as f:
    pickle.dump(gb, f)

print(f"\n💾 Models saved to models/")

# Get this weekend's games
print("\n" + "=" * 80)
print("🏈 PREDICTING THIS WEEKEND'S GAMES")
print("=" * 80)

api_key = os.getenv("ODDS_API_KEY", "e84d496405014d166f5dce95094ea024")
url = f"https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds/?apiKey={api_key}&regions=us&markets=spreads"

print("\n📊 Fetching games from Odds API...")
try:
    resp = requests.get(url, timeout=10)
    games = resp.json()
    
    # Filter for this weekend (next 72 hours)
    now = datetime.utcnow()
    weekend_games = []
    
    for g in games:
        game_time = datetime.fromisoformat(g['commence_time'].replace('Z', '+00:00'))
        hours_away = (game_time - now.replace(tzinfo=game_time.tzinfo)).total_seconds() / 3600
        
        if 0 <= hours_away <= 72:
            weekend_games.append((g, hours_away))
    
    weekend_games.sort(key=lambda x: x[1])
    
    print(f"✅ Found {len(weekend_games)} games this weekend\n")
    
    if not weekend_games:
        print("⚠️  No games in next 72 hours. Try again closer to Saturday!")
    else:
        predictions = []
        
        for g, hours in weekend_games:
            # Prepare features for prediction
            pred_features = {
                'home_conference_game': 0,  # Unknown
                'neutral_site': 0,
                'season_type': 0,
                'week': 11,  # Current week
                'home_team_encoded': hash(g['home_team']) % 1000,
                'away_team_encoded': hash(g['away_team']) % 1000,
            }
            
            pred_df = pd.DataFrame([pred_features])
            
            # Get ensemble prediction
            rf_prob = rf.predict_proba(pred_df)[0, 1]
            gb_prob = gb.predict_proba(pred_df)[0, 1]
            ensemble_prob = (rf_prob + gb_prob) / 2
            
            # Get spread
            spread = None
            for book in g.get('bookmakers', [])[:1]:
                for market in book.get('markets', []):
                    if market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == g['home_team']:
                                spread = outcome['point']
            
            predictions.append({
                'game': f"{g['away_team']} @ {g['home_team']}",
                'home_win_prob': ensemble_prob,
                'confidence': max(ensemble_prob, 1 - ensemble_prob),
                'edge': abs(ensemble_prob - 0.5),
                'spread': spread,
                'hours': hours
            })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        print("🎯 TOP PREDICTIONS (Sorted by Confidence):\n")
        for i, pred in enumerate(predictions[:10], 1):
            pick = "HOME" if pred['home_win_prob'] > 0.5 else "AWAY"
            conf = pred['confidence'] * 100
            edge = pred['edge'] * 100
            
            print(f"{i:2d}. {pred['game']}")
            print(f"    Pick: {pick} | Prob: {pred['home_win_prob']:.1%} | Confidence: {conf:.1f}% | Edge: {edge:.1f}%")
            if pred['spread']:
                print(f"    Spread: {pred['spread']} | Starts in: {pred['hours']:.1f} hours")
            print()

except Exception as e:
    print(f"❌ Error fetching games: {e}")

print("\n" + "=" * 80)
print("✅ Training and prediction complete!")
print("=" * 80)
