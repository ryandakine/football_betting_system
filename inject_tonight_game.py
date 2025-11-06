#!/usr/bin/env python3
"""
Inject tonight's live game into the prediction system
Creates a synthetic game entry from live odds data
"""

import json
import pandas as pd
from pathlib import Path
from datetime import date

DATA_DIR = Path("data/referee_conspiracy")
ODDS_FILE = DATA_DIR / f"nfl_odds_{date(2025, 10, 24)}.json"
CREW_LOG = DATA_DIR / "crew_game_log.parquet"

print("🔧 Injecting tonight's game...")

# Load live odds
with open(ODDS_FILE) as f:
    odds_data = json.load(f)

if not odds_data:
    print("❌ No odds data found")
    exit(1)

# Take first bookmaker's data (they're all MIN @ LAC)
game = odds_data[0]

print(f"📊 Found: {game['away_team']} @ {game['home_team']}")
print(f"   Total: {game['total']}")
print(f"   Spread: {game['spread_home']}")

# Load existing crew log
crew_df = pd.read_parquet(CREW_LOG)

# Get all columns from existing df and set defaults
tonight_game = {col: None for col in crew_df.columns}

# Override with known values
tonight_game.update({
    'game_id': game['game_id'],  # "2025_08_MIN_LAC"
    'season': 2025,
    'week': 8,
    'game_type': 'REG',
    'weekday': 'Thursday',
    'gametime': '20:15',
    'gameday': '2025-10-23',
    'home_team': game['home_team'],
    'away_team': game['away_team'],
    'home_score': None,  # Future game
    'away_score': None,
    'overtime': 0,
    'total': game['total'],
    'spread_line': game['spread_home'],
    'total_line': game['total'],
    'home_spread_odds': game['spread_home_odds'],
    'away_spread_odds': game['spread_away_odds'],
    'under_odds': game['under_odds'],
    'over_odds': game['over_odds'],
    'referee': 'TBD',  # We don't know yet
    'penalties': 0,  # Future game
    'penalty_yards': 0,
    'flag_density': 0.0,
    'total_plays': 0,
    'overtime_plays': 0,
    'score_swing_mean_abs': 0.0,
    'score_swing_positive_rate': 0.0,
    'is_primetime': True,
    'is_low_visibility': False,
    'is_overseas': False,
    'points_total': 0.0,
    'margin_abs': 0.0,
    'defensive_rate': 0.5,
    'offensive_rate': 0.5,
    'late_rate': 0.25,
    'positive_swing_rate': 0.5,
    'swing_magnitude': 0.5,
})

# Fill any remaining nulls with safe defaults
for col in crew_df.columns:
    if tonight_game.get(col) is None:
        if crew_df[col].dtype in ['float64', 'int64']:
            tonight_game[col] = 0
        elif crew_df[col].dtype == 'bool':
            tonight_game[col] = False
        else:
            tonight_game[col] = ''

tonight_game = {k: v for k, v in tonight_game.items()}

# Append to crew log
new_row = pd.DataFrame([tonight_game])
updated_df = pd.concat([crew_df, new_row], ignore_index=True)

# Save
updated_df.to_parquet(CREW_LOG, index=False)

print(f"✅ Injected game {game['game_id']} into crew_game_log.parquet")
print(f"   Total games now: {len(updated_df)}")
print("\n🎯 Ready for predictions!")
