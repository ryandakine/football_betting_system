#!/usr/bin/env python3
"""Simple dummy data creation - guaranteed to work"""

import os

import numpy as np
import pandas as pd

print("📊 CREATING SIMPLE DUMMY BASEBALL DATA")

# Create data directory
os.makedirs("data", exist_ok=True)

# Create historical player data
print("1. Creating historical_player_data.csv...")
player_data = {
    "player_id": range(1, 101),
    "name": [f"Player{i}" for i in range(1, 101)],
    "team": [f"Team{np.random.randint(1, 31)}" for _ in range(100)],
    "position": np.random.choice(
        ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "P"], 100
    ),
    "batting_avg": np.random.uniform(0.200, 0.350, 100),
    "home_runs": np.random.randint(0, 50, 100),
    "rbi": np.random.randint(0, 120, 100),
    "era": np.random.uniform(2.00, 6.00, 100),
    "wins": np.random.randint(0, 20, 100),
    "strikeouts": np.random.randint(50, 300, 100),
}

df_players = pd.DataFrame(player_data)
df_players.to_csv("data/historical_player_data.csv", index=False)
print("   ✅ Saved historical_player_data.csv")

# Create historical team data
print("2. Creating historical_team_data.csv...")
team_data = {
    "team_id": range(1, 31),
    "team_name": [f"Team{i}" for i in range(1, 31)],
    "wins": np.random.randint(60, 110, 30),
    "losses": np.random.randint(52, 102, 30),
    "runs_scored": np.random.randint(600, 900, 30),
    "runs_allowed": np.random.randint(600, 900, 30),
    "batting_avg": np.random.uniform(0.240, 0.280, 30),
    "era": np.random.uniform(3.50, 5.00, 30),
    "home_record": [
        f"{np.random.randint(30, 55)}-{np.random.randint(26, 51)}" for _ in range(30)
    ],
    "away_record": [
        f"{np.random.randint(30, 55)}-{np.random.randint(26, 51)}" for _ in range(30)
    ],
}

df_teams = pd.DataFrame(team_data)
df_teams.to_csv("data/historical_team_data.csv", index=False)
print("   ✅ Saved historical_team_data.csv")

# Create game data
print("3. Creating historical_games.csv...")
game_data = {
    "game_id": range(1, 501),
    "date": pd.date_range("2024-04-01", periods=500, freq="D"),
    "home_team": [f"Team{np.random.randint(1, 31)}" for _ in range(500)],
    "away_team": [f"Team{np.random.randint(1, 31)}" for _ in range(500)],
    "home_score": np.random.randint(0, 15, 500),
    "away_score": np.random.randint(0, 15, 500),
    "weather": np.random.choice(["Clear", "Cloudy", "Rain", "Hot"], 500),
    "temperature": np.random.randint(60, 95, 500),
}

df_games = pd.DataFrame(game_data)
df_games.to_csv("data/historical_games.csv", index=False)
print("   ✅ Saved historical_games.csv")

# Create umpire data
print("4. Creating umpire_data.csv...")
umpire_data = {
    "umpire_id": range(1, 51),
    "name": [f"Umpire{i}" for i in range(1, 51)],
    "games_worked": np.random.randint(50, 200, 50),
    "strike_zone_size": np.random.uniform(0.8, 1.2, 50),
    "consistency_rating": np.random.uniform(7.0, 9.5, 50),
}

df_umpires = pd.DataFrame(umpire_data)
df_umpires.to_csv("data/umpire_data.csv", index=False)
print("   ✅ Saved umpire_data.csv")

print("\n🎯 DUMMY DATA CREATION COMPLETE!")
print(f"📁 Created files in ./data/ directory:")
print(f"   - historical_player_data.csv ({len(df_players)} rows)")
print(f"   - historical_team_data.csv ({len(df_teams)} rows)")
print(f"   - historical_games.csv ({len(df_games)} rows)")
print(f"   - umpire_data.csv ({len(df_umpires)} rows)")
print(f"\n✅ Your AI can now load historical data for ML predictions!")
