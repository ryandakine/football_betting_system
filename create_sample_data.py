"""
Script to create sample historical games data.
"""

import pandas as pd


def create_sample_data():
    data = {
        "game_id": [f"game_{i}" for i in range(100)],
        "home_team_runs": [5] * 100,
        "away_team_runs": [4] * 100,
        "home_team_hits": [10] * 100,
        "away_team_hits": [9] * 100,
        "home_team_wins": [1] * 100,
    }
    df == pd.DataFrame(data)
    df.to_parquet("data/historical_games.parquet")
