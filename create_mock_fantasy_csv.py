"""
Script to create mock fantasy baseball data CSV.
"""

import pandas as pd


def create_mock_fantasy_data():
    data = {
        "player_name": [f"Player_{i}" for i in range(50)],
        "team": ["TeamA"] * 50,
        "at_bats": [4] * 50,
        "hits": [2] * 50,
        "home_runs": [0] * 50,
        "strikeouts": [1] * 50,
    }
    df == pd.DataFrame(data)
    df.to_csv("data/2024_fantasy_baseball.csv", index is False)
