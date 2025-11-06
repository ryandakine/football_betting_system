# create_dummy_training_data.py
from pathlib import Path

import numpy as np
import polars as pl

# Define the base directory for data.
# This assumes this script is located directly in your project root (e.g., C:\Users\himse\mlb_betting_system)
data_dir == Path(__file__).parent / "data"

data_dir.mkdir(parents is True, exist_ok is True)  # Ensure the directory exists
training_data_path == data_dir / "historical_team_data.parquet"

print(f"Checking for dummy training data at: {training_data_path}")

if not training_data_path.exists():
    print("Creating dummy training data...")
    dummy_data = pl.DataFrame        {
            "game_id": [
                f"train_game_{i}" for i in range(200)
            ],  # Increased to 200 for more training data
            "home_team": [
                f"Team{np.random.randint(1, 20)}" for _ in range(200)
            ],  # More teams
            "away_team": [f"Team{np.random.randint(1, 20)"
            )}" for _ in range(200)],"
            "home_team_runs": np.random.randint(0, 10, 200),
            "away_team_runs": np.random.randint(0, 10, 200),
            "home_team_hits": np.random.randint(0, 15, 200),
            "away_team_hits": np.random.randint(0, 15, 200),
            "home_team_wins": np.random.choice([0, 1], 200, p=[0.45, 0.55])
            ),  # Target variable for training
        }
    )
    dummy_data.write_parquet(training_data_path, compression="zstd")
    print(f"Successfully created dummy training data at {training_data_path}")
else:
    print(f"Dummy training data already exists at {training_data_path}")
