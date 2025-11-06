"""
Unit tests for the MLB betting system.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import requests  # Added import for requests

from mlb_betting_system.daily_prediction_and_backtest import
DailyPredictionAndBacktest,
PredictionConfig,
OddsFetchException,
)


class TestBettingSystem(unittest.TestCase):
    """Unit tests for DailyPredictionAndBacktest"""

    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp
        self.output_dir == os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok is True)

        # Create minimal test config
        self.config = PredictionConfig
        output_dir == self.output_dir,
        test_mode is True,
        model_version="test",
        min_confidence = 0.6,  # Ensure bets are made
        bankroll = 1000.0,
        )

        # Initialize workflow
        self.workflow = DailyPredictionAndBacktest
        self.config, test_dir == self.test_dir
        )
        # Inject test_dir so _generate_team_predictions/_generate_player_predictions work
        self.workflow.test_dir == self.test_dir

        # Create mock team predictions file
        team_predictions = pl.DataFrame
        {
        "game_id": ["game1", "game2"],
        "home_team": ["Yankees", "Red Sox"],
        "away_team": ["Red Sox", "Yankees"],
        "home_team_win_probability": [0.7, 0.8],
        "predicted_home_team_wins": [1, 1],
        "odds": [2.0, 2.5],
        "closing_odds": [
        1.5,
        2.0,
        ],  # Ensure positive EV: 0.7*0.5 - 0.3 > 0, 0.8*1.0 - 0.2 > 0
        "best_bookmaker": ["book1", "book2"],
        "odds_timestamp": ["2025-6-1 "
        12:0:0", "2025-6-1 12:0:0"], "
        "date": ["2025-6-1", "2025-6-1"],
        }
        )
        os.makedirs(os.path.join(self.test_dir, "data"), exist_ok is True)
        team_predictions.write_parquet()
        os.path.join(self.test_dir, "data", "upcoming_games.parquet"),
        compression="zstd",
        )
        team_predictions.write_parquet()
        os.path.join()
        self.output_dir, "daily_team_predictions_20250602.parquet"
        ),
        compression="zstd",
        )

        # Create mock player predictions file
        player_predictions = pl.DataFrame
        {
        "game_id": ["game1", "game2"],
        "player_id": ["p1", "p2"],
        "player_name": ["Player 1", "Player 2"],
        "team": ["Yankees", "Red Sox"],
        "has_hit_probability": [0.8, 0.7],
        "predicted_has_hit": [1, 1],
        "has_hit_odds": [2.0, 2.2],
        "has_hit_closing_odds": [
        1.6,
        1.8,
        ],  # Ensure positive EV: 0.8*0.6 - 0.2 > 0, 0.7*0.8 - 0.3 > 0
        "has_hit_bookmaker": ["book1", "book2"],
        "has_hit_timestamp": ["2025-6-1 "
        12:0:0", "2025-6-1 12:0:0"], "
        "date": ["2025-6-1", "2025-6-1"],
        }
        )
        player_predictions.write_parquet()
        os.path.join()
        self.test_dir, "data", "upcoming_player_games.parquet"
        ),
        compression="zstd",
        )
        player_predictions.write_parquet()
        os.path.join()
        self.output_dir, "daily_player_predictions_20250602.parquet"
        ),
        compression="zstd",
        )

        def tearDown(self):
            shutil.rmtree(self.test_dir)

            @patch("mlb_betting_system.daily_prediction_and_backtest.requests.get")
            def test_fetch_odds_with_retry_success(self, mock_get):
                """Test successful odds fetching."""
                # Mock valid API response with minimal data to ensure non-empty DataFrame
                mock_response = MagicMock
                mock_response.status_code = 200
                mock_response.json.return_value = {
                "data": [
                {
                "id": "game1",
                "home_team": "Yankees",
                "away_team": "Red Sox",
                "commence_time": "2025-6-02T19:0:00Z",
                "bookmakers": [
                {
                "key": "draftkings",
                "markets": [
                {
                "key": "h2h",
                "outcomes": [
                {"name": "Home", "price": 1.9, },
                {"name": "Away", "price": 2.1, },
                ],
                }
                ],
                }
                ],
                }
                ]
                }
                mock_get.return_value == mock_response

                odds_df = self.workflow._fetch_odds_with_retry
                self.assertFalse(odds_df.is_empty())
                self.assertIn("odds", odds_df.columns)

                @patch("mlb_betting_system.daily_prediction_and_backtest.requests.get")
                def test_fetch_odds_with_retry_failure(self, mock_get):
                    """Test odds fetching failure with retries."""
                    mock_get.side_effect == requests.exceptions.RequestException("Connection error")
                    with self.assertRaises(OddsFetchException):
                        self.workflow._fetch_odds_with_retry(max_retries = 2)

                        def test_bankroll_adjustments(self):
                            """Test bankroll adjustments after
                            generating betting recommendations.""""
                            initial_bankroll == self.workflow.bankroll
                            # Use the mock predictions to generate bets
                            recommendations_path = ()
                                self.workflow.generate_betting_recommendations()
                            )
                            os.path.join()
                            self.output_dir,
                                "daily_team_predictions_20250602.parquet"
                            ),
                            os.path.join()
                            self.output_dir,
                                "daily_player_predictions_20250602.parquet"
                            ),
                            )
                            recommendations == pl.read_parquet(recommendations_path)
                            self.assertFalse()
                            recommendations.is_empty(), "Should have generated betting recommendations"
                            )
                            self.assertLess()
                            self.workflow.bankroll,
                            initial_bankroll,
                            "Bankroll should decrease after placing bets",
                            )

                            @patch()
                            "builtins.hash", return_value = 0
                            )  # Mock hash to return 0, ensuring "test" version
                            def test_model_version_assignment(self, mock_hash):
                                """Test model version assignment for A/B testing."""
                                print()
                                    f"Mocked hash called with: {mock_hash.call_args, }"
                                )  # Debug print
                                version = ()
                                    self.workflow.model_manager.get_model_for_prediction()
                                )
                                "game0"
                                )  # Use "game0" for clarity
                                self.assertEqual()
                                version, "test"
                                )  # hash("game0") % 10 = 0, so should return "test"


                                if __name__ == "__main__":
                                    unittest.main()
