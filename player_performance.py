# File: mlb_betting_system/player_performance.py

import logging
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score

logger = logging.getLogger(__name__)


def _load_data_safe(path: str, schema: dict = None) -> pl.DataFrame:
    """
    Safely loads a Polars DataFrame,
    returns empty DataFrame if not found/error.
    """
    if os.path.exists(path):
        try:
            df = pl.read_parquet(path)
            logger.debug(f"Successfully loaded data from {path} (rows: {df.height})")
            return df
        except Exception as e:
            logger.error(
                f"Error loading data from {path}: {e}. " "Returning empty DataFrame."
            )
            return pl.DataFrame(schema=schema if schema else {})
    else:
        logger.warning(f"Data file not found: {path}. Returning empty DataFrame.")
        return pl.DataFrame(schema=schema if schema else {})


class PlayerPerformancePredictor:
    def __init__(self, model_type: str = "random_forest", model_dir: str = "models"):
        """
        Initializes the player performance predictor.
        Args:
            model_type (str): Type of model to use ('random_forest').
            model_dir (str): Directory to save/load models.
        """
        self.model_type = model_type
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {}  # Stores models for different prop bets
        self.feature_columns = {}  # Stores feature names used for each model

    def _get_model_path(self, target_col: str):
        """Constructs the full path for a specific model file."""
        return os.path.join(self.model_dir, f"player_{target_col}_model.pkl")

    def load_model(self, target_col: str):
        """Loads a trained model for a specific target from disk."""
        model_path = self._get_model_path(target_col)
        if os.path.exists(model_path):
            try:
                self.models[target_col] = joblib.load(model_path)
                logger.info(
                    f"Loaded {self.model_type} model for {target_col} "
                    f"from {model_path}"
                )
                return True
            except Exception as e:
                logger.error(
                    f"Error loading model for {target_col} from {model_path}: "
                    f"{e}. Retraining will be attempted."
                )
                return False
        else:
            logger.warning(
                f"Model for {target_col} not found at {model_path}. "
                "A new model will be trained."
            )
            return False

    def validate_features(
        self, df: pl.DataFrame, target_col: str, is_training: bool = False
    ) -> bool:
        """
        Validates that a DataFrame contains the expected feature columns
        for a given target. If is_training is True, it sets the
        feature_columns for the target.
        """
        if is_training:  # When training, establish the feature columns
            # Exclude identifying columns and target column itself from features
            excluded_cols = [
                "game_id",
                "date",
                "player_id",
                "player_name",
                "team",
                "opponent_team",
                "opponent_team_starting_pitcher",
                target_col,  # The actual target column for this model
            ]
            self.feature_columns[target_col] = [
                col for col in df.columns if col not in excluded_cols
            ]
            if not self.feature_columns[target_col]:
                logger.error(
                    f"No valid features found for training {target_col} "
                    "after exclusion. Check data and schema."
                )
                return False
            logger.debug(
                f"Features set for {target_col} training: "
                f"{self.feature_columns[target_col]}"
            )
            return True
        else:  # When predicting, ensure all expected features are present
            required_features = self.feature_columns.get(target_col)
            if required_features is None:
                logger.warning(
                    f"No feature columns defined for {target_col}. "
                    "Model might not have been trained or loaded correctly. "
                    "Attempting to infer."
                )
                # Fallback to a general exclusion if feature_columns was not set
                required_features = [
                    col
                    for col in df.columns
                    if col
                    not in [
                        "game_id",
                        "date",
                        "player_id",
                        "player_name",
                        "team",
                        "opponent_team",
                        "opponent_team_starting_pitcher",
                    ]
                ]
                if not required_features:
                    logger.error(
                        f"Cannot infer features for {target_col} prediction. "
                        "Returning False."
                    )
                    return False

            missing = [col for col in required_features if col not in df.columns]
            if missing:
                logger.error(
                    f"Missing features in prediction data for {target_col}: "
                    f"{missing}. Cannot predict."
                )
                return False

            # Check for extra features not used in training
            excluded_cols = [
                "game_id",
                "date",
                "player_id",
                "player_name",
                "team",
                "opponent_team",
                "opponent_team_starting_pitcher",
            ]
            extra_features = [
                col
                for col in df.columns
                if col not in required_features and col not in excluded_cols
            ]
            if extra_features:
                logger.warning(
                    f"Extra features found in {target_col} prediction data: "
                    f"{extra_features}. They will be ignored."
                )

            return True

    def train_model(
        self,
        player_features_df: pl.DataFrame,
        historical_player_game_logs: pl.DataFrame,
        target_col: str,
    ):
        """
        Trains a model for a specific player performance target.
        Includes cross-validation and hyperparameter tuning.

        Args:
            player_features_df: DataFrame of engineered features.
            historical_player_game_logs: Game-by-game player historical data.
            target_col: The column name representing the target variable.
        """
        logger.info(f"Training {self.model_type} model for {target_col}...")

        # 1. Generate Target Variable: Join features with historical logs
        game_logs_processed = historical_player_game_logs.with_columns(
            [
                pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d"),
                pl.col("game_id").cast(pl.Utf8),
                pl.col("player_id").cast(pl.Utf8),
                pl.col(target_col).cast(pl.Float64),  # Ensure target is numeric
            ]
        )

        # Join player features with game logs to get actual outcomes
        training_data = player_features_df.join(
            game_logs_processed.select(
                ["game_id", "player_id", "Date", target_col]
            ).rename({"Date": "actual_date"}),
            on=["game_id", "player_id"],
            how="inner",  # Only keep records where both features and actuals exist
        )

        # Filter out records where actual target is null
        training_data = training_data.filter(pl.col(target_col).is_not_null())

        if training_data.is_empty():
            logger.error(
                f"No valid training data found for {target_col} "
                "after joining and filtering. Skipping training."
            )
            raise ValueError(f"Insufficient training data for {target_col}.")

        # Validate and set feature columns for training
        if not self.validate_features(training_data, target_col, is_training=True):
            logger.error(
                f"Feature validation failed for {target_col} training. " "Skipping."
            )
            raise ValueError(f"Feature validation failed for {target_col}.")

        X = training_data.select(self.feature_columns[target_col]).to_numpy()
        y = training_data.get_column(target_col).to_numpy()

        if X.size == 0 or y.size == 0:
            logger.error(
                f"Feature matrix or target vector is empty for {target_col}. "
                "Cannot train model."
            )
            raise ValueError(f"Empty feature matrix or target vector for {target_col}.")

        # Initialize model and perform Hyperparameter Tuning
        if self.model_type == "random_forest":
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            # Define a smaller, faster grid for demonstration
            param_grid = {
                "n_estimators": [50, 100],  # Fewer estimators for quicker test
                "max_depth": [None, 10],  # Less depth for quicker test
                "min_samples_split": [2, 5],
            }
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=3,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=0,
            )
            logger.info(f"Starting GridSearchCV for {target_col}...")
            grid_search.fit(X, y)
            self.models[target_col] = grid_search.best_estimator_
            logger.info(f"Best parameters for {target_col}: {grid_search.best_params_}")

            # Evaluate best model with cross-validation
            scores = cross_val_score(
                self.models[target_col],
                X,
                y,
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            rmse_scores = np.sqrt(-scores)
            logger.info(
                f"{target_col} Cross-Validation RMSE: "
                f"{rmse_scores.mean():.3f} (+/- {rmse_scores.std():.3f})"
            )
        else:
            raise ValueError(
                f"Unsupported model type for player performance: " f"{self.model_type}"
            )

        joblib.dump(self.models[target_col], self._get_model_path(target_col))
        logger.info(
            f"Trained {self.model_type} model for {target_col} saved to "
            f"{self._get_model_path(target_col)}"
        )

    def predict_performance(
        self, player_features_df: pl.DataFrame, target_col: str
    ) -> pl.DataFrame:
        """
        Generates predictions for player performance.

        Args:
            player_features_df: DataFrame of engineered features for upcoming games.
            target_col: The name of the target being predicted.

        Returns:
            pl.DataFrame: DataFrame with game_id, player_id, player_name,
            and prediction.
        """
        if target_col not in self.models or self.models[target_col] is None:
            raise RuntimeError(
                f"Model for {target_col} is not trained or loaded. "
                "Call train_model() or load_model() first."
            )

        logger.info(f"Generating predictions for player {target_col}...")

        # Validate features before prediction
        if not self.validate_features(
            player_features_df, target_col, is_training=False
        ):
            logger.error(
                f"Prediction feature validation failed for {target_col}. "
                "Returning empty predictions."
            )
            return pl.DataFrame(
                {
                    "game_id": pl.Series([], dtype=pl.Utf8),
                    "player_id": pl.Series([], dtype=pl.Utf8),
                    "player_name": pl.Series([], dtype=pl.Utf8),
                    f"{target_col.replace('_actual', '')}_prediction": pl.Series(
                        [], dtype=pl.Float64
                    ),
                }
            )

        # Select features in the order used during training
        X_predict = player_features_df.select(
            self.feature_columns[target_col]
        ).to_numpy()

        if X_predict.size == 0:
            logger.warning(
                f"No data to predict for player {target_col}. "
                "Returning empty DataFrame."
            )
            return pl.DataFrame(
                {
                    "game_id": pl.Series([], dtype=pl.Utf8),
                    "player_id": pl.Series([], dtype=pl.Utf8),
                    "player_name": pl.Series([], dtype=pl.Utf8),
                    f"{target_col.replace('_actual', '')}_prediction": pl.Series(
                        [], dtype=pl.Float64
                    ),
                }
            )

        preds = self.models[target_col].predict(X_predict)

        predictions_df = player_features_df.select(
            ["game_id", "player_id", "player_name"]
        ).with_columns(
            pl.Series(
                f'{target_col.replace("_actual", "")}_prediction',
                preds.round().clip(lower_bound=0),
            )  # Round and ensure non-negative
        )

        logger.info(
            f"Generated predictions for {predictions_df.height} player "
            f"{target_col} outcomes."
        )
        return predictions_df


if __name__ == "__main__":
    from feature_engineering import run_feature_engineering

    from data_collection import DATA_DIR, HISTORICAL_GAME_LOGS_PATH, run_data_collection
    from logging_config import configure_logging

    configure_logging(level=logging.INFO)

    # Define paths
    HISTORICAL_PLAYER_GAME_LOGS_PATH = os.path.join(
        DATA_DIR, "historical_player_game_logs.parquet"
    )

    # Step 1: Run Data Collection
    current_test_date = datetime(2025, 6, 7)
    logger.info(
        f"--- Running Data Collection for "
        f"{current_test_date.strftime('%Y-%m-%d')} ---"
    )
    run_data_collection(
        collect_historical_fangraphs=True,
        historical_fangraphs_start_season=2020,
        historical_fangraphs_end_season=current_test_date.year - 1,
        collect_historical_game_logs=True,
        historical_game_logs_start_date=datetime(current_test_date.year - 2, 3, 20),
        historical_game_logs_end_date=current_test_date - timedelta(days=1),
        collect_daily_schedule=True,
        target_date=current_test_date,
        collect_upcoming_player_games=True,
        collect_umpire_data=True,
        collect_standings_data=True,
    )

    # Step 2: Create Mock Historical Player Game Logs
    if not os.path.exists(HISTORICAL_PLAYER_GAME_LOGS_PATH):
        logger.warning(
            "No historical player game logs found. "
            "Creating a larger mock for training."
        )
        mock_train_player_logs_data = []
        start_date = datetime(2023, 3, 30)
        end_date = datetime(2024, 10, 30)
        current_log_date = start_date
        player_ids = [
            "stant_mike01",
            "judge_aaro01",
            "dever_rafa01",
            "betts_mook01",
            "trout_mike01",
            "ohtan_sho01",
        ]

        while current_log_date <= end_date:
            for player_id in player_ids:
                mock_train_player_logs_data.append(
                    {
                        "game_id": f"game_{current_log_date.strftime('%Y%m%d')}_player_{player_id}",
                        "Date": current_log_date.strftime("%Y-%m-%d"),
                        "player_id": player_id,
                        "home_runs_actual": int(np.random.poisson(lam=0.15)),
                        "strikeouts_actual": int(np.random.poisson(lam=1.2)),
                    }
                )
            current_log_date += timedelta(days=np.random.randint(1, 3))

        pl.DataFrame(mock_train_player_logs_data).write_parquet(
            HISTORICAL_PLAYER_GAME_LOGS_PATH, compression="zstd"
        )
        logger.info(
            f"Created large mock historical_player_game_logs.parquet at "
            f"{HISTORICAL_PLAYER_GAME_LOGS_PATH}"
        )

    # Step 3: Create dummy training features
    logger.info("--- Creating Dummy Training Player Features (for model training) ---")
    num_train_player_features = 2000
    player_ids = [
        "stant_mike01",
        "judge_aaro01",
        "dever_rafa01",
        "betts_mook01",
        "trout_mike01",
        "ohtan_sho01",
    ]

    mock_train_player_features_data = {
        "game_id": [
            f"train_game_{i}_player_{np.random.choice(player_ids)}"
            for i in range(num_train_player_features)
        ],
        "date": [
            (
                current_test_date - timedelta(days=np.random.randint(1, 365 * 2))
            ).strftime("%Y-%m-%d")
            for _ in range(num_train_player_features)
        ],
        "player_id": np.random.choice(player_ids, num_train_player_features),
        "player_name": [
            f"Player {pid}"
            for pid in np.random.choice(player_ids, num_train_player_features)
        ],
        "team": np.random.choice(["NYY", "LAD"], num_train_player_features),
        "opponent_team": np.random.choice(["BOS", "SFG"], num_train_player_features),
        "opponent_team_starting_pitcher": np.random.choice(
            ["Pitcher E", "Pitcher F", "Pitcher G"], num_train_player_features
        ),
        "plate_appearances": np.random.randint(3, 6, num_train_player_features),
        "at_bats": np.random.randint(2, 5, num_train_player_features),
        "walks": np.random.randint(0, 2, num_train_player_features),
        "singles": np.random.randint(0, 2, num_train_player_features),
        "doubles": np.random.randint(0, 1, num_train_player_features),
        "triples": np.zeros(num_train_player_features, dtype=np.int64),
        "home_runs": np.random.randint(0, 2, num_train_player_features),
        "strikeouts": np.random.randint(0, 3, num_train_player_features),
        "rbis": np.random.randint(0, 4, num_train_player_features),
        "runs_scored": np.random.randint(0, 3, num_train_player_features),
        "player_career_avg": np.random.rand(num_train_player_features) * 0.3 + 0.15,
        "player_career_obp": np.random.rand(num_train_player_features) * 0.1 + 0.3,
        "player_career_slg": np.random.rand(num_train_player_features) * 0.2 + 0.3,
        "player_career_total_hrs": np.random.randint(
            10, 500, num_train_player_features
        ).astype(np.float64),
        "player_career_k_rate": np.random.rand(num_train_player_features) * 0.2 + 0.1,
        "player_career_bb_rate": np.random.rand(num_train_player_features) * 0.1 + 0.5,
        "opponent_pitcher_era": np.random.rand(num_train_player_features) * 3 + 2,
        "opponent_pitcher_fip": np.random.rand(num_train_player_features) * 3 + 2,
        "opponent_pitcher_xfip": np.random.rand(num_train_player_features) * 3 + 2,
        "opponent_pitcher_k_per_9": np.random.rand(num_train_player_features) * 3 + 7,
        "opponent_pitcher_bb_per_9": np.random.rand(num_train_player_features) * 1 + 2,
    }
    dummy_training_player_features_df = pl.DataFrame(mock_train_player_features_data)

    # Step 4: Initialize and Train PlayerPerformancePredictor models
    logger.info("--- Initializing and Training PlayerPerformancePredictor ---")
    player_predictor = PlayerPerformancePredictor()

    # Train model for 'home_runs_actual'
    try:
        player_predictor.train_model(
            dummy_training_player_features_df,
            _load_data_safe(HISTORICAL_PLAYER_GAME_LOGS_PATH),
            target_col="home_runs_actual",
        )
    except ValueError as e:
        logger.error(
            f"Home run model training failed: {e}. " "Skipping home run prediction."
        )

    # Train model for 'strikeouts_actual'
    try:
        player_predictor.train_model(
            dummy_training_player_features_df,
            _load_data_safe(HISTORICAL_PLAYER_GAME_LOGS_PATH),
            target_col="strikeouts_actual",
        )
    except ValueError as e:
        logger.error(
            f"Strikeout model training failed: {e}. " "Skipping strikeout prediction."
        )

    logger.info("Player performance predictor setup complete!")
