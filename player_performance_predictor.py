"""
Gradient Boosting Model for Player Performance Prediction
YOLO Mode Implementation - Production-Ready ML Pipeline

Uses LightGBM for superior performance on tabular football data:
- Player performance prediction (fantasy points, stats)
- Feature importance analysis
- Hyperparameter optimization
- Model validation and metrics
- Production deployment ready
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import optuna
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class PlayerPerformancePredictor:
    """
    Professional gradient boosting model for NFL player performance prediction.
    YOLO Mode: Cutting-edge ML with automated optimization.
    """
    
    def __init__(self, model_dir: str = "models/player_performance"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
        self.best_params = {}
        
        # Model configurations for different player positions
        self.position_configs = {
            "QB": {
                "target_cols": ["passing_yds", "passing_tds", "qb_rating", "fantasy_points"],
                "feature_cols": [
                    "season", "week", "age", "experience_years", "height_inches", "weight_lbs",
                    "games_played", "games_started", "completion_pct", "yards_per_attempt",
                    "td_to_int_ratio", "team_offensive_epa", "opponent_defensive_epa",
                    "home_advantage", "rest_days", "against_pass_rank"
                ]
            },
            "RB": {
                "target_cols": ["rushing_yds", "rushing_tds", "receiving_yds", "receiving_tds", "fantasy_points"],
                "feature_cols": [
                    "season", "week", "age", "experience_years", "height_inches", "weight_lbs",
                    "games_played", "games_started", "yards_per_carry", "yards_after_contact",
                    "broken_tackle_rate", "team_offensive_epa", "opponent_defensive_epa",
                    "home_advantage", "rest_days", "against_run_rank"
                ]
            },
            "WR": {
                "target_cols": ["receiving_yds", "receiving_tds", "receptions", "fantasy_points"],
                "feature_cols": [
                    "season", "week", "age", "experience_years", "height_inches", "weight_lbs",
                    "games_played", "games_started", "yards_per_reception", "catch_percentage",
                    "yac_per_reception", "team_offensive_epa", "opponent_defensive_epa",
                    "home_advantage", "rest_days", "against_pass_rank", "target_share"
                ]
            },
            "TE": {
                "target_cols": ["receiving_yds", "receiving_tds", "receptions", "fantasy_points"],
                "feature_cols": [
                    "season", "week", "age", "experience_years", "height_inches", "weight_lbs",
                    "games_played", "games_started", "yards_per_reception", "catch_percentage",
                    "yac_per_reception", "team_offensive_epa", "opponent_defensive_epa",
                    "home_advantage", "rest_days", "against_pass_rank", "target_share"
                ]
            }
        }
    
    def prepare_features(self, df: pd.DataFrame, position: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features and targets for a specific position.
        """
        logger.info(f"Preparing features for {position} position")
        
        config = self.position_configs.get(position, self.position_configs["QB"])
        feature_cols = config["feature_cols"]
        target_cols = config["target_cols"]
        
        # Filter for position
        if "position" in df.columns:
            pos_df = df[df["position"] == position].copy()
        else:
            pos_df = df.copy()
        
        if pos_df.empty:
            logger.warning(f"No data found for position {position}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_cols if col in pos_df.columns]
        missing_features = [col for col in feature_cols if col not in pos_df.columns]
        
        if missing_features:
            logger.warning(f"Missing features for {position}: {missing_features}")
        
        # Prepare features
        X = pos_df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Prepare targets (create fantasy points if not present)
        y = pd.DataFrame(index=X.index)
        for target in target_cols:
            if target in pos_df.columns:
                y[target] = pos_df[target]
            elif target == "fantasy_points":
                y[target] = self._calculate_fantasy_points(pos_df, position)
            else:
                y[target] = 0  # Default for missing targets
        
        # Scale features
        if not X.empty:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        logger.info(f"Prepared {len(X_scaled)} samples with {len(available_features)} features for {position}")
        return X_scaled, y
    
    def _calculate_fantasy_points(self, df: pd.DataFrame, position: str) -> pd.Series:
        """
        Calculate fantasy points using standard scoring.
        """
        fantasy_points = pd.Series(0, index=df.index)
        
        if position == "QB":
            # QB scoring: 1 pt per 25 pass yds, 4 pts per pass TD, -2 per INT, 0.1 per rush yd, 6 per rush TD
            if "passing_yds" in df.columns:
                fantasy_points += df["passing_yds"].fillna(0) / 25
            if "passing_tds" in df.columns:
                fantasy_points += df["passing_tds"].fillna(0) * 4
            if "interceptions" in df.columns:
                fantasy_points -= df["interceptions"].fillna(0) * 2
            if "rushing_yds" in df.columns:
                fantasy_points += df["rushing_yds"].fillna(0) * 0.1
            if "rushing_tds" in df.columns:
                fantasy_points += df["rushing_tds"].fillna(0) * 6
        
        elif position in ["RB", "WR", "TE"]:
            # Skill position scoring: 0.1 per rush/rec yd, 6 per rush/rec TD, 0.5 per reception
            if "rushing_yds" in df.columns:
                fantasy_points += df["rushing_yds"].fillna(0) * 0.1
            if "rushing_tds" in df.columns:
                fantasy_points += df["rushing_tds"].fillna(0) * 6
            if "receiving_yds" in df.columns:
                fantasy_points += df["receiving_yds"].fillna(0) * 0.1
            if "receiving_tds" in df.columns:
                fantasy_points += df["receiving_tds"].fillna(0) * 6
            if "receptions" in df.columns:
                fantasy_points += df["receptions"].fillna(0) * 0.5
        
        return fantasy_points
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, position: str) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters using Optuna.
        """
        logger.info(f"Optimizing hyperparameters for {position}")
        
        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "mae",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            }
            
            # Cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mae_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                
                y_pred = model.predict(X_val)
                mae_scores.append(mean_absolute_error(y_val, y_pred))
            
            return np.mean(mae_scores)
        
        # Run optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50, timeout=300)  # 5 minutes timeout
        
        best_params = study.best_params
        best_params.update({
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbosity": -1,
        })
        
        self.best_params[position] = best_params
        logger.info(f"Best MAE for {position}: {study.best_value:.4f}")
        
        return best_params
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, position: str, target: str) -> lgb.Booster:
        """
        Train LightGBM model for a specific position and target.
        """
        logger.info(f"Training {position} model for {target}")
        
        if X.empty or y.empty:
            logger.warning(f"No data available for {position} {target}")
            return None
        
        # Get optimized parameters
        if position not in self.best_params:
            self.best_params[position] = self.optimize_hyperparameters(X, y, position)
        
        params = self.best_params[position]
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        
        # Store model
        model_key = f"{position}_{target}"
        self.models[model_key] = model
        
        # Calculate feature importance
        self.feature_importance[model_key] = dict(zip(
            X.columns,
            model.feature_importance(importance_type="gain")
        ))
        
        # Evaluate model
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        logger.info(f"{position} {target} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return model
    
    def train_all_models(self, df: pd.DataFrame):
        """
        Train models for all positions and targets.
        """
        logger.info("Training all player performance models")
        
        for position in self.position_configs.keys():
            logger.info(f"Training models for {position}")
            
            X, y = self.prepare_features(df, position)
            
            if X.empty:
                logger.warning(f"Skipping {position} - no data available")
                continue
            
            config = self.position_configs[position]
            for target in config["target_cols"]:
                if target in y.columns:
                    model = self.train_model(X, y[target], position, target)
                    if model:
                        # Save model
                        model_path = self.model_dir / f"{position}_{target}_model.pkl"
                        joblib.dump(model, model_path)
                        logger.info(f"Saved {position} {target} model to {model_path}")
    
    def predict_player_performance(self, features: pd.DataFrame, position: str, target: str) -> np.ndarray:
        """
        Predict player performance for a specific target.
        """
        model_key = f"{position}_{target}"
        
        if model_key not in self.models:
            # Try to load from disk
            model_path = self.model_dir / f"{model_key}_model.pkl"
            if model_path.exists():
                self.models[model_key] = joblib.load(model_path)
            else:
                logger.error(f"No model found for {model_key}")
                return np.array([])
        
        model = self.models[model_key]
        
        # Prepare features
        config = self.position_configs.get(position, self.position_configs["QB"])
        feature_cols = [col for col in config["feature_cols"] if col in features.columns]
        
        if not feature_cols:
            logger.error(f"No valid features found for {position}")
            return np.array([])
        
        X = features[feature_cols].fillna(features[feature_cols].mean())
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        predictions = model.predict(X_scaled)
        return predictions
    
    def get_feature_importance(self, position: str, target: str) -> Dict[str, float]:
        """
        Get feature importance for a specific model.
        """
        model_key = f"{position}_{target}"
        return self.feature_importance.get(model_key, {})
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, position: str, target: str) -> Dict[str, float]:
        """
        Evaluate model performance.
        """
        predictions = self.predict_player_performance(X, position, target)
        
        if len(predictions) == 0:
            return {}
        
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mean_prediction": np.mean(predictions),
            "std_prediction": np.std(predictions)
        }
    
    def save_models(self):
        """Save all models and configurations."""
        for model_key, model in self.models.items():
            model_path = self.model_dir / f"{model_key}_model.pkl"
            joblib.dump(model, model_path)
        
        # Save configurations
        config_path = self.model_dir / "model_config.pkl"
        config = {
            "best_params": self.best_params,
            "feature_importance": self.feature_importance,
            "position_configs": self.position_configs
        }
        joblib.dump(config, config_path)
        
        logger.info(f"Saved all models and configurations to {self.model_dir}")
    
    def load_models(self):
        """Load all models and configurations."""
        config_path = self.model_dir / "model_config.pkl"
        if config_path.exists():
            config = joblib.load(config_path)
            self.best_params = config.get("best_params", {})
            self.feature_importance = config.get("feature_importance", {})
            self.position_configs = config.get("position_configs", self.position_configs)
        
        # Load individual models
        for position in self.position_configs.keys():
            for target in self.position_configs[position]["target_cols"]:
                model_key = f"{position}_{target}"
                model_path = self.model_dir / f"{model_key}_model.pkl"
                if model_path.exists():
                    self.models[model_key] = joblib.load(model_path)
        
        logger.info(f"Loaded models from {self.model_dir}")

# Example usage and testing
if __name__ == "__main__":
    predictor = PlayerPerformancePredictor()
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        "position": ["QB", "RB", "WR", "QB", "RB"] * 20,
        "season": [2024] * 100,
        "week": list(range(1, 21)) * 5,
        "age": np.random.normal(28, 4, 100),
        "experience_years": np.random.normal(5, 3, 100),
        "games_played": np.random.normal(15, 2, 100),
        "passing_yds": np.random.normal(250, 50, 100),
        "passing_tds": np.random.normal(2, 1, 100),
        "qb_rating": np.random.normal(95, 15, 100),
        "rushing_yds": np.random.normal(50, 20, 100),
        "rushing_tds": np.random.normal(0.5, 0.3, 100),
        "receiving_yds": np.random.normal(60, 25, 100),
        "receiving_tds": np.random.normal(0.4, 0.2, 100),
        "receptions": np.random.normal(5, 2, 100),
        "home_advantage": np.random.choice([0, 1], 100),
        "rest_days": np.random.normal(7, 2, 100),
        "team_offensive_epa": np.random.normal(0.1, 0.05, 100),
        "opponent_defensive_epa": np.random.normal(-0.1, 0.05, 100)
    })
    
    print("🚀 Training Player Performance Prediction Models...")
    predictor.train_all_models(sample_data)
    
    print("
✅ Player Performance Models Trained Successfully!")
    print("YOLO Mode: Production-ready gradient boosting models for NFL player prediction!")
    
    # Test prediction
    test_features = pd.DataFrame({
        "season": [2024],
        "week": [10],
        "age": [29],
        "experience_years": [6],
        "games_played": [8],
        "home_advantage": [1],
        "rest_days": [7],
        "team_offensive_epa": [0.15],
        "opponent_defensive_epa": [-0.05]
    })
    
    qb_prediction = predictor.predict_player_performance(test_features, "QB", "fantasy_points")
    if len(qb_prediction) > 0:
        print(f"Sample QB fantasy points prediction: {qb_prediction[0]:.1f}")
    
    rb_prediction = predictor.predict_player_performance(test_features, "RB", "fantasy_points")
    if len(rb_prediction) > 0:
        print(f"Sample RB fantasy points prediction: {rb_prediction[0]:.1f}")
    
    predictor.save_models()
    print("Models saved to disk!")
        return model.predict(X_scaled)
