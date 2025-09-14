#!/usr/bin/env python3
"""
Self-Learning MLB Betting System
================================
A comprehensive system that learns from successes and failures to continuously improve predictions.
This system tracks every prediction, outcome, and uses the feedback to strengthen successful patterns
and weaken unsuccessful ones.

Features:
- Historical data training (2024 season + 2025 first half)
- Real-time learning from daily predictions
- Backtesting with 2024/2025 data
- Pattern recognition and feature importance
- Model retraining with new insights
- Performance tracking and optimization
"""

import json
import logging
import sqlite3
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

aiosqlite: Any
try:
    import aiosqlite  # type: ignore[no-redef]
except Exception:  # pragma: no cover
    aiosqlite = None
import joblib
import pandas as pd
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Record of a single prediction made by the system."""

    prediction_id: str
    timestamp: str
    game_id: str
    home_team: str
    away_team: str
    predicted_winner: str
    confidence: float
    stake_amount: float
    odds: float
    expected_value: float

    # Features used in prediction
    features: dict[str, Any]

    # Model/strategy that made the prediction
    model_name: str
    strategy_type: str

    # Outcome tracking (filled after game)
    actual_winner: str | None = None
    was_correct: bool | None = None
    actual_profit: float | None = None
    actual_roi: float | None = None

    # Learning metadata
    learning_weight: float = 1.0
    pattern_strength: float = 1.0


@dataclass
class LearningMetrics:
    """Metrics for tracking learning progress."""

    total_predictions: int
    correct_predictions: int
    accuracy: float
    total_profit: float
    total_invested: float
    roi: float
    model_performance: dict[str, float]
    feature_importance: dict[str, float]
    pattern_insights: list[str]


class SelfLearningSystem:
    """Main self-learning system for MLB betting predictions."""

    def __init__(self, db_path: str = "data/learning_system.db", model_dir: str = "models"):
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # Initialize database
        self._init_database()

        # Load or create models
        self.models = self._load_models()

        # Learning parameters
        self.retrain_threshold = 100  # Retrain after 100 new predictions
        self.min_confidence_threshold = 0.55
        self.learning_rate = 0.1

        logger.info("Self-Learning System initialized")

    def _init_database(self):
        """Initialize SQLite database for storing predictions and outcomes."""
        # Ensure the database directory exists
        db_parent = Path(self.db_path).parent
        try:
            if str(db_parent) and not db_parent.exists():
                db_parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create database directory '{db_parent}': {e}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Predictions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                timestamp TEXT,
                game_id TEXT,
                home_team TEXT,
                away_team TEXT,
                predicted_winner TEXT,
                confidence REAL,
                stake_amount REAL,
                odds REAL,
                expected_value REAL,
                features TEXT,
                model_name TEXT,
                strategy_type TEXT,
                actual_winner TEXT,
                was_correct INTEGER,
                actual_profit REAL,
                actual_roi REAL,
                learning_weight REAL,
                pattern_strength REAL
            )
        """
        )

        # Learning metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_metrics (
                date TEXT PRIMARY KEY,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                accuracy REAL,
                total_profit REAL,
                total_invested REAL,
                roi REAL,
                model_performance TEXT,
                feature_importance TEXT,
                pattern_insights TEXT
            )
        """
        )

        # Historical data table for backtesting
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_data (
                date TEXT,
                game_id TEXT,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                winner TEXT,
                features TEXT,
                PRIMARY KEY (date, game_id)
            )
        """
        )

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def _load_models(self) -> dict[str, Any]:
        """Load existing models or create new ones."""
        models = {}

        # Team outcome prediction model
        team_model_path = self.model_dir / "team_outcome_model.pkl"
        if team_model_path.exists():
            models["team_outcome"] = joblib.load(team_model_path)
            logger.info("Loaded existing team outcome model")
        else:
            models["team_outcome"] = RandomForestClassifier(n_estimators=200, random_state=42)
            logger.info("Created new team outcome model")

        # Value betting model
        value_model_path = self.model_dir / "value_betting_model.pkl"
        if value_model_path.exists():
            models["value_betting"] = joblib.load(value_model_path)
            logger.info("Loaded existing value betting model")
        else:
            models["value_betting"] = GradientBoostingClassifier(random_state=42)
            logger.info("Created new value betting model")

        return models

    def load_historical_data(self, data_paths: list[str]) -> pd.DataFrame:
        """Load historical MLB data for training."""
        all_data = []

        for path in data_paths:
            if Path(path).exists():
                try:
                    if path.endswith(".parquet"):
                        df = pl.read_parquet(path).to_pandas()
                    elif path.endswith(".csv"):
                        df = pd.read_csv(path)
                    else:
                        continue

                    all_data.append(df)
                    logger.info(f"Loaded {len(df)} records from {path}")
                except Exception as e:
                    logger.warning(f"Could not load {path}: {e}")

        if not all_data:
            logger.warning("No historical data found")
            return pd.DataFrame()

        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined {len(combined_data)} historical records")
        return combined_data

    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training."""
        # Basic features (you can expand this)
        feature_columns = [
            "home_team_win_pct",
            "away_team_win_pct",
            "home_team_runs_per_game",
            "away_team_runs_per_game",
            "home_team_era",
            "away_team_era",
            "home_team_ops",
            "away_team_ops",
            "home_team_war",
            "away_team_war",
            "home_odds",
            "away_odds",
            "home_implied_prob",
            "away_implied_prob",
            "edge_home",
            "edge_away",
            "sentiment_home",
            "sentiment_away",
            "weather_temp",
            "weather_wind_speed",
            "rest_days_home",
            "rest_days_away",
        ]

        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]

        if not available_features:
            logger.warning("No expected features found in data")
            return pd.DataFrame(), pd.Series()

        X = df[available_features].fillna(0)

        # Create target variable (home team wins)
        if "home_score" in df.columns and "away_score" in df.columns:
            y = (df["home_score"] > df["away_score"]).astype(int)
        elif "winner" in df.columns:
            y = (df["winner"] == df["home_team"]).astype(int)
        else:
            logger.warning("No target variable found")
            return pd.DataFrame(), pd.Series()

        logger.info(f"Prepared {len(X)} samples with {len(available_features)} features")
        return X, y

    def train_models(self, historical_data: pd.DataFrame, force_retrain: bool = False):
        """Train models on historical data."""
        if historical_data.empty:
            logger.warning("No historical data available for training")
            return

        X, y = self.prepare_features(historical_data)

        if X.empty or y.empty:
            logger.warning("Could not prepare features for training")
            return

        # Train team outcome model
        logger.info("Training team outcome model...")
        self.models["team_outcome"].fit(X, y)

        # Evaluate model
        y_pred = self.models["team_outcome"].predict(X)
        accuracy = accuracy_score(y, y_pred)
        logger.info(f"Team outcome model accuracy: {accuracy:.3f}")

        # Save model
        joblib.dump(self.models["team_outcome"], self.model_dir / "team_outcome_model.pkl")

        # Train value betting model (predicts if bet has positive expected value)
        logger.info("Training value betting model...")
        # Create value betting target (positive edge bets)
        if "edge_home" in X.columns and "edge_away" in X.columns:
            value_target = ((X["edge_home"] > 0.02) | (X["edge_away"] > 0.02)).astype(int)
            self.models["value_betting"].fit(X, value_target)

            y_pred_value = self.models["value_betting"].predict(X)
            value_accuracy = accuracy_score(value_target, y_pred_value)
            logger.info(f"Value betting model accuracy: {value_accuracy:.3f}")

            joblib.dump(self.models["value_betting"], self.model_dir / "value_betting_model.pkl")

        logger.info("Model training completed")

    def make_prediction(self, game_features: dict[str, Any]) -> dict[str, Any]:
        """Make a prediction for a single game."""
        # Convert features to DataFrame
        feature_df = pd.DataFrame([game_features])

        # Get team outcome prediction
        outcome_prob = self.models["team_outcome"].predict_proba(feature_df)[0]
        home_win_prob = outcome_prob[1] if len(outcome_prob) > 1 else 0.5

        # Get value betting prediction
        if "value_betting" in self.models:
            value_prob = self.models["value_betting"].predict_proba(feature_df)[0]
            has_value = value_prob[1] > 0.5 if len(value_prob) > 1 else False
        else:
            has_value = True  # Default to True if no value model

        # Calculate expected value
        home_odds = game_features.get("home_odds", 2.0)
        away_odds = game_features.get("away_odds", 2.0)

        home_implied_prob = 1 / home_odds
        away_implied_prob = 1 / away_odds

        home_edge = home_win_prob - home_implied_prob
        away_edge = (1 - home_win_prob) - away_implied_prob

        # Determine best bet
        if home_edge > away_edge and home_edge > 0.02:
            recommended_bet = "home"
            confidence = home_win_prob
            edge = home_edge
            odds = home_odds
        elif away_edge > 0.02:
            recommended_bet = "away"
            confidence = 1 - home_win_prob
            edge = away_edge
            odds = away_odds
        else:
            recommended_bet = "no_bet"
            confidence = 0.0
            edge = 0.0
            odds = 0.0

        return {
            "prediction_id": f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{game_features.get('game_id', 'unknown')}",
            "game_id": game_features.get("game_id", "unknown"),
            "home_team": game_features.get("home_team", ""),
            "away_team": game_features.get("away_team", ""),
            "predicted_winner": recommended_bet,
            "confidence": confidence,
            "edge": edge,
            "odds": odds,
            "has_value": has_value,
            "home_win_probability": home_win_prob,
            "away_win_probability": 1 - home_win_prob,
            "home_edge": home_edge,
            "away_edge": away_edge,
            "recommended_stake": (self._calculate_kelly_stake(edge, odds) if edge > 0 else 0),
        }

    def _calculate_kelly_stake(self, edge: float, odds: float) -> float:
        """Calculate Kelly Criterion stake size."""
        if edge <= 0 or odds <= 1:
            return 0

        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = probability of win, q = probability of loss
        b = odds - 1
        p = 1 / odds + edge  # True probability
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        # Limit to reasonable stake size (max 5% of bankroll)
        return max(0, min(kelly_fraction, 0.05))

    def record_prediction(self, prediction: dict[str, Any]):
        """Record a prediction in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions (
                prediction_id, timestamp, game_id, home_team, away_team,
                predicted_winner, confidence, stake_amount, odds, expected_value,
                features, model_name, strategy_type, learning_weight, pattern_strength
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                prediction["prediction_id"],
                datetime.now().isoformat(),
                prediction["game_id"],
                prediction["home_team"],
                prediction["away_team"],
                prediction["predicted_winner"],
                prediction["confidence"],
                prediction.get("recommended_stake", 0),
                prediction["odds"],
                prediction["edge"],
                json.dumps(prediction.get("features", {})),
                "self_learning_system",
                "ml_enhanced",
                prediction.get("learning_weight", 1.0),
                prediction.get("pattern_strength", 1.0),
            ),
        )

        conn.commit()
        conn.close()
        logger.info(f"Recorded prediction {prediction['prediction_id']}")

    def record_outcome(self, prediction_id: str, actual_winner: str, actual_profit: float):
        """Record the actual outcome of a prediction."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get original prediction
        cursor.execute(
            "SELECT predicted_winner, confidence FROM predictions WHERE prediction_id = ?",
            (prediction_id,),
        )
        result = cursor.fetchone()

        if result:
            predicted_winner, confidence = result
            was_correct = predicted_winner == actual_winner
            actual_roi = actual_profit / 100 if actual_profit != 0 else 0  # Assuming $100 base stake

            cursor.execute(
                """
                UPDATE predictions
                SET actual_winner = ?, was_correct = ?, actual_profit = ?, actual_roi = ?
                WHERE prediction_id = ?
            """,
                (actual_winner, was_correct, actual_profit, actual_roi, prediction_id),
            )

            conn.commit()
            logger.info(f"Recorded outcome for {prediction_id}: {'CORRECT' if was_correct else 'INCORRECT'}")

        conn.close()

    def analyze_performance(self) -> LearningMetrics:
        """Analyze system performance and extract learning insights."""
        conn = sqlite3.connect(self.db_path)

        # Get all predictions with outcomes
        df = pd.read_sql_query(
            """
            SELECT * FROM predictions
            WHERE actual_winner IS NOT NULL
        """,
            conn,
        )

        if df.empty:
            conn.close()
            return LearningMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, {}, {}, [])

        # Calculate basic metrics
        total_predictions = len(df)
        correct_predictions = df["was_correct"].sum()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        total_profit = df["actual_profit"].sum()
        total_invested = df["stake_amount"].sum()
        roi = (total_profit / total_invested * 100) if total_invested > 0 else 0

        # Model performance by confidence level
        model_performance = {}
        for confidence_level in [0.5, 0.6, 0.7, 0.8, 0.9]:
            high_conf = df[df["confidence"] >= confidence_level]
            if len(high_conf) > 0:
                model_performance[f"accuracy_above_{confidence_level}"] = high_conf["was_correct"].mean()

        # Feature importance (if available)
        feature_importance: dict[str, list[Any]] = {}
        if "features" in df.columns:
            # Analyze which features correlate with success
            for idx, row in df.iterrows():
                try:
                    features = json.loads(row["features"])
                    for feature, value in features.items():
                        if feature not in feature_importance:
                            feature_importance[feature] = []
                        feature_importance[feature].append(value)
                except Exception:
                    continue

        # Pattern insights
        pattern_insights = []

        # High confidence predictions perform better
        high_conf = df[df["confidence"] >= 0.7]
        if len(high_conf) > 10:
            high_conf_accuracy = high_conf["was_correct"].mean()
            pattern_insights.append(f"High confidence predictions (≥70%) have {high_conf_accuracy:.1%} accuracy")

        # Positive edge predictions perform better
        positive_edge = df[df["expected_value"] > 0.02]
        if len(positive_edge) > 10:
            pos_edge_accuracy = positive_edge["was_correct"].mean()
            pattern_insights.append(f"Positive edge predictions (>2%) have {pos_edge_accuracy:.1%} accuracy")

        # Recent performance trend
        recent_df = df.tail(50)  # Last 50 predictions
        if len(recent_df) >= 20:
            recent_accuracy = recent_df["was_correct"].mean()
            pattern_insights.append(f"Recent 50 predictions: {recent_accuracy:.1%} accuracy")

        conn.close()

        # Summarize feature importance as average values to satisfy type expectations
        feature_importance_summary: dict[str, float] = {
            k: (float(sum(v)) / len(v) if len(v) > 0 else 0.0)
            for k, v in feature_importance.items()
        }

        return LearningMetrics(
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            total_profit=total_profit,
            total_invested=total_invested,
            roi=roi,
            model_performance=model_performance,
            feature_importance=feature_importance_summary,
            pattern_insights=pattern_insights,
        )

    def retrain_with_new_data(self):
        """Retrain models with new prediction data."""
        conn = sqlite3.connect(self.db_path)

        # Get all predictions with outcomes
        df = pd.read_sql_query(
            """
            SELECT * FROM predictions
            WHERE actual_winner IS NOT NULL
        """,
            conn,
        )

        if len(df) < self.retrain_threshold:
            logger.info(f"Only {len(df)} predictions available, need {self.retrain_threshold} for retraining")
            conn.close()
            return

        # Prepare features for retraining
        features_list = []
        targets = []

        for _, row in df.iterrows():
            try:
                features = json.loads(row["features"])
                # Add prediction metadata as features
                features["confidence"] = row["confidence"]
                features["expected_value"] = row["expected_value"]
                features["stake_amount"] = row["stake_amount"]

                features_list.append(features)
                targets.append(1 if row["was_correct"] else 0)
            except Exception:
                continue

        if len(features_list) < 50:
            logger.warning("Not enough valid feature data for retraining")
            conn.close()
            return

        # Convert to DataFrame
        retrain_df = pd.DataFrame(features_list)
        retrain_df["target"] = targets

        # Retrain models
        X, y = self.prepare_features(retrain_df)

        if not X.empty and not y.empty:
            logger.info("Retraining models with new data...")
            self.train_models(retrain_df, force_retrain=True)

            # Update learning metrics
            metrics = self.analyze_performance()
            self._save_learning_metrics(metrics)

        conn.close()

    def _save_learning_metrics(self, metrics: LearningMetrics):
        """Save learning metrics to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO learning_metrics (
                date, total_predictions, correct_predictions, accuracy,
                total_profit, total_invested, roi, model_performance,
                feature_importance, pattern_insights
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().strftime("%Y-%m-%d"),
                metrics.total_predictions,
                metrics.correct_predictions,
                metrics.accuracy,
                metrics.total_profit,
                metrics.total_invested,
                metrics.roi,
                json.dumps(metrics.model_performance),
                json.dumps(metrics.feature_importance),
                json.dumps(metrics.pattern_insights),
            ),
        )

        conn.commit()
        conn.close()

    def run_backtest(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Run backtest on historical data."""
        logger.info(f"Running backtest from {start_date} to {end_date}")

        conn = sqlite3.connect(self.db_path)

        # Get historical data
        df = pd.read_sql_query(
            """
            SELECT * FROM historical_data
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        """,
            conn,
            params=(start_date, end_date),
        )

        if df.empty:
            logger.warning("No historical data found for backtest period")
            conn.close()
            return {}

        # Simulate predictions for each game
        backtest_results = []
        total_profit = 0
        total_bets = 0
        correct_bets = 0

        for _, game in df.iterrows():
            try:
                features = json.loads(game["features"])
                features["game_id"] = game["game_id"]
                features["home_team"] = game["home_team"]
                features["away_team"] = game["away_team"]

                # Make prediction
                prediction = self.make_prediction(features)

                if prediction["predicted_winner"] != "no_bet":
                    total_bets += 1

                    # Determine actual winner
                    actual_winner = "home" if game["home_score"] > game["away_score"] else "away"

                    # Calculate profit/loss
                    if prediction["predicted_winner"] == actual_winner:
                        profit = prediction["recommended_stake"] * (prediction["odds"] - 1)
                        correct_bets += 1
                    else:
                        profit = -prediction["recommended_stake"]

                    total_profit += profit

                    backtest_results.append(
                        {
                            "date": game["date"],
                            "game_id": game["game_id"],
                            "prediction": prediction["predicted_winner"],
                            "actual": actual_winner,
                            "confidence": prediction["confidence"],
                            "stake": prediction["recommended_stake"],
                            "profit": profit,
                            "correct": prediction["predicted_winner"] == actual_winner,
                        }
                    )

            except Exception as e:
                logger.warning(f"Error processing game {game['game_id']}: {e}")
                continue

        conn.close()

        # Calculate backtest metrics
        accuracy = correct_bets / total_bets if total_bets > 0 else 0
        roi = (total_profit / sum(r["stake"] for r in backtest_results) * 100) if backtest_results else 0

        results = {
            "start_date": start_date,
            "end_date": end_date,
            "total_games": len(df),
            "total_bets": total_bets,
            "correct_bets": correct_bets,
            "accuracy": accuracy,
            "total_profit": total_profit,
            "roi": roi,
            "bet_details": backtest_results,
        }

        logger.info(f"Backtest completed: {accuracy:.1%} accuracy, {roi:.1f}% ROI")
        return results

    def get_learning_summary(self) -> dict[str, Any]:
        """Get a summary of the system's learning progress."""
        metrics = self.analyze_performance()

        # Get recent performance trend
        conn = sqlite3.connect(self.db_path)
        recent_df = pd.read_sql_query(
            """
            SELECT date, was_correct, actual_profit
            FROM predictions
            WHERE actual_winner IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 100
        """,
            conn,
        )
        conn.close()

        # Calculate rolling accuracy
        if len(recent_df) >= 20:
            recent_df["rolling_accuracy"] = recent_df["was_correct"].rolling(20).mean()
            recent_trend = recent_df["rolling_accuracy"].iloc[-1] - recent_df["rolling_accuracy"].iloc[0]
        else:
            recent_trend = 0

        return {
            "overall_metrics": {
                "total_predictions": metrics.total_predictions,
                "accuracy": f"{metrics.accuracy:.1%}",
                "total_profit": f"${metrics.total_profit:.2f}",
                "roi": f"{metrics.roi:.1f}%",
            },
            "recent_trend": f"{'Improving' if recent_trend > 0.01 else 'Declining' if recent_trend < -0.01 else 'Stable'}",
            "pattern_insights": metrics.pattern_insights,
            "model_performance": metrics.model_performance,
            "recommendations": self._generate_recommendations(metrics),
        }

    def _generate_recommendations(self, metrics: LearningMetrics) -> list[str]:
        """Generate recommendations for system improvement."""
        recommendations = []

        if metrics.accuracy < 0.55:
            recommendations.append("Consider reducing confidence threshold - current accuracy is below 55%")

        if metrics.roi < 0:
            recommendations.append("System is losing money - review feature engineering and model parameters")

        if metrics.total_predictions < 100:
            recommendations.append("Need more predictions to make reliable recommendations")

        # Check for overfitting
        if "accuracy_above_0.9" in metrics.model_performance:
            if metrics.model_performance["accuracy_above_0.9"] < 0.8:
                recommendations.append("High confidence predictions underperforming - possible overfitting")

        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Initialize the system
    learning_system = SelfLearningSystem()

    # Load historical data (2024 season + 2025 first half)
    historical_paths = [
        "data/historical_team_data.parquet",
        "data/2024_season_data.parquet",
        "data/2025_first_half_data.parquet",
    ]

    historical_data = learning_system.load_historical_data(historical_paths)

    if not historical_data.empty:
        # Train initial models
        learning_system.train_models(historical_data)

        # Run backtest on 2024 season
        backtest_results = learning_system.run_backtest("2024-03-28", "2024-10-01")
        print("2024 Season Backtest Results:")
        print(f"Accuracy: {backtest_results.get('accuracy', 0):.1%}")
        print(f"ROI: {backtest_results.get('roi', 0):.1f}%")
        print(f"Total Profit: ${backtest_results.get('total_profit', 0):.2f}")

    # Get learning summary
    summary = learning_system.get_learning_summary()
    print("\nLearning System Summary:")
    print(json.dumps(summary, indent=2))
