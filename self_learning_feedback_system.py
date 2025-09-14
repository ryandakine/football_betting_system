#!/usr/bin/env python3
"""
Self-Learning Feedback System for MLB Betting
=============================================
A comprehensive system that learns from successes and failures to continuously improve predictions.
This system tracks every prediction, outcome, and uses the feedback to strengthen successful patterns
and weaken unsuccessful ones.
"""

import asyncio
import json
import logging
import pickle
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class RealTimeOutcomeTracker:
    """Tracks game outcomes in real-time and provides immediate feedback"""

    def __init__(self):
        self.active_games = {}  # game_id -> game data
        self.completed_games = {}  # game_id -> outcome data
        self.outcome_callbacks = []  # Functions to call when outcomes are determined

    def register_game(self, game_id: str, game_data: dict):
        """Register a game for outcome tracking"""
        self.active_games[game_id] = {
            'data': game_data,
            'predictions': [],
            'registered_at': datetime.now(),
            'status': 'scheduled'
        }

    def add_prediction(self, game_id: str, prediction_data: dict):
        """Add a prediction for outcome tracking"""
        if game_id in self.active_games:
            self.active_games[game_id]['predictions'].append({
                'data': prediction_data,
                'timestamp': datetime.now()
            })

    def update_game_status(self, game_id: str, status: str, score_data: dict = None):
        """Update game status and trigger outcome processing"""
        if game_id in self.active_games:
            self.active_games[game_id]['status'] = status

            if status == 'completed' and score_data:
                self._process_game_outcome(game_id, score_data)

    def _process_game_outcome(self, game_id: str, score_data: dict):
        """Process game outcome and notify predictions"""
        if game_id not in self.active_games:
            return

        game_info = self.active_games[game_id]
        actual_winner = self._determine_winner(score_data)

        # Move to completed games
        self.completed_games[game_id] = {
            'game_data': game_info,
            'actual_winner': actual_winner,
            'score_data': score_data,
            'completed_at': datetime.now()
        }

        # Process predictions for this game
        for prediction in game_info['predictions']:
            self._evaluate_prediction(prediction, actual_winner)

        # Remove from active games
        del self.active_games[game_id]

        # Trigger callbacks
        self._trigger_outcome_callbacks(game_id, actual_winner)

    def _determine_winner(self, score_data: dict) -> str:
        """Determine the actual winner from score data"""
        home_score = score_data.get('home_score', 0)
        away_score = score_data.get('away_score', 0)

        if home_score > away_score:
            return score_data.get('home_team', 'home')
        elif away_score > home_score:
            return score_data.get('away_team', 'away')
        else:
            return 'push'  # Tie

    def _evaluate_prediction(self, prediction: dict, actual_winner: str):
        """Evaluate a prediction against actual outcome"""
        pred_data = prediction['data']
        predicted_winner = pred_data.get('predicted_team', pred_data.get('predicted_winner'))

        was_correct = predicted_winner == actual_winner
        stake = pred_data.get('stake', 0)
        odds = pred_data.get('odds', 1)

        if was_correct:
            profit = stake * (odds - 1)  # Win profit
        else:
            profit = -stake  # Loss

        # Store evaluation results
        prediction['evaluation'] = {
            'actual_winner': actual_winner,
            'was_correct': was_correct,
            'profit': profit,
            'roi': (profit / stake) if stake > 0 else 0,
            'evaluated_at': datetime.now()
        }

    def add_outcome_callback(self, callback_func):
        """Add a callback function for outcome notifications"""
        self.outcome_callbacks.append(callback_func)

    def _trigger_outcome_callbacks(self, game_id: str, actual_winner: str):
        """Trigger all registered outcome callbacks"""
        for callback in self.outcome_callbacks:
            try:
                callback(game_id, actual_winner)
            except Exception as e:
                logger.error(f"Error in outcome callback: {e}")

    def get_game_predictions(self, game_id: str) -> list:
        """Get all predictions for a specific game"""
        if game_id in self.active_games:
            return self.active_games[game_id]['predictions']
        elif game_id in self.completed_games:
            return self.completed_games[game_id]['game_data']['predictions']
        return []

    def get_prediction_accuracy(self, predictor_name: str = None, days: int = 30) -> dict:
        """Get prediction accuracy statistics"""
        cutoff_date = datetime.now() - timedelta(days=days)
        relevant_predictions = []

        # Collect predictions from completed games
        for game_data in self.completed_games.values():
            if game_data['completed_at'] > cutoff_date:
                for prediction in game_data['game_data']['predictions']:
                    if predictor_name is None or prediction['data'].get('predictor') == predictor_name:
                        if 'evaluation' in prediction:
                            relevant_predictions.append(prediction)

        if not relevant_predictions:
            return {'total': 0, 'correct': 0, 'accuracy': 0.0}

        total = len(relevant_predictions)
        correct = sum(1 for p in relevant_predictions if p['evaluation']['was_correct'])

        return {
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0.0,
            'period_days': days
        }


class AdaptiveConfidenceScorer:
    """Adapts confidence scores based on historical performance"""

    def __init__(self):
        self.performance_history = {}
        self.adjustment_factors = {}
        self.learning_rate = 0.1

    def update_confidence_model(self, predictor_name: str, actual_confidence: float, was_correct: bool):
        """Update confidence adjustment model for a predictor"""
        if predictor_name not in self.performance_history:
            self.performance_history[predictor_name] = []
            self.adjustment_factors[predictor_name] = 1.0

        # Store performance data
        self.performance_history[predictor_name].append({
            'confidence': actual_confidence,
            'correct': was_correct,
            'timestamp': datetime.now()
        })

        # Keep only recent history (last 100 predictions)
        if len(self.performance_history[predictor_name]) > 100:
            self.performance_history[predictor_name] = self.performance_history[predictor_name][-100:]

        # Update adjustment factor
        self._recalculate_adjustment_factor(predictor_name)

    def _recalculate_adjustment_factor(self, predictor_name: str):
        """Recalculate confidence adjustment factor based on recent performance"""
        history = self.performance_history[predictor_name]
        if len(history) < 10:
            return

        # Calculate calibration: how well confidence matches actual probability
        confidence_bins = {}
        for item in history:
            conf_bin = round(item['confidence'] * 10) / 10  # Round to nearest 0.1
            if conf_bin not in confidence_bins:
                confidence_bins[conf_bin] = {'total': 0, 'correct': 0}
            confidence_bins[conf_bin]['total'] += 1
            confidence_bins[conf_bin]['correct'] += 1 if item['correct'] else 0

        # Calculate adjustment factor
        total_error = 0
        total_samples = 0

        for conf_level, stats in confidence_bins.items():
            if stats['total'] >= 3:  # Minimum samples for reliability
                actual_prob = stats['correct'] / stats['total']
                expected_prob = conf_level
                error = abs(actual_prob - expected_prob)
                total_error += error * stats['total']
                total_samples += stats['total']

        if total_samples > 0:
            avg_error = total_error / total_samples
            # Adjust factor to reduce error
            current_factor = self.adjustment_factors[predictor_name]
            if avg_error > 0.1:  # If poorly calibrated
                adjustment = 1 - (avg_error * self.learning_rate)
                self.adjustment_factors[predictor_name] = max(0.5, min(2.0, current_factor * adjustment))

    def adjust_confidence(self, predictor_name: str, raw_confidence: float) -> float:
        """Adjust raw confidence score based on historical calibration"""
        if predictor_name not in self.adjustment_factors:
            return raw_confidence

        adjusted = raw_confidence * self.adjustment_factors[predictor_name]
        return max(0.1, min(0.95, adjusted))  # Clamp to reasonable range

    def get_predictor_stats(self, predictor_name: str) -> dict:
        """Get performance statistics for a predictor"""
        if predictor_name not in self.performance_history:
            return {'samples': 0, 'accuracy': 0.0, 'adjustment_factor': 1.0}

        history = self.performance_history[predictor_name]
        correct = sum(1 for item in history if item['correct'])
        accuracy = correct / len(history) if history else 0.0

        return {
            'samples': len(history),
            'accuracy': accuracy,
            'adjustment_factor': self.adjustment_factors.get(predictor_name, 1.0),
            'avg_confidence': sum(item['confidence'] for item in history) / len(history) if history else 0.0
        }


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
class LearningPattern:
    """A pattern that the system has identified as predictive."""

    pattern_id: str
    pattern_type: str  # 'feature_combination', 'odds_range', 'team_pattern', etc.
    pattern_description: str
    confidence_threshold: float
    success_rate: float
    sample_size: int
    last_updated: str
    is_active: bool = True
    strength_multiplier: float = 1.0


@dataclass
class ModelPerformance:
    """Performance metrics for a specific model or strategy."""

    model_name: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    total_profit: float
    total_roi: float
    avg_confidence: float
    last_30_days_accuracy: float
    last_30_days_profit: float
    pattern_effectiveness: dict[str, float]


class SelfLearningFeedbackSystem:
    """
    Core self-learning system that tracks predictions and outcomes,
    identifies patterns, and continuously improves the betting strategy.
    """

    def __init__(self, db_path: str = "data/learning_system.db"):
        self.db_path = db_path
        self.predictions: list[PredictionRecord] = []
        self.learning_patterns: list[LearningPattern] = []
        self.model_performance: dict[str, ModelPerformance] = {}

        # Enhanced learning parameters
        self.min_sample_size = 10
        self.confidence_threshold = 0.6
        self.pattern_strength_decay = 0.95
        self.learning_rate = 0.1

        # New enhancement parameters
        self.outcome_tracking_enabled = True
        self.real_time_learning = True
        self.adaptive_confidence = True
        self.cross_validation_enabled = True
        self.performance_memory_days = 90  # Remember performance for 90 days

        # Initialize database
        self._init_database()
        self._load_existing_data()

        # Initialize real-time outcome tracker
        self.outcome_tracker = RealTimeOutcomeTracker()
        self.confidence_adapter = AdaptiveConfidenceScorer()

        logger.info("🧠 Enhanced Self-Learning Feedback System initialized")

    def _init_database(self):
        """Initialize the learning database with all necessary tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Predictions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                game_id TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                predicted_winner TEXT NOT NULL,
                confidence REAL NOT NULL,
                stake_amount REAL NOT NULL,
                odds REAL NOT NULL,
                expected_value REAL NOT NULL,
                features TEXT NOT NULL,
                model_name TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                actual_winner TEXT,
                was_correct INTEGER,
                actual_profit REAL,
                actual_roi REAL,
                learning_weight REAL DEFAULT 1.0,
                pattern_strength REAL DEFAULT 1.0
            )
        """
        )

        # Learning patterns table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                pattern_description TEXT NOT NULL,
                confidence_threshold REAL NOT NULL,
                success_rate REAL NOT NULL,
                sample_size INTEGER NOT NULL,
                last_updated TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                strength_multiplier REAL DEFAULT 1.0
            )
        """
        )

        # Model performance table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_performance (
                model_name TEXT PRIMARY KEY,
                total_predictions INTEGER NOT NULL,
                correct_predictions INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                total_profit REAL NOT NULL,
                total_roi REAL NOT NULL,
                avg_confidence REAL NOT NULL,
                last_30_days_accuracy REAL NOT NULL,
                last_30_days_profit REAL NOT NULL,
                pattern_effectiveness TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """
        )

        # Create indexes for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_correct ON predictions(was_correct)"
        )

        conn.commit()
        conn.close()

    def _load_existing_data(self):
        """Load existing predictions and patterns from database."""
        conn = sqlite3.connect(self.db_path)

        # Load predictions
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        for _, row in df.iterrows():
            prediction = PredictionRecord(
                prediction_id=row["prediction_id"],
                timestamp=row["timestamp"],
                game_id=row["game_id"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                predicted_winner=row["predicted_winner"],
                confidence=row["confidence"],
                stake_amount=row["stake_amount"],
                odds=row["odds"],
                expected_value=row["expected_value"],
                features=json.loads(row["features"]),
                model_name=row["model_name"],
                strategy_type=row["strategy_type"],
                actual_winner=row["actual_winner"],
                was_correct=(
                    bool(row["was_correct"]) if row["was_correct"] is not None else None
                ),
                actual_profit=row["actual_profit"],
                actual_roi=row["actual_roi"],
                learning_weight=row["learning_weight"],
                pattern_strength=row["pattern_strength"],
            )
            self.predictions.append(prediction)

        # Load learning patterns
        df_patterns = pd.read_sql_query(
            "SELECT * FROM learning_patterns WHERE is_active = 1", conn
        )
        for _, row in df_patterns.iterrows():
            pattern = LearningPattern(
                pattern_id=row["pattern_id"],
                pattern_type=row["pattern_type"],
                pattern_description=row["pattern_description"],
                confidence_threshold=row["confidence_threshold"],
                success_rate=row["success_rate"],
                sample_size=row["sample_size"],
                last_updated=row["last_updated"],
                is_active=bool(row["is_active"]),
                strength_multiplier=row["strength_multiplier"],
            )
            self.learning_patterns.append(pattern)

        conn.close()
        logger.info(
            f"📊 Loaded {len(self.predictions)} predictions and {len(self.learning_patterns)} patterns"
        )

    async def record_prediction(self, prediction: PredictionRecord):
        """Record a new prediction for future learning."""
        self.predictions.append(prediction)

        # Save to database
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    prediction.prediction_id,
                    prediction.timestamp,
                    prediction.game_id,
                    prediction.home_team,
                    prediction.away_team,
                    prediction.predicted_winner,
                    prediction.confidence,
                    prediction.stake_amount,
                    prediction.odds,
                    prediction.expected_value,
                    json.dumps(prediction.features),
                    prediction.model_name,
                    prediction.strategy_type,
                    prediction.actual_winner,
                    prediction.was_correct,
                    prediction.actual_profit,
                    prediction.actual_roi,
                    prediction.learning_weight,
                    prediction.pattern_strength,
                ),
            )
            await conn.commit()

        logger.info(
            f"📝 Recorded prediction {prediction.prediction_id} for {prediction.home_team} vs {prediction.away_team}"
        )

    async def update_prediction_outcome(
        self, prediction_id: str, actual_winner: str, actual_profit: float
    ):
        """Update a prediction with the actual outcome and trigger learning."""
        # Find and update the prediction
        prediction_updated = False
        for pred in self.predictions:
            if pred.prediction_id == prediction_id:
                pred.actual_winner = actual_winner
                pred.was_correct = pred.predicted_winner == actual_winner
                pred.actual_profit = actual_profit
                pred.actual_roi = (
                    (actual_profit / pred.stake_amount) if pred.stake_amount > 0 else 0
                )
                prediction_updated = True

                # Enhanced learning: Update confidence adapter
                if self.adaptive_confidence:
                    self.confidence_adapter.update_confidence_model(
                        pred.model_name,
                        pred.confidence,
                        pred.was_correct
                    )

                # Enhanced learning: Update outcome tracker
                if self.outcome_tracking_enabled:
                    self.outcome_tracker.update_game_status(
                        pred.game_id,
                        'completed',
                        {
                            'home_team': pred.home_team,
                            'away_team': pred.away_team,
                            'actual_winner': actual_winner
                        }
                    )
                break

        if not prediction_updated:
            logger.warning(f"Prediction {prediction_id} not found for outcome update")
            return

        # Update database
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                UPDATE predictions
                SET actual_winner = ?, was_correct = ?, actual_profit = ?, actual_roi = ?
                WHERE prediction_id = ?
            """,
                (
                    actual_winner,
                    pred.was_correct,
                    actual_profit,
                    pred.actual_roi,
                    prediction_id,
                ),
            )
            await conn.commit()

        # Trigger real-time learning if enabled
        if self.real_time_learning and prediction_updated:
            await self._real_time_learning_adjustment(pred)

        logger.info(
            f"✅ Updated outcome for prediction {prediction_id}: {'WIN' if pred.was_correct else 'LOSS'} "
            f"(Enhanced learning applied)"
        )

    async def _real_time_learning_adjustment(self, prediction: PredictionRecord):
        """Apply real-time learning adjustments based on prediction outcome."""
        if not self.real_time_learning:
            return

        try:
            # Update learning weights for this prediction pattern
            weight_adjustment = 1.1 if prediction.was_correct else 0.9

            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(
                    """
                    UPDATE predictions
                    SET learning_weight = learning_weight * ?
                    WHERE prediction_id = ?
                    """,
                    (weight_adjustment, prediction.prediction_id)
                )
                await conn.commit()

            # Trigger immediate pattern analysis for similar predictions
            await self._analyze_similar_patterns(prediction)

            logger.info(f"🔄 Real-time learning: Adjusted weights for {prediction.prediction_id}")

        except Exception as e:
            logger.error(f"Error in real-time learning adjustment: {e}")

    async def _analyze_similar_patterns(self, prediction: PredictionRecord):
        """Analyze patterns similar to this prediction for immediate learning."""
        try:
            # Find similar recent predictions
            similar_predictions = []
            for pred in self.predictions[-50:]:  # Last 50 predictions
                if (pred.model_name == prediction.model_name and
                    pred.strategy_type == prediction.strategy_type and
                    pred.prediction_id != prediction.prediction_id):
                    similar_predictions.append(pred)

            if len(similar_predictions) < 3:
                return

            # Calculate pattern effectiveness
            correct_similar = sum(1 for p in similar_predictions if p.was_correct)
            pattern_accuracy = correct_similar / len(similar_predictions)

            # Update pattern strength
            pattern_key = f"{prediction.model_name}_{prediction.strategy_type}"
            strength_multiplier = 1.05 if prediction.was_correct else 0.95

            # Store pattern adjustment for future predictions
            if hasattr(self, 'pattern_adjustments'):
                self.pattern_adjustments[pattern_key] = \
                    self.pattern_adjustments.get(pattern_key, 1.0) * strength_multiplier

            logger.info(f"🎯 Pattern analysis: {pattern_key} accuracy {pattern_accuracy:.1%}")

        except Exception as e:
            logger.error(f"Error in similar pattern analysis: {e}")

    def get_adaptive_confidence(self, model_name: str, raw_confidence: float) -> float:
        """Get adaptively adjusted confidence score."""
        if not self.adaptive_confidence:
            return raw_confidence

        return self.confidence_adapter.adjust_confidence(model_name, raw_confidence)

    def get_enhanced_learning_insights(self) -> dict:
        """Get comprehensive learning insights including new enhancements."""
        base_insights = self.get_learning_insights()

        # Add enhanced insights
        enhanced_insights = {
            'outcome_tracking': {
                'active_games': len(self.outcome_tracker.active_games),
                'completed_games': len(self.outcome_tracker.completed_games),
                'total_predictions_tracked': sum(
                    len(game_data['predictions'])
                    for game_data in self.outcome_tracker.active_games.values()
                )
            },
            'adaptive_confidence': {
                predictor: self.confidence_adapter.get_predictor_stats(predictor)
                for predictor in self.confidence_adapter.performance_history.keys()
            },
            'real_time_learning': {
                'enabled': self.real_time_learning,
                'outcome_tracking': self.outcome_tracking_enabled,
                'adaptive_confidence': self.adaptive_confidence
            },
            'performance_memory_days': self.performance_memory_days
        }

        return {**base_insights, **enhanced_insights}

    async def analyze_and_learn(self):
        """Analyze recent predictions and update learning patterns."""
        logger.info("🧠 Starting learning analysis...")

        # Get recent predictions (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        recent_predictions = [
            p for p in self.predictions if p.timestamp >= thirty_days_ago
        ]

        if len(recent_predictions) < self.min_sample_size:
            logger.info(
                f"Not enough recent predictions ({len(recent_predictions)}) for learning analysis"
            )
            return

        # Analyze feature patterns
        await self._analyze_feature_patterns(recent_predictions)

        # Analyze odds patterns
        await self._analyze_odds_patterns(recent_predictions)

        # Analyze team-specific patterns
        await self._analyze_team_patterns(recent_predictions)

        # Update model performance
        await self._update_model_performance()

        # Apply learning adjustments
        await self._apply_learning_adjustments()

        logger.info("🎓 Learning analysis complete")

    async def _analyze_feature_patterns(self, predictions: list[PredictionRecord]):
        """Analyze which feature combinations lead to successful predictions."""
        successful_predictions = [p for p in predictions if p.was_correct]
        failed_predictions = [p for p in predictions if p.was_correct is False]

        if len(successful_predictions) < 5 or len(failed_predictions) < 5:
            return

        # Extract common features
        all_features = set()
        for pred in predictions:
            all_features.update(pred.features.keys())

        # Analyze each feature combination
        for feature_name in all_features:
            if feature_name in ["timestamp", "game_id"]:  # Skip non-predictive features
                continue

            # Check if this feature is predictive
            feature_success_rate = self._calculate_feature_success_rate(
                predictions, feature_name
            )

            if feature_success_rate > 0.6:  # Feature is predictive
                pattern_id = (
                    f"feature_{feature_name}_{datetime.now().strftime('%Y%m%d')}"
                )

                # Check if pattern already exists
                existing_pattern = next(
                    (p for p in self.learning_patterns if p.pattern_id == pattern_id),
                    None,
                )

                if existing_pattern:
                    # Update existing pattern
                    existing_pattern.success_rate = feature_success_rate
                    existing_pattern.sample_size = len(predictions)
                    existing_pattern.last_updated = datetime.now().isoformat()
                    existing_pattern.strength_multiplier = min(
                        2.0, existing_pattern.strength_multiplier * 1.1
                    )
                else:
                    # Create new pattern
                    new_pattern = LearningPattern(
                        pattern_id=pattern_id,
                        pattern_type="feature_combination",
                        pattern_description=f"High success rate when {feature_name} is present",
                        confidence_threshold=0.6,
                        success_rate=feature_success_rate,
                        sample_size=len(predictions),
                        last_updated=datetime.now().isoformat(),
                        strength_multiplier=1.2,
                    )
                    self.learning_patterns.append(new_pattern)

                    # Save to database
                    async with aiosqlite.connect(self.db_path) as conn:
                        await conn.execute(
                            """
                            INSERT INTO learning_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                new_pattern.pattern_id,
                                new_pattern.pattern_type,
                                new_pattern.pattern_description,
                                new_pattern.confidence_threshold,
                                new_pattern.success_rate,
                                new_pattern.sample_size,
                                new_pattern.last_updated,
                                new_pattern.is_active,
                                new_pattern.strength_multiplier,
                            ),
                        )
                        await conn.commit()

    async def _analyze_odds_patterns(self, predictions: list[PredictionRecord]):
        """Analyze which odds ranges lead to successful predictions."""
        # Group predictions by odds ranges
        odds_ranges = {
            "heavy_favorite": (1.0, 1.5),
            "favorite": (1.5, 2.0),
            "slight_favorite": (2.0, 2.5),
            "even": (2.5, 3.5),
            "underdog": (3.5, 5.0),
            "heavy_underdog": (5.0, 10.0),
        }

        for range_name, (min_odds, max_odds) in odds_ranges.items():
            range_predictions = [
                p for p in predictions if min_odds <= p.odds <= max_odds
            ]

            if len(range_predictions) >= 5:
                success_rate = sum(1 for p in range_predictions if p.was_correct) / len(
                    range_predictions
                )

                if success_rate > 0.55:  # Odds range is predictive
                    pattern_id = (
                        f"odds_{range_name}_{datetime.now().strftime('%Y%m%d')}"
                    )

                    # Update or create pattern
                    existing_pattern = next(
                        (
                            p
                            for p in self.learning_patterns
                            if p.pattern_id == pattern_id
                        ),
                        None,
                    )

                    if existing_pattern:
                        existing_pattern.success_rate = success_rate
                        existing_pattern.sample_size = len(range_predictions)
                        existing_pattern.last_updated = datetime.now().isoformat()
                    else:
                        new_pattern = LearningPattern(
                            pattern_id=pattern_id,
                            pattern_type="odds_range",
                            pattern_description=f"High success rate for {range_name} odds ({min_odds}-{max_odds})",
                            confidence_threshold=0.55,
                            success_rate=success_rate,
                            sample_size=len(range_predictions),
                            last_updated=datetime.now().isoformat(),
                            strength_multiplier=1.1,
                        )
                        self.learning_patterns.append(new_pattern)

    async def _analyze_team_patterns(self, predictions: list[PredictionRecord]):
        """Analyze which teams are consistently over/under-valued."""
        team_performance = {}

        for pred in predictions:
            if pred.was_correct is None:
                continue

            for team in [pred.home_team, pred.away_team]:
                if team not in team_performance:
                    team_performance[team] = {"wins": 0, "total": 0}

                team_performance[team]["total"] += 1
                if pred.was_correct and pred.predicted_winner == team:
                    team_performance[team]["wins"] += 1

        # Identify teams with consistent patterns
        for team, stats in team_performance.items():
            if stats["total"] >= 5:  # Minimum sample size
                success_rate = stats["wins"] / stats["total"]

                if success_rate > 0.65:  # Team is consistently undervalued
                    pattern_id = f"team_{team}_{datetime.now().strftime('%Y%m%d')}"

                    new_pattern = LearningPattern(
                        pattern_id=pattern_id,
                        pattern_type="team_pattern",
                        pattern_description=f"{team} is consistently undervalued (success rate: {success_rate:.2%})",
                        confidence_threshold=0.6,
                        success_rate=success_rate,
                        sample_size=stats["total"],
                        last_updated=datetime.now().isoformat(),
                        strength_multiplier=1.3,
                    )
                    self.learning_patterns.append(new_pattern)

    async def _update_model_performance(self):
        """Update performance metrics for each model/strategy."""
        models = {p.model_name for p in self.predictions}

        for model_name in models:
            model_predictions = [
                p for p in self.predictions if p.model_name == model_name
            ]

            if len(model_predictions) == 0:
                continue

            # Calculate overall performance
            total_predictions = len(model_predictions)
            correct_predictions = sum(1 for p in model_predictions if p.was_correct)
            accuracy = (
                correct_predictions / total_predictions if total_predictions > 0 else 0
            )
            total_profit = sum(p.actual_profit or 0 for p in model_predictions)
            total_roi = (
                total_profit / sum(p.stake_amount for p in model_predictions)
                if sum(p.stake_amount for p in model_predictions) > 0
                else 0
            )
            avg_confidence = np.mean([p.confidence for p in model_predictions])

            # Calculate last 30 days performance
            thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
            recent_predictions = [
                p for p in model_predictions if p.timestamp >= thirty_days_ago
            ]

            if recent_predictions:
                recent_correct = sum(1 for p in recent_predictions if p.was_correct)
                last_30_days_accuracy = recent_correct / len(recent_predictions)
                last_30_days_profit = sum(
                    p.actual_profit or 0 for p in recent_predictions
                )
            else:
                last_30_days_accuracy = 0
                last_30_days_profit = 0

            # Calculate pattern effectiveness
            pattern_effectiveness = {}
            for pattern in self.learning_patterns:
                if pattern.is_active:
                    pattern_predictions = [
                        p
                        for p in model_predictions
                        if self._prediction_matches_pattern(p, pattern)
                    ]
                    if pattern_predictions:
                        pattern_success = sum(
                            1 for p in pattern_predictions if p.was_correct
                        )
                        pattern_effectiveness[pattern.pattern_id] = (
                            pattern_success / len(pattern_predictions)
                        )

            performance = ModelPerformance(
                model_name=model_name,
                total_predictions=total_predictions,
                correct_predictions=correct_predictions,
                accuracy=accuracy,
                total_profit=total_profit,
                total_roi=total_roi,
                avg_confidence=avg_confidence,
                last_30_days_accuracy=last_30_days_accuracy,
                last_30_days_profit=last_30_days_profit,
                pattern_effectiveness=pattern_effectiveness,
            )

            self.model_performance[model_name] = performance

    async def _apply_learning_adjustments(self):
        """Apply learning adjustments to improve future predictions."""
        # Deactivate patterns that are no longer effective
        for pattern in self.learning_patterns:
            if pattern.sample_size >= 20 and pattern.success_rate < 0.5:
                pattern.is_active = False
                pattern.strength_multiplier *= self.pattern_strength_decay
                logger.info(
                    f"🔽 Deactivated ineffective pattern: {pattern.pattern_description}"
                )

        # Strengthen successful patterns
        for pattern in self.learning_patterns:
            if pattern.is_active and pattern.success_rate > 0.6:
                pattern.strength_multiplier = min(
                    2.0, pattern.strength_multiplier * 1.05
                )

        # Update learning weights for predictions
        for pred in self.predictions:
            if pred.was_correct is not None:
                # Increase weight for successful predictions
                if pred.was_correct:
                    pred.learning_weight = min(2.0, pred.learning_weight * 1.1)
                else:
                    pred.learning_weight = max(0.5, pred.learning_weight * 0.9)

    def _calculate_feature_success_rate(
        self, predictions: list[PredictionRecord], feature_name: str
    ) -> float:
        """Calculate success rate for predictions that have a specific feature."""
        feature_predictions = [p for p in predictions if feature_name in p.features]

        if len(feature_predictions) < 3:
            return 0.0

        successful = sum(1 for p in feature_predictions if p.was_correct)
        return successful / len(feature_predictions)

    def _prediction_matches_pattern(
        self, prediction: PredictionRecord, pattern: LearningPattern
    ) -> bool:
        """Check if a prediction matches a learning pattern."""
        if pattern.pattern_type == "feature_combination":
            # Check if prediction has the feature mentioned in pattern description
            feature_name = pattern.pattern_description.split("when ")[-1].split(" is")[
                0
            ]
            return feature_name in prediction.features

        elif pattern.pattern_type == "odds_range":
            # Check if prediction odds fall in the specified range
            odds_range = pattern.pattern_description.split("(")[-1].split(")")[0]
            min_odds, max_odds = map(float, odds_range.split("-"))
            return min_odds <= prediction.odds <= max_odds

        elif pattern.pattern_type == "team_pattern":
            # Check if prediction involves the team mentioned in pattern
            team_name = pattern.pattern_description.split(" ")[0]
            return team_name in [prediction.home_team, prediction.away_team]

        return False

    def get_learning_insights(self) -> dict[str, Any]:
        """Get insights from the learning system."""
        insights = {
            "total_predictions": len(self.predictions),
            "active_patterns": len([p for p in self.learning_patterns if p.is_active]),
            "model_performance": {},
            "top_patterns": [],
            "recent_accuracy": 0.0,
            "learning_recommendations": [],
        }

        # Model performance
        for model_name, performance in self.model_performance.items():
            insights["model_performance"][model_name] = {
                "accuracy": performance.accuracy,
                "total_profit": performance.total_profit,
                "last_30_days_accuracy": performance.last_30_days_accuracy,
                "last_30_days_profit": performance.last_30_days_profit,
            }

        # Top patterns
        active_patterns = [p for p in self.learning_patterns if p.is_active]
        top_patterns = sorted(
            active_patterns,
            key=lambda p: p.success_rate * p.strength_multiplier,
            reverse=True,
        )[:5]

        for pattern in top_patterns:
            insights["top_patterns"].append(
                {
                    "description": pattern.pattern_description,
                    "success_rate": pattern.success_rate,
                    "strength": pattern.strength_multiplier,
                    "sample_size": pattern.sample_size,
                }
            )

        # Recent accuracy
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        recent_predictions = [
            p
            for p in self.predictions
            if p.timestamp >= thirty_days_ago and p.was_correct is not None
        ]

        if recent_predictions:
            insights["recent_accuracy"] = sum(
                1 for p in recent_predictions if p.was_correct
            ) / len(recent_predictions)

        # Learning recommendations
        if insights["recent_accuracy"] < 0.5:
            insights["learning_recommendations"].append(
                "Consider reducing confidence thresholds"
            )

        if len(insights["top_patterns"]) > 0:
            best_pattern = insights["top_patterns"][0]
            if best_pattern["success_rate"] > 0.7:
                insights["learning_recommendations"].append(
                    f"Strengthen focus on: {best_pattern['description']}"
                )

        return insights

    async def generate_enhanced_prediction(
        self, game_data: dict[str, Any], base_confidence: float, model_name: str
    ) -> tuple[float, dict[str, Any]]:
        """Generate an enhanced prediction using learned patterns."""
        enhanced_confidence = base_confidence
        applied_patterns = []

        # Apply learning patterns
        for pattern in self.learning_patterns:
            if not pattern.is_active:
                continue

            # Check if this game matches the pattern
            if self._game_matches_pattern(game_data, pattern):
                # Boost confidence based on pattern strength
                confidence_boost = (
                    (pattern.success_rate - 0.5)
                    * pattern.strength_multiplier
                    * self.learning_rate
                )
                enhanced_confidence += confidence_boost

                applied_patterns.append(
                    {
                        "pattern_id": pattern.pattern_id,
                        "description": pattern.pattern_description,
                        "success_rate": pattern.success_rate,
                        "confidence_boost": confidence_boost,
                    }
                )

        # Cap confidence at 0.95
        enhanced_confidence = min(0.95, max(0.05, enhanced_confidence))

        return enhanced_confidence, {
            "base_confidence": base_confidence,
            "enhanced_confidence": enhanced_confidence,
            "applied_patterns": applied_patterns,
            "learning_boost": enhanced_confidence - base_confidence,
        }

    def _game_matches_pattern(
        self, game_data: dict[str, Any], pattern: LearningPattern
    ) -> bool:
        """Check if a game matches a learning pattern."""
        # This is a simplified version - you would implement more sophisticated matching
        # based on your actual game data structure

        if pattern.pattern_type == "team_pattern":
            team_name = pattern.pattern_description.split(" ")[0]
            return team_name in [
                game_data.get("home_team", ""),
                game_data.get("away_team", ""),
            ]

        elif pattern.pattern_type == "odds_range":
            odds = game_data.get("odds", 0)
            odds_range = pattern.pattern_description.split("(")[-1].split(")")[0]
            min_odds, max_odds = map(float, odds_range.split("-"))
            return min_odds <= odds <= max_odds

        return False


# Example usage and integration
async def main():
    """Example of how to use the self-learning feedback system."""
    learning_system = SelfLearningFeedbackSystem()

    # Example: Record a prediction
    prediction = PredictionRecord(
        prediction_id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now().isoformat(),
        game_id="game_123",
        home_team="Yankees",
        away_team="Red Sox",
        predicted_winner="Yankees",
        confidence=0.75,
        stake_amount=100.0,
        odds=1.85,
        expected_value=0.12,
        features={"home_advantage": True, "recent_form": "good", "weather": "clear"},
        model_name="ensemble_model",
        strategy_type="value_betting",
    )

    await learning_system.record_prediction(prediction)

    # Later, update with outcome
    await learning_system.update_prediction_outcome(
        prediction.prediction_id, "Yankees", 85.0
    )

    # Run learning analysis
    await learning_system.analyze_and_learn()

    # Get insights
    insights = learning_system.get_learning_insights()
    print("Learning Insights:", json.dumps(insights, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
