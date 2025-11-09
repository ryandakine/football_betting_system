#!/usr/bin/env python3
"""
Model 12: Prop Bet Specialist Model
Predicts player and team props with high accuracy
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from .super_intelligence import BaseNCAAModel


class PropBetSpecialistModel(BaseNCAAModel):
    """
    Model 12: Prop Bet Specialist

    Specialized predictions for:
    - Player props (passing yards, rushing yards, TDs, receptions)
    - Team props (first half total, team total, total TDs)
    - Game props (time of possession, turnovers, field goals)

    Uses separate models for each prop type for maximum accuracy.
    """

    def __init__(self):
        super().__init__("Prop Bet Specialist", "regression")
        # Player prop models
        self.passing_yards_model = None
        self.rushing_yards_model = None
        self.receiving_yards_model = None
        self.touchdowns_model = None

        # Team prop models
        self.team_total_model = None
        self.first_half_total_model = None
        self.quarter_total_model = None

        # Game prop models
        self.turnover_model = None
        self.field_goal_model = None
        self.sack_model = None

    def extract_prop_features(
        self,
        team_stats: Dict[str, Any],
        opponent_stats: Dict[str, Any],
        player_stats: Optional[Dict[str, Any]] = None,
        weather: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Extract features specific to prop betting.

        Args:
            team_stats: Team offensive/defensive stats
            opponent_stats: Opponent's defensive/offensive stats
            player_stats: Individual player stats (if predicting player props)
            weather: Weather conditions

        Returns:
            Dictionary of prop-specific features
        """
        features = {}

        # Team offensive features
        features['team_ppg'] = float(team_stats.get('points_per_game', 30.0))
        features['team_ypg'] = float(team_stats.get('yards_per_game', 400.0))
        features['team_pass_ypg'] = float(team_stats.get('pass_yards_per_game', 250.0))
        features['team_rush_ypg'] = float(team_stats.get('rush_yards_per_game', 150.0))
        features['team_td_per_game'] = float(team_stats.get('tds_per_game', 3.5))
        features['team_turnovers_per_game'] = float(team_stats.get('turnovers_per_game', 1.5))

        # Opponent defensive features
        features['opp_ppg_allowed'] = float(opponent_stats.get('points_allowed_per_game', 25.0))
        features['opp_ypg_allowed'] = float(opponent_stats.get('yards_allowed_per_game', 380.0))
        features['opp_pass_ypg_allowed'] = float(opponent_stats.get('pass_yards_allowed', 230.0))
        features['opp_rush_ypg_allowed'] = float(opponent_stats.get('rush_yards_allowed', 150.0))
        features['opp_sacks_per_game'] = float(opponent_stats.get('sacks_per_game', 2.5))
        features['opp_turnovers_forced'] = float(opponent_stats.get('turnovers_forced', 1.8))

        # Pace/tempo features (critical for props)
        features['team_plays_per_game'] = float(team_stats.get('plays_per_game', 70.0))
        features['opp_plays_per_game'] = float(opponent_stats.get('plays_per_game', 70.0))
        features['combined_pace'] = features['team_plays_per_game'] + features['opp_plays_per_game']

        # Efficiency features
        features['team_yards_per_play'] = features['team_ypg'] / max(features['team_plays_per_game'], 1)
        features['opp_yards_per_play_allowed'] = features['opp_ypg_allowed'] / max(features['opp_plays_per_game'], 1)

        # Red zone features
        features['team_rz_td_pct'] = float(team_stats.get('red_zone_td_percentage', 0.55))
        features['opp_rz_td_pct_allowed'] = float(opponent_stats.get('red_zone_td_pct_allowed', 0.50))

        # Third down features
        features['team_3rd_down_pct'] = float(team_stats.get('third_down_conversion_pct', 0.40))
        features['opp_3rd_down_pct_allowed'] = float(opponent_stats.get('third_down_pct_allowed', 0.38))

        # Player-specific features (if available)
        if player_stats:
            features['player_avg'] = float(player_stats.get('season_average', 0))
            features['player_recent_avg'] = float(player_stats.get('last_3_games_avg', 0))
            features['player_vs_opponent'] = float(player_stats.get('vs_opponent_avg', 0))
            features['player_usage_rate'] = float(player_stats.get('usage_rate', 0.25))
            features['player_target_share'] = float(player_stats.get('target_share', 0.20))

        # Weather features (important for passing/kicking props)
        if weather:
            features['wind_speed'] = float(weather.get('wind_speed', 5.0))
            features['temperature'] = float(weather.get('temperature', 60.0)) / 100.0  # Normalize
            features['precipitation'] = 1.0 if weather.get('precipitation') else 0.0
            features['is_dome'] = 1.0 if weather.get('dome') else 0.0
        else:
            features['wind_speed'] = 5.0
            features['temperature'] = 0.6
            features['precipitation'] = 0.0
            features['is_dome'] = 0.0

        # Derived matchup features
        features['pass_matchup_advantage'] = features['team_pass_ypg'] - features['opp_pass_ypg_allowed']
        features['rush_matchup_advantage'] = features['team_rush_ypg'] - features['opp_rush_ypg_allowed']
        features['pace_advantage'] = features['team_plays_per_game'] - features['opp_plays_per_game']

        return features

    def train_passing_yards_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model to predict QB passing yards."""
        self.passing_yards_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
        self.passing_yards_model.fit(X, y)

    def train_rushing_yards_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model to predict RB rushing yards."""
        self.rushing_yards_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        self.rushing_yards_model.fit(X, y)

    def train_team_total_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model to predict team total points."""
        # Use ensemble of RF + GB
        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        gb = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )

        rf.fit(X, y)
        gb.fit(X, y)

        # Store both for ensemble prediction
        self.team_total_model = {'rf': rf, 'gb': gb}

    def train_touchdown_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model to predict total touchdowns."""
        self.touchdowns_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
        self.touchdowns_model.fit(X, y)

    def train_first_half_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model to predict first half total."""
        self.first_half_total_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.first_half_total_model.fit(X, y)

    def train(self, training_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Train all prop models.

        Args:
            training_data: Dict with keys like 'passing_yards', 'team_total', etc.
                          Each value is (X, y) tuple
        """
        if 'passing_yards' in training_data:
            X, y = training_data['passing_yards']
            self.train_passing_yards_model(X, y)

        if 'rushing_yards' in training_data:
            X, y = training_data['rushing_yards']
            self.train_rushing_yards_model(X, y)

        if 'team_total' in training_data:
            X, y = training_data['team_total']
            self.train_team_total_model(X, y)

        if 'touchdowns' in training_data:
            X, y = training_data['touchdowns']
            self.train_touchdown_model(X, y)

        if 'first_half' in training_data:
            X, y = training_data['first_half']
            self.train_first_half_model(X, y)

        self.is_trained = True

    def predict_passing_yards(self, X: np.ndarray) -> np.ndarray:
        """Predict QB passing yards."""
        if self.passing_yards_model is None:
            raise ValueError("Passing yards model not trained")
        return self.passing_yards_model.predict(X)

    def predict_rushing_yards(self, X: np.ndarray) -> np.ndarray:
        """Predict RB rushing yards."""
        if self.rushing_yards_model is None:
            raise ValueError("Rushing yards model not trained")
        return self.rushing_yards_model.predict(X)

    def predict_team_total(self, X: np.ndarray) -> np.ndarray:
        """Predict team total points (ensemble)."""
        if self.team_total_model is None:
            raise ValueError("Team total model not trained")

        rf_pred = self.team_total_model['rf'].predict(X)
        gb_pred = self.team_total_model['gb'].predict(X)

        # 50/50 ensemble
        return (rf_pred + gb_pred) / 2.0

    def predict_touchdowns(self, X: np.ndarray) -> np.ndarray:
        """Predict total touchdowns."""
        if self.touchdowns_model is None:
            raise ValueError("Touchdowns model not trained")
        return self.touchdowns_model.predict(X)

    def predict_first_half_total(self, X: np.ndarray) -> np.ndarray:
        """Predict first half total points."""
        if self.first_half_total_model is None:
            raise ValueError("First half model not trained")
        return self.first_half_total_model.predict(X)

    def predict_all_props(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict all available props at once.

        Returns:
            Dictionary with predictions for each prop type
        """
        predictions = {}

        if self.passing_yards_model:
            predictions['passing_yards'] = self.predict_passing_yards(X)

        if self.rushing_yards_model:
            predictions['rushing_yards'] = self.predict_rushing_yards(X)

        if self.team_total_model:
            predictions['team_total'] = self.predict_team_total(X)

        if self.touchdowns_model:
            predictions['touchdowns'] = self.predict_touchdowns(X)

        if self.first_half_total_model:
            predictions['first_half_total'] = self.predict_first_half_total(X)

        return predictions

    def get_prop_confidence(self, prediction: float, historical_variance: float) -> float:
        """
        Calculate confidence for a prop prediction.

        Args:
            prediction: The predicted value
            historical_variance: Historical variance of this prop

        Returns:
            Confidence score (0-1)
        """
        # Lower variance = higher confidence
        # Normalize variance to confidence score
        confidence = 1.0 / (1.0 + historical_variance / 10.0)
        return min(max(confidence, 0.3), 0.9)  # Clamp between 30-90%
