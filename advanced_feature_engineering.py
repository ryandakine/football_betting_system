"""
Advanced Feature Engineering Module for Football Betting System
YOLO Mode Implementation - Professional ML Feature Pipeline

Transforms raw football data into advanced ML features:
- EPA (Expected Points Added) calculations
- DVOA (Defense-adjusted Value Over Average)
- Advanced efficiency metrics
- Situational performance indicators
- Temporal trend features
- Interaction and composite features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Professional feature engineering pipeline for football analytics.
    YOLO Mode: Comprehensive, cutting-edge feature engineering.
    """
    
    def __init__(self):
        self.scalers = {}
        self.pca_models = {}
        self.feature_cache = {}
        
        # EPA calculation parameters (based on NFL research)
        self.epa_params = {
            "pass_completion": 0.25,
            "pass_incompletion": -0.15,
            "pass_touchdown": 6.0,
            "pass_interception": -3.0,
            "rush_success": 0.05,
            "rush_touchdown": 6.0,
            "rush_fumble": -2.0,
            "field_goal_good": 3.0,
            "field_goal_miss": -1.0,
            "extra_point_good": 1.0,
            "extra_point_miss": -1.0,
            "two_point_good": 2.0,
            "two_point_miss": -2.0,
            "safety": 2.0,
            "defensive_touchdown": -6.0,
            "turnover_recovery": 1.5
        }
        
        # DVOA calculation weights
        self.dvoa_weights = {
            "epa_weight": 0.6,
            "success_rate_weight": 0.3,
            "explosive_play_weight": 0.1
        }
    
    def calculate_epa(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Expected Points Added (EPA) for plays.
        EPA measures how much a play contributes to scoring.
        """
        logger.info("Calculating EPA for play data")
        
        df = df.copy()
        
        # Initialize EPA column
        df["epa"] = 0.0
        
        # Passing plays
        pass_mask = df["play_type"] == "pass"
        df.loc[pass_mask & (df["completion"] == 1), "epa"] += self.epa_params["pass_completion"]
        df.loc[pass_mask & (df["completion"] == 0), "epa"] += self.epa_params["pass_incompletion"]
        df.loc[pass_mask & (df["touchdown"] == 1), "epa"] += self.epa_params["pass_touchdown"]
        df.loc[pass_mask & (df["interception"] == 1), "epa"] += self.epa_params["pass_interception"]
        
        # Rushing plays
        rush_mask = df["play_type"] == "rush"
        df.loc[rush_mask, "epa"] += df.loc[rush_mask, "yards_gained"] * self.epa_params["rush_success"]
        df.loc[rush_mask & (df["touchdown"] == 1), "epa"] += self.epa_params["rush_touchdown"]
        if "fumble" in df.columns:
            df.loc[rush_mask & (df["fumble"] == 1), "epa"] += self.epa_params.get("rush_fumble", -2.0)
        
        # Kicking plays
        fg_mask = df["play_type"] == "field_goal"
        df.loc[fg_mask & (df["fg_made"] == 1), "epa"] += self.epa_params["field_goal_good"]
        df.loc[fg_mask & (df["fg_made"] == 0), "epa"] += self.epa_params["field_goal_miss"]
        # Kicking plays
        fg_mask = df["play_type"] == "field_goal"
        df.loc[fg_mask & (df["fg_made"] == 1), "epa"] += self.epa_params["field_goal_good"]
        df.loc[fg_mask & (df["fg_made"] == 0), "epa"] += self.epa_params["field_goal_miss"]
        
        # Extra points (check if column exists)
        if "xp_made" in df.columns:
            ep_mask = df["play_type"] == "extra_point"
            df.loc[ep_mask & (df["xp_made"] == 1), "epa"] += self.epa_params.get("extra_point_good", 1.0)
            df.loc[ep_mask & (df["xp_made"] == 0), "epa"] += self.epa_params.get("extra_point_miss", -1.0)
        
        # Two point conversions (check if column exists)
        if "two_pt_made" in df.columns:
            two_pt_mask = df["play_type"] == "two_point"
            df.loc[two_pt_mask & (df["two_pt_made"] == 1), "epa"] += self.epa_params.get("two_point_good", 2.0)
        # Defensive plays (check if columns exist)
        if "safety" in df.columns:
            df.loc[df["safety"] == 1, "epa"] += self.epa_params.get("safety", 2.0)
        if "defensive_td" in df.columns:
            df.loc[df["defensive_td"] == 1, "epa"] += self.epa_params.get("defensive_touchdown", -6.0)
        if "turnover_recovery" in df.columns:
            df.loc[df["turnover_recovery"] == 1, "epa"] += self.epa_params.get("turnover_recovery", 1.5)
        df["epa"] = df["epa"] * self._calculate_field_position_multiplier(df)
        df["epa"] = df["epa"] * self._calculate_situation_multiplier(df)
        
        logger.info(f"Calculated EPA for {len(df)} plays")
        return df
    
    def _calculate_field_position_multiplier(self, df: pd.DataFrame) -> pd.Series:
        """Calculate EPA multiplier based on field position"""
        multipliers = []
        
        for _, row in df.iterrows():
            yardline = row.get("yardline", 50)
            
            if yardline >= 80:  # Red zone
                multiplier = 1.3
            elif yardline >= 60:  # Own territory
                multiplier = 1.1
            elif yardline <= 20:  # Opponent red zone
                multiplier = 1.2
            else:  # Midfield
                multiplier = 1.0
            
            multipliers.append(multiplier)
        
        return pd.Series(multipliers, index=df.index)
    
    def _calculate_situation_multiplier(self, df: pd.DataFrame) -> pd.Series:
        """Calculate EPA multiplier based on game situation"""
        multipliers = []
        
        for _, row in df.iterrows():
            down = row.get("down", 1)
            yards_to_go = row.get("yards_to_go", 10)
            time_remaining = row.get("time_remaining", 1800)  # seconds
            
            multiplier = 1.0
            
            # Down multipliers
            if down == 4:
                multiplier *= 1.5  # 4th down is critical
            elif down == 3 and yards_to_go > 5:
                multiplier *= 1.2  # 3rd and long
            
            # Time multipliers
            if time_remaining < 300:  # Last 5 minutes
                multiplier *= 1.3
            elif time_remaining < 900:  # Last 15 minutes
                multiplier *= 1.1
            
            multipliers.append(multiplier)
        
        return pd.Series(multipliers, index=df.index)
    
    def calculate_dvoa(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Defense-adjusted Value Over Average (DVOA).
        DVOA measures team performance adjusted for opponent strength.
        """
        logger.info("Calculating DVOA metrics")
        
        df = df.copy()
        
        # Calculate raw EPA per play
        df["epa_per_play"] = df.groupby("team")["epa"].transform("mean")
        
        # Calculate success rate
        df["success_rate"] = df.groupby("team")["success"].transform("mean")
        
        # Calculate explosive play rate
        df["explosive_rate"] = df.groupby("team")["explosive_play"].transform("mean")
        
        # Calculate league averages
        league_avg_epa = df["epa"].mean()
        league_avg_success = df["success"].mean()
        league_avg_explosive = df["explosive_play"].mean()
        
        # Calculate DVOA components
        df["epa_dvoa"] = (df["epa_per_play"] - league_avg_epa) / abs(league_avg_epa)
        df["success_dvoa"] = (df["success_rate"] - league_avg_success) / abs(league_avg_success)
        df["explosive_dvoa"] = (df["explosive_rate"] - league_avg_explosive) / abs(league_avg_explosive)
        
        # Combine into overall DVOA
        df["dvoa"] = (
            df["epa_dvoa"] * self.dvoa_weights["epa_weight"] +
            df["success_dvoa"] * self.dvoa_weights["success_rate_weight"] +
            df["explosive_dvoa"] * self.dvoa_weights["explosive_play_weight"]
        )
        
        logger.info("DVOA calculations completed")
        return df
    
    def calculate_advanced_team_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced team performance metrics.
        """
        logger.info("Calculating advanced team metrics")
        
        # Efficiency metrics
        df["offensive_efficiency"] = df["points_scored"] / df["total_offensive_yds"].replace(0, 1)
        df["defensive_efficiency"] = df["points_allowed"] / df["total_defensive_yds"].replace(0, 1)
        
        # Turnover differential (check if columns exist)
        if all(col in df.columns for col in ["interceptions", "fumbles_lost", "interceptions_forced", "fumbles_recovered"]):
            df["turnover_differential"] = (df["interceptions"] + df["fumbles_lost"]) - (df["interceptions_forced"] + df["fumbles_recovered"])
        else:
            df["turnover_differential"] = 0  # Default value
            df["turnover_differential"] = 0  # Default value
        
        # Red zone efficiency
        if all(col in df.columns for col in ["red_zone_tds", "red_zone_opportunities"]):
            df["red_zone_efficiency"] = df["red_zone_tds"] / df["red_zone_opportunities"].replace(0, 1)
        else:
            df["red_zone_efficiency"] = 0.5  # League average
        
        # Third down conversion rate
        if all(col in df.columns for col in ["third_down_conversions", "third_down_attempts"]):
            df["third_down_rate"] = df["third_down_conversions"] / df["third_down_attempts"].replace(0, 1)
        else:
            df["third_down_rate"] = 0.4  # League average
        
        # Explosive play rate (plays > 20 yards)
        if "explosive_plays" in df.columns:
            df["explosive_play_rate"] = df["explosive_plays"] / df["total_plays"].replace(0, 1)
        else:
            df["explosive_play_rate"] = 0.1  # League average
        
        # Yards per play
        df["yards_per_play"] = df["total_offensive_yds"] / df["total_plays"].replace(0, 1)
        
        # Points per drive
        if "drives" in df.columns:
            df["points_per_drive"] = df["points_scored"] / df["drives"].replace(0, 1)
        else:
            df["points_per_drive"] = df["points_scored"] / 12  # Approximate drives per game
        logger.info("Advanced team metrics calculated")
        return df
    
    def calculate_player_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced player performance metrics.
        """
        logger.info("Calculating advanced player metrics")
        
        df = df.copy()
        
        # Quarterback metrics
        qb_mask = df["position"] == "QB"
        df.loc[qb_mask, "qb_rating"] = self._calculate_qb_rating(df[qb_mask])
        df.loc[qb_mask, "passer_rating"] = self._calculate_passer_rating(df[qb_mask])
        df.loc[qb_mask, "completion_percentage"] = df["pass_completions"] / df["pass_attempts"].replace(0, 1)
        df.loc[qb_mask, "yards_per_attempt"] = df["passing_yds"] / df["pass_attempts"].replace(0, 1)
        df.loc[qb_mask, "td_to_int_ratio"] = df["passing_tds"] / df["interceptions"].replace(0, 1)
        
        # Running back metrics
        rb_mask = df["position"] == "RB"
        df.loc[rb_mask, "yards_per_carry"] = df["rushing_yds"] / df["rushing_attempts"].replace(0, 1)
        df.loc[rb_mask, "yards_after_contact"] = df["rush_yac"] / df["rushing_attempts"].replace(0, 1)
        df.loc[rb_mask, "broken_tackle_rate"] = df["broken_tackles"] / df["rushing_attempts"].replace(0, 1)
        
        # Wide receiver metrics
        wr_mask = df["position"].isin(["WR", "TE"])
        df.loc[wr_mask, "yards_per_reception"] = df["receiving_yds"] / df["receptions"].replace(0, 1)
        df.loc[wr_mask, "catch_percentage"] = df["receptions"] / df["targets"].replace(0, 1)
        df.loc[wr_mask, "yards_after_catch"] = df["yac"] / df["receptions"].replace(0, 1)
        
        # Defensive metrics
        def_mask = df["position"].isin(["DL", "LB", "DB"])
        df.loc[def_mask, "tackles_per_game"] = df["tackles"] / df["games_played"].replace(0, 1)
        df.loc[def_mask, "qb_hits_per_game"] = df["qb_hits"] / df["games_played"].replace(0, 1)
        
        logger.info("Advanced player metrics calculated")
        return df
    
    def _calculate_qb_rating(self, qb_df: pd.DataFrame) -> pd.Series:
        """Calculate NFL QB Rating"""
        completions = qb_df["pass_completions"]
        attempts = qb_df["pass_attempts"]
        tds = qb_df["passing_tds"]
        ints = qb_df["interceptions"]
        yards = qb_df["passing_yds"]
        
        completion_pct = (completions / attempts.replace(0, 1) - 0.3) * 5
        yards_per_attempt = (yards / attempts.replace(0, 1) - 3) * 0.25
        td_pct = (tds / attempts.replace(0, 1)) * 20
        int_pct = 2.375 - (ints / attempts.replace(0, 1) * 25)
        
        # Cap values between 0 and 2.375
        for series in [completion_pct, yards_per_attempt, td_pct, int_pct]:
            series.clip(0, 2.375, inplace=True)
        
        qb_rating = (completion_pct + yards_per_attempt + td_pct + int_pct) / 6 * 100
        return qb_rating
    
    def _calculate_passer_rating(self, qb_df: pd.DataFrame) -> pd.Series:
        """Calculate alternative passer rating"""
        completions = qb_df["pass_completions"]
        attempts = qb_df["pass_attempts"]
        tds = qb_df["passing_tds"]
        ints = qb_df["interceptions"]
        yards = qb_df["passing_yds"]
        
        # More modern passer rating formula
        completion_bonus = (completions / attempts.replace(0, 1)) * 2.5
        yards_bonus = (yards / attempts.replace(0, 1)) * 0.05
        td_bonus = (tds / attempts.replace(0, 1)) * 5
        int_penalty = (ints / attempts.replace(0, 1)) * 5
        
        passer_rating = completion_bonus + yards_bonus + td_bonus - int_penalty
        return passer_rating.clip(0, 10)
    
    def create_temporal_features(self, df: pd.DataFrame, window_sizes: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """
        Create temporal features showing performance trends over time.
        """
        logger.info("Creating temporal features")
        
        df = df.copy()
        df = df.sort_values(["team", "season", "week"])
        
        for window in window_sizes:
            # Rolling averages
            df[f"epa_last_{window}"] = df.groupby("team")["epa"].transform(lambda x: x.rolling(window).mean())
            df[f"success_rate_last_{window}"] = df.groupby("team")["success_rate"].transform(lambda x: x.rolling(window).mean())
            
            # Rolling standard deviations (volatility)
            df[f"epa_volatility_{window}"] = df.groupby("team")["epa"].transform(lambda x: x.rolling(window).std())
            
            # Trend indicators
            df[f"epa_trend_{window}"] = df.groupby("team")["epa"].transform(lambda x: x.rolling(window).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0))
        
        logger.info("Temporal features created")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different metrics.
        """
        logger.info("Creating interaction features")
        
        df = df.copy()
        
        # Offensive efficiency interactions
        df["epa_x_success_rate"] = df["epa_per_play"] * df["success_rate"]
        df["epa_x_explosive_rate"] = df["epa_per_play"] * df["explosive_play_rate"]
        
        # Defensive interactions
        df["def_epa_x_turnovers"] = df["defensive_epa"] * df["turnover_differential"]
        
        # Situational efficiency
        df["red_zone_x_third_down"] = df["red_zone_efficiency"] * df["third_down_rate"]
        
        # Player-team interactions (for individual player predictions)
        if "team_epa" in df.columns and "player_epa" in df.columns:
            df["player_team_synergy"] = df["player_epa"] * df["team_epa"]
            df["player_relative_contribution"] = df["player_epa"] / df["team_epa"].replace(0, 1)
        
        logger.info("Interaction features created")
        return df
    
    def normalize_features(self, df: pd.DataFrame, features: List[str], method: str = "standard") -> pd.DataFrame:
        """
        Normalize features for ML models.
        """
        logger.info(f"Normalizing {len(features)} features using {method} scaling")
        
        df = df.copy()
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        scaled_values = scaler.fit_transform(df[features])
        
        # Create new column names
        scaled_columns = [f"{col}_scaled" for col in features]
        df[scaled_columns] = scaled_values
        
        # Store scaler for future use
        self.scalers[f"{method}_scaler"] = scaler
        
        logger.info("Feature normalization completed")
        return df
    
    def apply_pca(self, df: pd.DataFrame, features: List[str], n_components: int = 5, prefix: str = "pca") -> pd.DataFrame:
        """
        Apply PCA dimensionality reduction to features.
        """
        logger.info(f"Applying PCA with {n_components} components to {len(features)} features")
        
        df = df.copy()
        
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(df[features])
        
        # Create PCA column names
        pca_columns = [f"{prefix}_{i+1}" for i in range(n_components)]
        df[pca_columns] = pca_features
        
        # Store PCA model
        self.pca_models[f"{prefix}_pca"] = pca
        
        # Log explained variance
        explained_variance = pca.explained_variance_ratio_
        logger.info(f"PCA explained variance: {explained_variance}")
        logger.info(f"Cumulative explained variance: {np.cumsum(explained_variance)}")
        
        return df
    
    def create_prediction_features(self, player_df: pd.DataFrame, team_df: pd.DataFrame, 
                                 game_df: pd.DataFrame, target_season: int, target_week: int) -> pd.DataFrame:
        """
        Create comprehensive feature set for game outcome prediction.
        """
        logger.info(f"Creating prediction features for {target_season} Week {target_week}")
        
        # Get historical data up to target week
        historical_cutoff = (target_season - 1) * 1000 + target_week  # Simple season-week encoding
        
        # Aggregate team season stats
        team_season_stats = self._calculate_team_season_averages(team_df, target_season, target_week)
        
        # Aggregate player season stats
        player_season_stats = self._calculate_player_season_averages(player_df, target_season, target_week)
        
        # Create game features
        game_features = []
        
        for _, game in game_df.iterrows():
            if game["season"] != target_season or game["week"] != target_week:
                continue
            
            home_team = game["home_team"]
            away_team = game["away_team"]
            
            # Get team stats
            home_stats = team_season_stats.get(home_team, {})
            away_stats = team_season_stats.get(away_team, {})
            
            # Calculate differentials
            feature_row = {
                "game_id": game["game_id"],
                "home_team": home_team,
                "away_team": away_team,
                "season": target_season,
                "week": target_week,
                
                # Offensive differentials
                "epa_diff": home_stats.get("epa_per_play", 0) - away_stats.get("epa_per_play", 0),
                "success_rate_diff": home_stats.get("success_rate", 0) - away_stats.get("success_rate", 0),
                "yards_per_play_diff": home_stats.get("yards_per_play", 0) - away_stats.get("yards_per_play", 0),
                
                # Defensive differentials
                "def_epa_diff": away_stats.get("defensive_epa", 0) - home_stats.get("defensive_epa", 0),
                "turnover_diff": home_stats.get("turnover_differential", 0) - away_stats.get("turnover_differential", 0),
                
                # Efficiency differentials
                "red_zone_diff": home_stats.get("red_zone_efficiency", 0) - away_stats.get("red_zone_efficiency", 0),
                "third_down_diff": home_stats.get("third_down_rate", 0) - away_stats.get("third_down_rate", 0),
                
                # Game situation factors
                "home_advantage": 1,  # Home team advantage
                "weather_factor": self._calculate_weather_factor(game),
                "rest_advantage": self._calculate_rest_advantage(game, game_df),
                
                # Target variable (to be filled later)
                "home_win": None
            }
            
            game_features.append(feature_row)
        
        features_df = pd.DataFrame(game_features)
        logger.info(f"Created prediction features for {len(features_df)} games")
        
        return features_df
    
    def _calculate_team_season_averages(self, team_df: pd.DataFrame, season: int, week: int) -> Dict[str, Dict[str, float]]:
        """Calculate season averages for teams up to specified week"""
        season_data = team_df[(team_df["season"] == season) & (team_df["week"] <= week)]
        
        team_averages = {}
        for team in season_data["team_name"].unique():
            team_data = season_data[season_data["team_name"] == team]
            
            averages = {
                "epa_per_play": team_data["offensive_epa"].mean(),
                "success_rate": team_data["success_rate"].mean(),
                "yards_per_play": team_data["yards_per_play"].mean(),
                "defensive_epa": team_data["defensive_epa"].mean(),
                "turnover_differential": team_data["turnover_differential"].mean(),
                "red_zone_efficiency": team_data["red_zone_efficiency"].mean(),
                "third_down_rate": team_data["third_down_rate"].mean()
            }
            
            # Fill NaN values with league averages
            for key, value in averages.items():
                if pd.isna(value):
                    averages[key] = season_data[key].mean()
            
            team_averages[team] = averages
        
        return team_averages
    
    def _calculate_player_season_averages(self, player_df: pd.DataFrame, season: int, week: int) -> Dict[str, Dict[str, float]]:
        """Calculate season averages for key players"""
        season_data = player_df[(player_df["season"] == season) & (player_df["week"] <= week)]
        
        player_averages = {}
        # Focus on starting QBs and key skill players
        starting_players = season_data[season_data["games_started"] > 0]
        
