#!/usr/bin/env python3
"""
Feature Engineering for NCAA Super Intelligence System
Extracts 100+ features from game data, SP+, stats, and situational factors
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json


class NCAAFeatureEngineer:
    """
    Advanced feature engineering for NCAA football predictions
    Generates features for all prediction targets
    """

    def __init__(self, data_dir="data/football/historical/ncaaf"):
        self.data_dir = Path(data_dir)
        self.sp_ratings = {}
        self.team_stats = {}
        self.feature_cache = {}

    def load_season_data(self, year):
        """Load all data for a season"""
        # Load games
        games_file = self.data_dir / f"ncaaf_{year}_games.json"
        with open(games_file) as f:
            games = json.load(f)

        # Load SP+ ratings
        sp_file = self.data_dir / f"ncaaf_{year}_sp_ratings.json"
        if sp_file.exists():
            with open(sp_file) as f:
                sp_data = json.load(f)
                self.sp_ratings[year] = {t['team']: t for t in sp_data}

        # Load team stats
        stats_file = self.data_dir / f"ncaaf_{year}_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats_data = json.load(f)
                self.team_stats[year] = {s['team']: s for s in stats_data}

        return games

    def engineer_features(self, game, year):
        """
        Generate all features for a game
        Returns dict with 100+ features
        """
        features = {}

        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')

        # 1. SP+ FEATURES (20 features)
        features.update(self._sp_features(home_team, away_team, year))

        # 2. TEAM STATS FEATURES (30 features)
        features.update(self._stats_features(home_team, away_team, year))

        # 3. SITUATIONAL FEATURES (15 features)
        features.update(self._situational_features(game))

        # 4. MATCHUP FEATURES (15 features)
        features.update(self._matchup_features(home_team, away_team, game))

        # 5. TEMPORAL FEATURES (10 features)
        features.update(self._temporal_features(game))

        # 6. CONFERENCE FEATURES (10 features)
        features.update(self._conference_features(game))

        # 7. DERIVED FEATURES (20 features)
        features.update(self._derived_features(features))

        return features

    def _sp_features(self, home_team, away_team, year):
        """SP+ rating features"""
        features = {}

        sp_ratings = self.sp_ratings.get(year, {})
        home_sp = sp_ratings.get(home_team, {})
        away_sp = sp_ratings.get(away_team, {})

        # Overall ratings
        features['home_sp_rating'] = home_sp.get('rating', 0)
        features['away_sp_rating'] = away_sp.get('rating', 0)
        features['sp_rating_diff'] = features['home_sp_rating'] - features['away_sp_rating']

        # Offensive ratings
        features['home_sp_offense'] = home_sp.get('offense', {}).get('rating', 0)
        features['away_sp_offense'] = away_sp.get('offense', {}).get('rating', 0)
        features['sp_offense_diff'] = features['home_sp_offense'] - features['away_sp_offense']

        # Defensive ratings
        features['home_sp_defense'] = home_sp.get('defense', {}).get('rating', 0)
        features['away_sp_defense'] = away_sp.get('defense', {}).get('rating', 0)
        features['sp_defense_diff'] = features['home_sp_defense'] - features['away_sp_defense']

        # Special teams
        features['home_sp_special'] = home_sp.get('specialTeams', {}).get('rating', 0)
        features['away_sp_special'] = away_sp.get('specialTeams', {}).get('rating', 0)

        # Tempo
        features['home_sp_tempo'] = home_sp.get('tempo', 0)
        features['away_sp_tempo'] = away_sp.get('tempo', 0)
        features['sp_tempo_diff'] = features['home_sp_tempo'] - features['away_sp_tempo']

        # Success rates
        features['home_sp_success_rate'] = home_sp.get('offense', {}).get('successRate', 0.5)
        features['away_sp_success_rate'] = away_sp.get('offense', {}).get('successRate', 0.5)

        # Explosiveness
        features['home_sp_explosiveness'] = home_sp.get('offense', {}).get('explosiveness', 0)
        features['away_sp_explosiveness'] = away_sp.get('offense', {}).get('explosiveness', 0)

        # Passing/Rushing splits
        features['home_sp_passing'] = home_sp.get('offense', {}).get('passingDowns', {}).get('ppa', 0)
        features['away_sp_passing'] = away_sp.get('offense', {}).get('passingDowns', {}).get('ppa', 0)

        return features

    def _stats_features(self, home_team, away_team, year):
        """Team statistics features"""
        features = {}

        team_stats = self.team_stats.get(year, {})
        home_stats = team_stats.get(home_team, {})
        away_stats = team_stats.get(away_team, {})

        # Offensive stats
        features['home_ppg'] = home_stats.get('pointsPerGame', 0)
        features['away_ppg'] = away_stats.get('pointsPerGame', 0)
        features['ppg_diff'] = features['home_ppg'] - features['away_ppg']

        features['home_yards_per_game'] = home_stats.get('yardsPerGame', 0)
        features['away_yards_per_game'] = away_stats.get('yardsPerGame', 0)

        features['home_pass_yards_pg'] = home_stats.get('passingYardsPerGame', 0)
        features['away_pass_yards_pg'] = away_stats.get('passingYardsPerGame', 0)

        features['home_rush_yards_pg'] = home_stats.get('rushingYardsPerGame', 0)
        features['away_rush_yards_pg'] = away_stats.get('rushingYardsPerGame', 0)

        # Defensive stats
        features['home_points_allowed_pg'] = home_stats.get('pointsAllowedPerGame', 0)
        features['away_points_allowed_pg'] = away_stats.get('pointsAllowedPerGame', 0)

        features['home_yards_allowed_pg'] = home_stats.get('yardsAllowedPerGame', 0)
        features['away_yards_allowed_pg'] = away_stats.get('yardsAllowedPerGame', 0)

        # Efficiency
        features['home_3rd_down_pct'] = home_stats.get('thirdDownConversionPct', 0.4)
        features['away_3rd_down_pct'] = away_stats.get('thirdDownConversionPct', 0.4)

        features['home_redzone_pct'] = home_stats.get('redZoneConversionPct', 0.7)
        features['away_redzone_pct'] = away_stats.get('redZoneConversionPct', 0.7)

        # Turnovers
        features['home_turnovers_pg'] = home_stats.get('turnoversPerGame', 0)
        features['away_turnovers_pg'] = away_stats.get('turnoversPerGame', 0)
        features['home_takeaways_pg'] = home_stats.get('takeawaysPerGame', 0)
        features['away_takeaways_pg'] = away_stats.get('takeawaysPerGame', 0)

        # Turnover margin
        features['home_turnover_margin'] = features['home_takeaways_pg'] - features['home_turnovers_pg']
        features['away_turnover_margin'] = features['away_takeaways_pg'] - features['away_turnovers_pg']

        # Win percentage
        features['home_win_pct'] = home_stats.get('winPercentage', 0.5)
        features['away_win_pct'] = away_stats.get('winPercentage', 0.5)

        # Penalties
        features['home_penalties_pg'] = home_stats.get('penaltiesPerGame', 5)
        features['away_penalties_pg'] = away_stats.get('penaltiesPerGame', 5)

        return features

    def _situational_features(self, game):
        """Situational factors"""
        features = {}

        # Home field advantage
        features['is_home_game'] = 1
        features['is_neutral_site'] = 1 if game.get('neutral_site') else 0

        # Time/day
        game_date = game.get('start_date', '')
        if game_date:
            dt = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
            features['day_of_week'] = dt.weekday()
            features['hour_of_day'] = dt.hour
            features['is_primetime'] = 1 if 19 <= dt.hour <= 21 else 0  # 7-9 PM
            features['is_saturday'] = 1 if dt.weekday() == 5 else 0
            features['month'] = dt.month
            features['week_of_season'] = game.get('week', 0)
        else:
            features['day_of_week'] = 5  # Default Saturday
            features['hour_of_day'] = 12
            features['is_primetime'] = 0
            features['is_saturday'] = 1
            features['month'] = 10
            features['week_of_season'] = game.get('week', 0)

        # Weather (if available)
        features['temperature'] = game.get('weather', {}).get('temperature', 70)
        features['wind_speed'] = game.get('weather', {}).get('windSpeed', 0)
        features['is_dome'] = 1 if game.get('venue_type') == 'dome' else 0

        # Rivalry
        features['is_rivalry'] = self._detect_rivalry(game)

        # Conference game
        home_conf = game.get('home_conference', '')
        away_conf = game.get('away_conference', '')
        features['is_conference_game'] = 1 if home_conf == away_conf else 0
        features['is_division_game'] = 1 if game.get('is_division_game') else 0

        return features

    def _matchup_features(self, home_team, away_team, game):
        """Historical matchup features"""
        features = {}

        # Conference matchups
        home_conf = game.get('home_conference', '')
        away_conf = game.get('away_conference', '')

        features['is_sec_game'] = 1 if 'SEC' in [home_conf, away_conf] else 0
        features['is_big10_game'] = 1 if 'Big Ten' in [home_conf, away_conf] else 0
        features['is_big12_game'] = 1 if 'Big 12' in [home_conf, away_conf] else 0
        features['is_acc_game'] = 1 if 'ACC' in [home_conf, away_conf] else 0
        features['is_pac12_game'] = 1 if 'Pac-12' in [home_conf, away_conf] else 0

        # Power 5 vs Group of 5
        power5_conferences = ['SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12']
        features['home_is_power5'] = 1 if home_conf in power5_conferences else 0
        features['away_is_power5'] = 1 if away_conf in power5_conferences else 0
        features['is_power5_matchup'] = features['home_is_power5'] * features['away_is_power5']

        # Ranked matchup (if available)
        features['home_is_ranked'] = 1 if game.get('home_rank', 100) <= 25 else 0
        features['away_is_ranked'] = 1 if game.get('away_rank', 100) <= 25 else 0
        features['is_ranked_matchup'] = features['home_is_ranked'] * features['away_is_ranked']

        # Spread (if available - for derived features)
        features['betting_spread'] = game.get('spread', 0)
        features['betting_total'] = game.get('total', 50)

        return features

    def _temporal_features(self, game):
        """Time-based features"""
        features = {}

        week = game.get('week', 0)

        # Season progression
        features['week_number'] = week
        features['is_early_season'] = 1 if week <= 4 else 0
        features['is_mid_season'] = 1 if 5 <= week <= 10 else 0
        features['is_late_season'] = 1 if week >= 11 else 0

        # Rest days
        features['days_since_last_game'] = game.get('days_rest', 7)
        features['is_short_rest'] = 1 if features['days_since_last_game'] < 7 else 0

        # Bye week
        features['coming_off_bye'] = game.get('home_bye_last_week', 0)

        return features

    def _conference_features(self, game):
        """Conference-specific features"""
        features = {}

        home_conf = game.get('home_conference', 'Other')
        away_conf = game.get('away_conference', 'Other')

        # One-hot encode conferences
        conferences = ['SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12', 'Other']

        for conf in conferences:
            features[f'home_conf_{conf.replace(" ", "_").lower()}'] = 1 if home_conf == conf else 0
            features[f'away_conf_{conf.replace(" ", "_").lower()}'] = 1 if away_conf == conf else 0

        return features

    def _derived_features(self, features):
        """Create derived/interaction features"""
        derived = {}

        # Offense vs Defense matchups
        if 'home_sp_offense' in features and 'away_sp_defense' in features:
            derived['home_offense_vs_away_defense'] = features['home_sp_offense'] - features['away_sp_defense']
            derived['away_offense_vs_home_defense'] = features['away_sp_offense'] - features['home_sp_defense']

        # Scoring potential
        if 'home_ppg' in features and 'away_points_allowed_pg' in features:
            derived['home_scoring_potential'] = features['home_ppg'] - features['away_points_allowed_pg']
            derived['away_scoring_potential'] = features['away_ppg'] - features['home_points_allowed_pg']

        # Pace interaction
        if 'home_sp_tempo' in features and 'away_sp_tempo' in features:
            derived['combined_tempo'] = (features['home_sp_tempo'] + features['away_sp_tempo']) / 2
            derived['tempo_difference'] = abs(features['home_sp_tempo'] - features['away_sp_tempo'])

        # Efficiency product
        if 'home_3rd_down_pct' in features and 'home_redzone_pct' in features:
            derived['home_efficiency_product'] = features['home_3rd_down_pct'] * features['home_redzone_pct']
            derived['away_efficiency_product'] = features['away_3rd_down_pct'] * features['away_redzone_pct']

        # Turnover battle
        if 'home_turnover_margin' in features:
            derived['turnover_battle'] = features['home_turnover_margin'] - features['away_turnover_margin']

        # Primetime + Ranked interaction
        if 'is_primetime' in features and 'is_ranked_matchup' in features:
            derived['primetime_ranked'] = features['is_primetime'] * features['is_ranked_matchup']

        # Power rating (simplified)
        if 'home_sp_rating' in features and 'home_win_pct' in features:
            derived['home_power_rating'] = features['home_sp_rating'] * features['home_win_pct']
            derived['away_power_rating'] = features['away_sp_rating'] * features['away_win_pct']
            derived['power_rating_diff'] = derived['home_power_rating'] - derived['away_power_rating']

        return derived

    def _detect_rivalry(self, game):
        """Detect if game is a rivalry"""
        home = game.get('home_team', '').lower()
        away = game.get('away_team', '').lower()

        # Famous rivalries (add more as needed)
        rivalries = [
            ('alabama', 'auburn'),
            ('michigan', 'ohio state'),
            ('texas', 'oklahoma'),
            ('florida', 'georgia'),
            ('usc', 'notre dame'),
            ('clemson', 'south carolina'),
            ('florida', 'florida state'),
            ('miami', 'florida state'),
            ('oregon', 'washington'),
            ('stanford', 'california'),
        ]

        for team1, team2 in rivalries:
            if (team1 in home and team2 in away) or (team2 in home and team1 in away):
                return 1

        return 0

    def create_dataset(self, games, year, target='spread'):
        """
        Create training dataset for a specific prediction target
        
        Args:
            games: List of game dicts
            year: Season year
            target: 'spread', 'total', 'moneyline', '1h_spread', 'home_total', 'away_total'
        
        Returns:
            X (features), y (labels)
        """
        X_list = []
        y_list = []

        for game in games:
            # Skip if not completed
            if not game.get('completed'):
                continue

            # Engineer features
            features = self.engineer_features(game, year)

            # Get target variable
            y = self._get_target(game, target)

            if y is not None:
                X_list.append(features)
                y_list.append(y)

        # Convert to DataFrame/arrays
        if X_list:
            X = pd.DataFrame(X_list)
            y = np.array(y_list)
            return X, y
        else:
            return None, None

    def _get_target(self, game, target_type):
        """Extract target variable based on prediction type"""
        home_score = game.get('homePoints') or game.get('home_points') or game.get('home_score')
        away_score = game.get('awayPoints') or game.get('away_points') or game.get('away_score')

        if home_score is None or away_score is None:
            return None

        if target_type == 'spread':
            # Home team spread (positive = home won by more)
            return home_score - away_score

        elif target_type == 'total':
            # Total points
            return home_score + away_score

        elif target_type == 'moneyline':
            # Binary: 1 = home win, 0 = away win
            return 1 if home_score > away_score else 0

        elif target_type == '1h_spread':
            # First half spread (if available)
            home_1h = game.get('home_points_1h', home_score * 0.45)  # Estimate if missing
            away_1h = game.get('away_points_1h', away_score * 0.45)
            return home_1h - away_1h

        elif target_type == 'home_total':
            # Home team total points
            return home_score

        elif target_type == 'away_total':
            # Away team total points
            return away_score

        else:
            return None
