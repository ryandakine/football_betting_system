#!/usr/bin/env python3
"""
Advanced Feature Engineering v2 - Production Quality
=====================================================

Transforms raw NFL game data into predictive features for better model accuracy.

Key features:
- Injury impact scoring (key players, positional groups)
- Rest differential (days rest between teams)
- Vegas line movement analysis (opening vs current lines)
- Team strength metrics (EPA, DVOA, efficiency)
- Advanced matchup analysis (offensive vs defensive metrics)
- Weather adjustments
- Home/away splits
- Historical head-to-head data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class InjuryProfile:
    """Injury impact on team"""
    team: str
    key_players_out: List[str]  # QB, RB, top WR, best OL, top CB/pass rusher
    impact_score: float  # 0-1, higher is worse
    position_group_impact: Dict[str, float]  # offense, defense, special teams
    games_affected: int
    

@dataclass
class TeamStrength:
    """Comprehensive team strength metrics"""
    team: str
    offensive_epa: float  # EPA per play on offense
    defensive_epa: float  # EPA per play allowed on defense
    offensive_dvoa: float  # DVOA (defense-adjusted value over average)
    defensive_dvoa: float
    pass_efficiency: float
    rush_efficiency: float
    third_down_conversion: float
    red_zone_efficiency: float
    pass_rush_rating: float
    coverage_rating: float
    

@dataclass
class MatchupAdvantage:
    """Specific matchup strengths/weaknesses"""
    matchup_id: str
    offense_vs_defense: float  # Positive favors offense
    key_player_edge: float  # Based on specific matchups
    personnel_efficiency: float
    historical_vs_this_opponent: float


class AdvancedFeatureEngineer:
    """Production-grade feature engineering pipeline"""
    
    def __init__(self):
        self.injury_cache = {}
        self.team_strength_cache = {}
        self.matchup_cache = {}
        self.line_movement_history = {}
        
        # Historical NFL averages (normalization)
        self.nfl_avg_epa_offense = 0.08
        self.nfl_avg_epa_defense = -0.08
        self.nfl_avg_dvoa = 0.0
        self.nfl_avg_third_down_conversion = 0.42
        self.nfl_avg_red_zone_efficiency = 0.62
        
    def calculate_injury_impact(self, game_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate injury impact for both teams.
        
        Returns injury scores for home and away teams (0-1, higher = worse impact)
        """
        injuries = game_data.get('injuries', {})
        
        impact = {
            'home_injury_score': 0.0,
            'away_injury_score': 0.0,
            'home_key_positions': [],
            'away_key_positions': [],
            'qb_injury_multiplier': 1.0,  # 1.0 = no QB injury, 1.5+ = QB injury
            'rb_injury_multiplier': 1.0,
            'wr_injury_multiplier': 1.0,
        }
        
        if not injuries:
            return impact
        
        # Key positions by impact severity
        key_positions = {
            'QB': 0.25,      # Largest impact
            'RB': 0.12,
            'WR': 0.08,
            'TE': 0.06,
            'OL': 0.15,      # Offensive line is critical
            'EDGE': 0.10,    # Pass rushers
            'CB': 0.08,
            'S': 0.06,
        }
        
        home_injuries = injuries.get('home', [])
        away_injuries = injuries.get('away', [])
        
        # Calculate home team impact
        for player_info in home_injuries:
            pos = player_info.get('position', '').upper()
            if pos in key_positions:
                impact['home_injury_score'] += key_positions[pos]
                impact['home_key_positions'].append(pos)
                
                # Special multipliers for QB
                if pos == 'QB':
                    impact['qb_injury_multiplier'] = 1.4
                elif pos == 'RB':
                    impact['rb_injury_multiplier'] = 1.2
                elif pos == 'WR':
                    impact['wr_injury_multiplier'] = 1.15
        
        # Calculate away team impact
        for player_info in away_injuries:
            pos = player_info.get('position', '').upper()
            if pos in key_positions:
                impact['away_injury_score'] += key_positions[pos]
                impact['away_key_positions'].append(pos)
                
                if pos == 'QB':
                    impact['qb_injury_multiplier'] = 1.4
                elif pos == 'RB':
                    impact['rb_injury_multiplier'] = 1.2
        
        # Cap at 1.0 (can't exceed total team injury impact)
        impact['home_injury_score'] = min(1.0, impact['home_injury_score'])
        impact['away_injury_score'] = min(1.0, impact['away_injury_score'])
        
        return impact
    
    def calculate_rest_differential(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate rest advantage based on days since last game.
        
        Home team advantage typically 2-3 days rest.
        """
        home_last_game = game_data.get('home_last_game_date')
        away_last_game = game_data.get('away_last_game_date')
        current_date = game_data.get('game_date', datetime.now())
        
        if not home_last_game or not away_last_game:
            return {
                'home_days_rest': 7,  # Default week
                'away_days_rest': 7,
                'rest_differential': 0.0,
                'is_short_week': False,
                'is_back_to_back': False,
            }
        
        # Convert to datetime if needed
        if isinstance(home_last_game, str):
            home_last_game = datetime.fromisoformat(home_last_game)
        if isinstance(away_last_game, str):
            away_last_game = datetime.fromisoformat(away_last_game)
        if isinstance(current_date, str):
            current_date = datetime.fromisoformat(current_date)
        
        home_rest = (current_date - home_last_game).days
        away_rest = (current_date - away_last_game).days
        
        rest_diff = home_rest - away_rest
        
        return {
            'home_days_rest': home_rest,
            'away_days_rest': away_rest,
            'rest_differential': rest_diff,
            'is_short_week': home_rest < 4 or away_rest < 4,
            'is_back_to_back': home_rest <= 2 or away_rest <= 2,
            'rest_advantage_multiplier': 1.0 + (rest_diff * 0.02),  # 2% per day
        }
    
    def calculate_line_movement(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze line movement from open to current.
        
        Positive movement favors home team (line moves up).
        Sharp money typically hits lines early.
        """
        opening_spread = game_data.get('opening_spread')
        current_spread = game_data.get('current_spread', game_data.get('spread'))
        opening_total = game_data.get('opening_total')
        current_total = game_data.get('current_total', game_data.get('total'))
        
        if not opening_spread or not current_spread:
            return {
                'spread_movement': 0.0,
                'total_movement': 0.0,
                'steam_move': False,
                'contrarian_signal': 'neutral',
            }
        
        spread_movement = current_spread - opening_spread  # Positive = moved toward home
        total_movement = (current_total or 0) - (opening_total or 0)
        
        # Steam detection: 1+ point move indicates sharp action
        is_steam = abs(spread_movement) >= 1.0 or abs(total_movement) >= 1.0
        
        # Contrarian signal: if line moved heavily one way, contrarian trade other side
        contrarian_signal = 'neutral'
        if spread_movement > 2.0:  # Moved heavily toward home
            contrarian_signal = 'fade_home_spread'
        elif spread_movement < -2.0:  # Moved heavily toward away
            contrarian_signal = 'fade_away_spread'
        
        return {
            'spread_movement': spread_movement,
            'total_movement': total_movement,
            'steam_move': is_steam,
            'contrarian_signal': contrarian_signal,
            'line_confidence': min(0.9, abs(spread_movement) * 0.3),  # More movement = more confidence
        }
    
    def calculate_team_strength(self, team: str, season_stats: Optional[Dict] = None) -> TeamStrength:
        """
        Calculate comprehensive team strength metrics.
        
        In production, these would come from NFL stats API or historical database.
        """
        if not season_stats:
            # Defaults if no stats available
            return TeamStrength(
                team=team,
                offensive_epa=0.0,
                defensive_epa=0.0,
                offensive_dvoa=0.0,
                defensive_dvoa=0.0,
                pass_efficiency=0.6,
                rush_efficiency=0.5,
                third_down_conversion=0.42,
                red_zone_efficiency=0.62,
                pass_rush_rating=0.5,
                coverage_rating=0.5,
            )
        
        return TeamStrength(
            team=team,
            offensive_epa=season_stats.get('offensive_epa', 0.0),
            defensive_epa=season_stats.get('defensive_epa', 0.0),
            offensive_dvoa=season_stats.get('offensive_dvoa', 0.0),
            defensive_dvoa=season_stats.get('defensive_dvoa', 0.0),
            pass_efficiency=season_stats.get('pass_efficiency', 0.6),
            rush_efficiency=season_stats.get('rush_efficiency', 0.5),
            third_down_conversion=season_stats.get('third_down_conversion', 0.42),
            red_zone_efficiency=season_stats.get('red_zone_efficiency', 0.62),
            pass_rush_rating=season_stats.get('pass_rush_rating', 0.5),
            coverage_rating=season_stats.get('coverage_rating', 0.5),
        )
    
    def calculate_matchup_advantage(
        self, 
        home_strength: TeamStrength, 
        away_strength: TeamStrength,
        game_data: Dict[str, Any]
    ) -> MatchupAdvantage:
        """
        Calculate specific matchup advantages.
        
        Offense EPA vs Defense EPA is most predictive metric.
        """
        # Primary matchup: Offensive EPA vs Defensive EPA allowed
        home_offense_vs_away_defense = (
            home_strength.offensive_epa - away_strength.defensive_epa
        )
        away_offense_vs_home_defense = (
            away_strength.offensive_epa - home_strength.defensive_epa
        )
        
        # Net advantage for home team
        matchup_edge = (home_offense_vs_away_defense - away_offense_vs_home_defense) / 2
        
        # Key player matchups
        key_player_edge = 0.0  # Would be calculated from specific player stats
        
        # Historical performance vs opponent
        historical_record = game_data.get('historical_vs_opponent', {})
        home_win_pct = historical_record.get('home_win_pct', 0.5)
        historical_vs_opponent = (home_win_pct - 0.5) * 2  # Scale to [-1, 1]
        
        return MatchupAdvantage(
            matchup_id=f"{home_strength.team}_{away_strength.team}",
            offense_vs_defense=matchup_edge,
            key_player_edge=key_player_edge,
            personnel_efficiency=(
                (home_strength.pass_efficiency - away_strength.coverage_rating) +
                (home_strength.rush_efficiency - away_strength.pass_rush_rating)
            ) / 2,
            historical_vs_this_opponent=historical_vs_opponent,
        )
    
    def calculate_weather_adjustment(self, game_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate weather impact on game performance.
        
        Wind affects passing games, cold affects ball handling.
        Dome games are weather-independent.
        """
        weather = game_data.get('weather', {})
        is_dome = game_data.get('is_dome', False)
        
        if is_dome:
            return {
                'weather_impact_spread': 0.0,
                'weather_impact_total': 0.0,
                'conditions': 'dome',
                'wind_effect_passing': 0.0,
                'cold_effect': 0.0,
            }
        
        temperature = weather.get('temperature', 70)
        wind_mph = weather.get('wind_mph', 0)
        precipitation = weather.get('precipitation_pct', 0)
        
        # Wind severely impacts passing
        wind_effect = wind_mph * 0.03  # 3% impact per mph
        
        # Cold reduces scoring
        cold_effect = max(0, (70 - temperature) * 0.01)  # 1% per degree below 70
        
        # Rain/snow increases total impact
        weather_impact_total = -0.05 * (precipitation / 100)
        
        # Combine effects
        weather_impact_spread = -(wind_effect + cold_effect) * 0.5
        weather_impact_total -= (wind_effect * 0.02)  # Wind reduces scoring slightly
        
        return {
            'weather_impact_spread': min(0.3, weather_impact_spread),  # Cap impact
            'weather_impact_total': min(0.2, weather_impact_total),
            'conditions': self._classify_weather(temperature, wind_mph, precipitation),
            'wind_effect_passing': wind_effect,
            'cold_effect': cold_effect,
        }
    
    def _classify_weather(self, temp: float, wind: float, precip: float) -> str:
        """Classify weather conditions"""
        if precip > 50:
            return 'heavy_precipitation'
        elif precip > 20:
            return 'light_precipitation'
        elif wind > 15:
            return 'high_wind'
        elif temp < 32:
            return 'freezing'
        elif temp > 85:
            return 'hot'
        else:
            return 'clear'
    
    def calculate_home_away_splits(self, team: str, season_stats: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate home vs away performance splits.
        
        Home teams average +2.5 to +3.5 points better.
        """
        if not season_stats:
            return {
                'home_points_per_game': 0.0,
                'away_points_per_game': 0.0,
                'home_points_allowed': 0.0,
                'away_points_allowed': 0.0,
                'home_vs_away_advantage': 2.5,  # NFL average
            }
        
        home_ppg = season_stats.get('home_ppg', 0)
        away_ppg = season_stats.get('away_ppg', 0)
        home_pa = season_stats.get('home_points_allowed', 0)
        away_pa = season_stats.get('away_points_allowed', 0)
        
        advantage = (home_ppg - away_ppg) + (away_pa - home_pa)
        
        return {
            'home_points_per_game': home_ppg,
            'away_points_per_game': away_ppg,
            'home_points_allowed': home_pa,
            'away_points_allowed': away_pa,
            'home_vs_away_advantage': max(0, min(7, advantage)),  # Reasonable bounds
        }
    
    def build_complete_feature_set(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build comprehensive feature set for a single game.
        
        This is the main entry point for feature engineering.
        """
        features = {
            'game_id': game_data.get('game_id'),
            'home_team': game_data.get('home_team'),
            'away_team': game_data.get('away_team'),
            'timestamp': game_data.get('timestamp'),
            'timestamp_str': str(game_data.get('timestamp', '')),
        }
        
        # 1. Injury impact
        injury_features = self.calculate_injury_impact(game_data)
        features.update({
            f'injury_{k}': v for k, v in injury_features.items()
        })
        
        # 2. Rest differential
        rest_features = self.calculate_rest_differential(game_data)
        features.update({
            f'rest_{k}': v for k, v in rest_features.items()
        })
        
        # 3. Line movement
        line_features = self.calculate_line_movement(game_data)
        features.update({
            f'line_movement_{k}': v for k, v in line_features.items()
        })
        
        # 4. Team strength (would come from NFL stats API)
        home_season_stats = game_data.get('home_season_stats')
        away_season_stats = game_data.get('away_season_stats')
        
        home_strength = self.calculate_team_strength(
            game_data.get('home_team', ''), 
            home_season_stats
        )
        away_strength = self.calculate_team_strength(
            game_data.get('away_team', ''), 
            away_season_stats
        )
        
        features.update({
            f'home_{k}': v for k, v in home_strength.__dict__.items() if k != 'team'
        })
        features.update({
            f'away_{k}': v for k, v in away_strength.__dict__.items() if k != 'team'
        })
        
        # 5. Matchup advantages
        matchup = self.calculate_matchup_advantage(home_strength, away_strength, game_data)
        features.update({
            f'matchup_{k}': v for k, v in matchup.__dict__.items() if k != 'matchup_id'
        })
        
        # 6. Weather adjustment
        weather_features = self.calculate_weather_adjustment(game_data)
        features.update({
            f'weather_{k}': v for k, v in weather_features.items()
        })
        
        # 7. Home/away splits
        home_splits = self.calculate_home_away_splits(
            game_data.get('home_team', ''), 
            home_season_stats
        )
        away_splits = self.calculate_home_away_splits(
            game_data.get('away_team', ''), 
            away_season_stats
        )
        
        features.update({
            f'home_splits_{k}': v for k, v in home_splits.items()
        })
        features.update({
            f'away_splits_{k}': v for k, v in away_splits.items()
        })
        
        return features
    
    def engineer_dataframe(self, games: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Engineer features for multiple games at once.
        """
        feature_sets = []
        for game in games:
            features = self.build_complete_feature_set(game)
            feature_sets.append(features)
        
        df = pd.DataFrame(feature_sets)
        logger.info(f"Engineered {len(df)} games with {len(df.columns)} features")
        return df


# Example usage
if __name__ == '__main__':
    engineer = AdvancedFeatureEngineer()
    
    # Example game data
    sample_game = {
        'game_id': 'test_001',
        'home_team': 'LAC',
        'away_team': 'MIN',
        'game_date': datetime.now(),
        'injuries': {
            'home': [{'position': 'WR', 'name': 'Star WR'}],
            'away': []
        },
        'current_spread': -3.5,
        'opening_spread': -2.5,
        'weather': {
            'temperature': 55,
            'wind_mph': 12,
            'precipitation_pct': 20
        }
    }
    
    features = engineer.build_complete_feature_set(sample_game)
    print(json.dumps(features, indent=2, default=str))