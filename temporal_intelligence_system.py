#!/usr/bin/env python3
"""
Temporal Intelligence System - YOLO MODE
=======================================

Analyzes performance trends over time, seasonal patterns,
rest advantage impacts, and momentum/streak quantification.

YOLO MODE: Maximum temporal analysis with time-series ML.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from collections import deque, defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction types"""
    STRONG_UP = "strong_up"
    MODERATE_UP = "moderate_up"
    FLAT = "flat"
    MODERATE_DOWN = "moderate_down"
    STRONG_DOWN = "strong_down"


@dataclass
class PerformanceTrend:
    """Performance trend analysis"""
    team: str
    metric: str
    trend_direction: TrendDirection
    trend_strength: float  # 0-1 scale
    recent_performance: List[float]
    trend_duration_weeks: int
    momentum_score: float
    confidence: float
    prediction_next_game: float


@dataclass
class SeasonalPattern:
    """Seasonal performance pattern"""
    team: str
    pattern_type: str
    early_season_performance: float
    mid_season_performance: float
    late_season_performance: float
    playoff_performance: float
    pattern_strength: float
    confidence: float


@dataclass
class RestAdvantage:
    """Rest advantage analysis"""
    team: str
    days_rest: int
    opponent_days_rest: int
    rest_advantage: float
    historical_performance_with_rest: float
    fatigue_factor: float
    travel_factor: float
    total_advantage: float


class PerformanceTrendAnalyzer:
    """Analyzes team and player performance trends"""
    
    def __init__(self, lookback_weeks: int = 8):
        self.lookback_weeks = lookback_weeks
        self.team_trends: Dict[str, Dict[str, PerformanceTrend]] = defaultdict(dict)
        
        # Performance metrics to track
        self.metrics = [
            'points_scored', 'points_allowed', 'total_yards', 'turnovers',
            'third_down_pct', 'red_zone_pct', 'time_of_possession'
        ]
    
    async def analyze_team_trends(self, team: str, historical_data: List[Dict]) -> Dict[str, PerformanceTrend]:
        """Analyze trends for all metrics for a team"""
        trends = {}
        
        try:
            if len(historical_data) < 3:
                return self._create_default_trends(team)
            
            # Sort data by date
            sorted_data = sorted(historical_data, key=lambda x: x.get('date', '2024-01-01'))
            recent_data = sorted_data[-self.lookback_weeks:]
            
            for metric in self.metrics:
                trend = await self._analyze_single_metric_trend(team, metric, recent_data)
                trends[metric] = trend
                self.team_trends[team][metric] = trend
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing trends for {team}: {e}")
            return self._create_default_trends(team)
    
    async def _analyze_single_metric_trend(self, team: str, metric: str, data: List[Dict]) -> PerformanceTrend:
        """Analyze trend for a single metric"""
        try:
            # Extract metric values
            values = []
            for game in data:
                value = game.get(metric, 0)
                if isinstance(value, (int, float)):
                    values.append(float(value))
                else:
                    values.append(0.0)
            
            if len(values) < 2:
                return self._create_default_trend(team, metric)
            
            # Fit linear regression to detect trend
            X = np.array(range(len(values))).reshape(-1, 1)
            y = np.array(values)
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate trend metrics
            slope = model.coef_[0]
            r_squared = model.score(X, y)
            
            # Determine trend direction and strength
            trend_direction = self._classify_trend_direction(slope, np.std(values))
            trend_strength = min(abs(slope) / max(np.std(values), 1), 1.0)
            
            # Calculate momentum (recent games weighted more)
            weights = np.linspace(0.5, 1.0, len(values))
            momentum_score = np.average(values, weights=weights) / max(np.mean(values), 1)
            momentum_score = min(max(momentum_score - 1, -1), 1)  # Normalize to -1 to 1
            
            # Predict next game
            next_prediction = model.predict([[len(values)]])[0]
            
            trend = PerformanceTrend(
                team=team,
                metric=metric,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                recent_performance=values[-5:],  # Last 5 games
                trend_duration_weeks=len(values),
                momentum_score=momentum_score,
                confidence=r_squared,
                prediction_next_game=next_prediction
            )
            
            return trend
            
        except Exception as e:
            logger.error(f"Error analyzing {metric} trend for {team}: {e}")
            return self._create_default_trend(team, metric)
    
    def _classify_trend_direction(self, slope: float, std_dev: float) -> TrendDirection:
        """Classify trend direction based on slope"""
        normalized_slope = slope / max(std_dev, 1)
        
        if normalized_slope > 0.5:
            return TrendDirection.STRONG_UP
        elif normalized_slope > 0.2:
            return TrendDirection.MODERATE_UP
        elif normalized_slope < -0.5:
            return TrendDirection.STRONG_DOWN
        elif normalized_slope < -0.2:
            return TrendDirection.MODERATE_DOWN
        else:
            return TrendDirection.FLAT
    
    def _create_default_trends(self, team: str) -> Dict[str, PerformanceTrend]:
        """Create default trends when insufficient data"""
        return {metric: self._create_default_trend(team, metric) for metric in self.metrics}
    
    def _create_default_trend(self, team: str, metric: str) -> PerformanceTrend:
        """Create default trend for insufficient data"""
        return PerformanceTrend(
            team=team,
            metric=metric,
            trend_direction=TrendDirection.FLAT,
            trend_strength=0.1,
            recent_performance=[0.0],
            trend_duration_weeks=1,
            momentum_score=0.0,
            confidence=0.2,
            prediction_next_game=0.0
        )


class SeasonalPatternAnalyzer:
    """Analyzes seasonal performance patterns"""
    
    def __init__(self):
        self.seasonal_patterns: Dict[str, SeasonalPattern] = {}
    
    async def analyze_seasonal_patterns(self, team: str, season_data: List[Dict]) -> SeasonalPattern:
        """Analyze seasonal performance patterns"""
        try:
            if len(season_data) < 6:
                return self._create_default_pattern(team)
            
            # Divide season into phases
            total_games = len(season_data)
            early_cutoff = total_games // 3
            late_cutoff = 2 * total_games // 3
            
            early_games = season_data[:early_cutoff]
            mid_games = season_data[early_cutoff:late_cutoff]
            late_games = season_data[late_cutoff:]
            
            # Calculate performance for each phase
            early_perf = self._calculate_phase_performance(early_games)
            mid_perf = self._calculate_phase_performance(mid_games)
            late_perf = self._calculate_phase_performance(late_games)
            
            # Detect pattern type
            pattern_type = self._detect_pattern_type(early_perf, mid_perf, late_perf)
            
            # Calculate pattern strength
            performance_variance = np.var([early_perf, mid_perf, late_perf])
            pattern_strength = min(performance_variance / 0.1, 1.0)  # Normalize
            
            pattern = SeasonalPattern(
                team=team,
                pattern_type=pattern_type,
                early_season_performance=early_perf,
                mid_season_performance=mid_perf,
                late_season_performance=late_perf,
                playoff_performance=late_perf * 1.1,  # Estimate playoff performance
                pattern_strength=pattern_strength,
                confidence=0.7 if len(season_data) > 10 else 0.5
            )
            
            self.seasonal_patterns[team] = pattern
            return pattern
            
        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns for {team}: {e}")
            return self._create_default_pattern(team)
    
    def _calculate_phase_performance(self, games: List[Dict]) -> float:
        """Calculate performance score for a phase"""
        if not games:
            return 0.5
        
        # Simple win rate calculation
        wins = sum(1 for game in games if game.get('won', False))
        return wins / len(games)
    
    def _detect_pattern_type(self, early: float, mid: float, late: float) -> str:
        """Detect seasonal pattern type"""
        if early > mid > late:
            return "Early Season Team"
        elif late > mid > early:
            return "Late Season Team"
        elif mid > early and mid > late:
            return "Mid Season Peak"
        elif early > 0.6 and late > 0.6:
            return "Consistent Performer"
        else:
            return "Inconsistent"
    
    def _create_default_pattern(self, team: str) -> SeasonalPattern:
        """Create default pattern when insufficient data"""
        return SeasonalPattern(
            team=team,
            pattern_type="Unknown",
            early_season_performance=0.5,
            mid_season_performance=0.5,
            late_season_performance=0.5,
            playoff_performance=0.5,
            pattern_strength=0.1,
            confidence=0.3
        )


class RestAdvantageCalculator:
    """Calculates rest and travel advantages"""
    
    def __init__(self):
        self.rest_impact_curve = {
            0: -0.15,   # No rest (back-to-back impossible in NFL, but short week)
            3: -0.05,   # Short rest
            7: 0.0,     # Normal rest
            10: 0.05,   # Extra rest
            14: 0.1,    # Long rest
            21: 0.05    # Too much rest can be negative
        }
    
    async def calculate_rest_advantage(
        self,
        team: str,
        days_rest: int,
        opponent_days_rest: int,
        travel_distance: float = 0
    ) -> RestAdvantage:
        """Calculate rest and travel advantages"""
        try:
            # Calculate rest impact for both teams
            team_rest_impact = self._interpolate_rest_impact(days_rest)
            opponent_rest_impact = self._interpolate_rest_impact(opponent_days_rest)
            
            # Rest advantage is the difference
            rest_advantage = team_rest_impact - opponent_rest_impact
            
            # Calculate travel factor
            travel_factor = self._calculate_travel_impact(travel_distance)
            
            # Calculate fatigue factor (inverse of rest)
            fatigue_factor = max(0, (7 - days_rest) / 7) * 0.1
            
            # Historical performance with similar rest
            historical_perf = self._estimate_historical_rest_performance(team, days_rest)
            
            # Total advantage
            total_advantage = rest_advantage - travel_factor - fatigue_factor
            
            advantage = RestAdvantage(
                team=team,
                days_rest=days_rest,
                opponent_days_rest=opponent_days_rest,
                rest_advantage=rest_advantage,
                historical_performance_with_rest=historical_perf,
                fatigue_factor=fatigue_factor,
                travel_factor=travel_factor,
                total_advantage=total_advantage
            )
            
            return advantage
            
        except Exception as e:
            logger.error(f"Error calculating rest advantage: {e}")
            return RestAdvantage(
                team=team,
                days_rest=days_rest,
                opponent_days_rest=opponent_days_rest,
                rest_advantage=0.0,
                historical_performance_with_rest=0.5,
                fatigue_factor=0.0,
                travel_factor=0.0,
                total_advantage=0.0
            )
    
    def _interpolate_rest_impact(self, days_rest: int) -> float:
        """Interpolate rest impact from curve"""
        if days_rest in self.rest_impact_curve:
            return self.rest_impact_curve[days_rest]
        
        # Linear interpolation between known points
        rest_days = sorted(self.rest_impact_curve.keys())
        
        if days_rest < rest_days[0]:
            return self.rest_impact_curve[rest_days[0]]
        if days_rest > rest_days[-1]:
            return self.rest_impact_curve[rest_days[-1]]
        
        # Find surrounding points
        for i in range(len(rest_days) - 1):
            if rest_days[i] <= days_rest <= rest_days[i + 1]:
                x1, x2 = rest_days[i], rest_days[i + 1]
                y1, y2 = self.rest_impact_curve[x1], self.rest_impact_curve[x2]
                
                # Linear interpolation
                return y1 + (y2 - y1) * (days_rest - x1) / (x2 - x1)
        
        return 0.0
    
    def _calculate_travel_impact(self, distance: float) -> float:
        """Calculate travel impact (negative factor)"""
        if distance < 500:
            return 0.0      # Minimal travel
        elif distance < 1500:
            return 0.02     # Moderate travel
        elif distance < 2500:
            return 0.04     # Long travel
        else:
            return 0.06     # Cross-country travel
    
    def _estimate_historical_rest_performance(self, team: str, days_rest: int) -> float:
        """Estimate historical performance with similar rest"""
        # Simplified estimation - in production would use actual historical data
        base_performance = 0.5  # 50% win rate baseline
        
        if days_rest >= 10:
            return base_performance + 0.1  # Well-rested teams perform better
        elif days_rest <= 4:
            return base_performance - 0.05  # Short rest hurts performance
        else:
            return base_performance


class MomentumAnalyzer:
    """Analyzes team momentum and streaks"""
    
    def __init__(self):
        self.momentum_factors = {
            'win_streak': 0.02,      # 2% per consecutive win
            'loss_streak': -0.02,    # -2% per consecutive loss
            'scoring_trend': 0.01,   # 1% per point trend
            'defensive_trend': 0.01, # 1% per point allowed trend
            'turnover_trend': 0.03   # 3% per turnover differential trend
        }
    
    async def calculate_momentum(self, team: str, recent_games: List[Dict]) -> Dict[str, float]:
        """Calculate team momentum score"""
        try:
            if len(recent_games) < 3:
                return {'momentum_score': 0.0, 'streak_type': 'none', 'streak_length': 0}
            
            # Sort games by date
            sorted_games = sorted(recent_games, key=lambda x: x.get('date', '2024-01-01'))
            
            # Calculate win/loss streak
            streak_length, streak_type = self._calculate_streak(sorted_games)
            
            # Calculate scoring trends
            scores = [game.get('points_scored', 0) for game in sorted_games]
            allowed = [game.get('points_allowed', 0) for game in sorted_games]
            
            scoring_trend = self._calculate_linear_trend(scores)
            defensive_trend = -self._calculate_linear_trend(allowed)  # Lower allowed = better
            
            # Calculate momentum score
            streak_momentum = streak_length * self.momentum_factors[f'{streak_type}_streak']
            scoring_momentum = scoring_trend * self.momentum_factors['scoring_trend']
            defensive_momentum = defensive_trend * self.momentum_factors['defensive_trend']
            
            total_momentum = streak_momentum + scoring_momentum + defensive_momentum
            total_momentum = max(-0.3, min(0.3, total_momentum))  # Cap at ±30%
            
            return {
                'momentum_score': total_momentum,
                'streak_type': streak_type,
                'streak_length': streak_length,
                'scoring_trend': scoring_trend,
                'defensive_trend': defensive_trend,
                'total_momentum': total_momentum
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum for {team}: {e}")
            return {'momentum_score': 0.0, 'streak_type': 'none', 'streak_length': 0}
    
    def _calculate_streak(self, games: List[Dict]) -> Tuple[int, str]:
        """Calculate current win/loss streak"""
        if not games:
            return 0, 'none'
        
        # Start from most recent game
        current_result = games[-1].get('won', False)
        streak_length = 1
        streak_type = 'win' if current_result else 'loss'
        
        # Count consecutive games with same result
        for i in range(len(games) - 2, -1, -1):
            if games[i].get('won', False) == current_result:
                streak_length += 1
            else:
                break
        
        return streak_length, streak_type
    
    def _calculate_linear_trend(self, values: List[float]) -> float:
        """Calculate linear trend slope"""
        if len(values) < 2:
            return 0.0
        
        X = np.array(range(len(values))).reshape(-1, 1)
        y = np.array(values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model.coef_[0]


class TemporalIntelligenceSystem:
    """Complete temporal intelligence system"""
    
    def __init__(self):
        self.trend_analyzer = PerformanceTrendAnalyzer()
        self.seasonal_analyzer = SeasonalPatternAnalyzer()
        self.rest_calculator = RestAdvantageCalculator()
        self.momentum_analyzer = MomentumAnalyzer()
        
        self.stats = {
            'trends_analyzed': 0,
            'patterns_detected': 0,
            'momentum_calculations': 0,
            'rest_advantages_calculated': 0,
            'start_time': datetime.now()
        }
    
    async def generate_temporal_intelligence(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete temporal intelligence"""
        try:
            home_team = game_data.get('home_team', 'HOME')
            away_team = game_data.get('away_team', 'AWAY')
            
            # Generate mock historical data for demo
            home_historical = self._generate_mock_historical_data(home_team)
            away_historical = self._generate_mock_historical_data(away_team)
            
            # Analyze trends
            home_trends = await self.trend_analyzer.analyze_team_trends(home_team, home_historical)
            away_trends = await self.trend_analyzer.analyze_team_trends(away_team, away_historical)
            self.stats['trends_analyzed'] += 2
            
            # Analyze seasonal patterns
            home_pattern = await self.seasonal_analyzer.analyze_seasonal_patterns(home_team, home_historical)
            away_pattern = await self.seasonal_analyzer.analyze_seasonal_patterns(away_team, away_historical)
            self.stats['patterns_detected'] += 2
            
            # Calculate momentum
            home_momentum = await self.momentum_analyzer.calculate_momentum(home_team, home_historical[-5:])
            away_momentum = await self.momentum_analyzer.calculate_momentum(away_team, away_historical[-5:])
            self.stats['momentum_calculations'] += 2
            
            # Calculate rest advantages
            home_rest = await self.rest_calculator.calculate_rest_advantage(
                home_team, 
                game_data.get('home_days_rest', 7),
                game_data.get('away_days_rest', 7),
                game_data.get('away_travel_distance', 0)
            )
            away_rest = await self.rest_calculator.calculate_rest_advantage(
                away_team,
                game_data.get('away_days_rest', 7), 
                game_data.get('home_days_rest', 7),
                game_data.get('home_travel_distance', 0)
            )
            self.stats['rest_advantages_calculated'] += 2
            
            # Combine all temporal factors
            intelligence = {
                'game_id': game_data.get('game_id', 'unknown'),
                'home_team_analysis': {
                    'trends': {k: v.__dict__ for k, v in home_trends.items()},
                    'seasonal_pattern': home_pattern.__dict__,
                    'momentum': home_momentum,
                    'rest_advantage': home_rest.__dict__
                },
                'away_team_analysis': {
                    'trends': {k: v.__dict__ for k, v in away_trends.items()},
                    'seasonal_pattern': away_pattern.__dict__,
                    'momentum': away_momentum,
                    'rest_advantage': away_rest.__dict__
                },
                'temporal_edge': self._calculate_temporal_edge(
                    home_trends, away_trends, home_momentum, away_momentum, home_rest, away_rest
                ),
                'timestamp': datetime.now()
            }
            
            return intelligence
            
        except Exception as e:
            logger.error(f"Error generating temporal intelligence: {e}")
            return {'error': str(e)}
    
    def _generate_mock_historical_data(self, team: str) -> List[Dict]:
        """Generate mock historical data for demo"""
        games = []
        base_score = 24
        
        for week in range(1, 13):  # 12 weeks of data
            # Add some randomness and trends
            points_scored = base_score + np.random.normal(0, 7) + (week * 0.5)  # Improving trend
            points_allowed = 20 + np.random.normal(0, 5) - (week * 0.2)  # Defensive improvement
            
            game = {
                'date': f'2024-{week:02d}-01',
                'week': week,
                'points_scored': max(0, points_scored),
                'points_allowed': max(0, points_allowed),
                'total_yards': 350 + np.random.normal(0, 50),
                'turnovers': max(0, np.random.poisson(1.5)),
                'third_down_pct': 0.4 + np.random.normal(0, 0.1),
                'red_zone_pct': 0.6 + np.random.normal(0, 0.1),
                'time_of_possession': 30 + np.random.normal(0, 5),
                'won': points_scored > points_allowed
            }
            
            games.append(game)
        
        return games
    
    def _calculate_temporal_edge(
        self,
        home_trends: Dict,
        away_trends: Dict,
        home_momentum: Dict,
        away_momentum: Dict,
        home_rest: RestAdvantage,
        away_rest: RestAdvantage
    ) -> Dict[str, float]:
        """Calculate overall temporal edge"""
        try:
            # Trend edge
            home_trend_score = np.mean([t.momentum_score for t in home_trends.values()])
            away_trend_score = np.mean([t.momentum_score for t in away_trends.values()])
            trend_edge = home_trend_score - away_trend_score
            
            # Momentum edge
            momentum_edge = home_momentum['momentum_score'] - away_momentum['momentum_score']
            
            # Rest edge
            rest_edge = home_rest.total_advantage - away_rest.total_advantage
            
            # Combined temporal edge
            total_edge = (trend_edge * 0.4 + momentum_edge * 0.4 + rest_edge * 0.2)
            
            return {
                'trend_edge': trend_edge,
                'momentum_edge': momentum_edge,
                'rest_edge': rest_edge,
                'total_temporal_edge': total_edge,
                'edge_strength': abs(total_edge),
                'favored_team': 'home' if total_edge > 0 else 'away'
            }
            
        except Exception as e:
            logger.error(f"Error calculating temporal edge: {e}")
            return {'total_temporal_edge': 0.0}


async def main():
    """YOLO MODE Demo - Temporal Intelligence System"""
    print("⏰ TEMPORAL INTELLIGENCE SYSTEM - YOLO MODE")
    print("=" * 60)
    
    system = TemporalIntelligenceSystem()
    
    # Test scenarios
    scenarios = [
        {
            'game_id': 'KC_vs_BAL',
            'home_team': 'KC',
            'away_team': 'BAL',
            'home_days_rest': 10,
            'away_days_rest': 6,
            'away_travel_distance': 1200
        },
        {
            'game_id': 'BUF_vs_MIA',
            'home_team': 'BUF',
            'away_team': 'MIA',
            'home_days_rest': 7,
            'away_days_rest': 7,
            'away_travel_distance': 800
        }
    ]
    
    print("⏱️ Analyzing temporal intelligence...")
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {i+1}: {scenario['game_id']} ---")
        
        intelligence = await system.generate_temporal_intelligence(scenario)
        
        if 'error' not in intelligence:
            edge = intelligence['temporal_edge']
            home_analysis = intelligence['home_team_analysis']
            away_analysis = intelligence['away_team_analysis']
            
            print(f"Temporal Edge: {edge['total_temporal_edge']:+.3f}")
            print(f"Favored Team: {edge['favored_team'].upper()}")
            print(f"Edge Strength: {edge['edge_strength']:.3f}")
            
            print(f"\nHOME ({scenario['home_team']}):")
            print(f"  Momentum: {home_analysis['momentum']['momentum_score']:+.3f}")
            print(f"  Rest Advantage: {home_analysis['rest_advantage']['total_advantage']:+.3f}")
            print(f"  Seasonal Pattern: {home_analysis['seasonal_pattern']['pattern_type']}")
            
            print(f"AWAY ({scenario['away_team']}):")
            print(f"  Momentum: {away_analysis['momentum']['momentum_score']:+.3f}")
            print(f"  Rest Advantage: {away_analysis['rest_advantage']['total_advantage']:+.3f}")
            print(f"  Seasonal Pattern: {away_analysis['seasonal_pattern']['pattern_type']}")
    
    print("\n" + "=" * 60)
    print("📊 TEMPORAL INTELLIGENCE SUMMARY")
    print("=" * 60)
    
    stats = system.stats
    uptime = (datetime.now() - stats['start_time']).total_seconds()
    
    print(f"Trends Analyzed: {stats['trends_analyzed']}")
    print(f"Patterns Detected: {stats['patterns_detected']}")
    print(f"Momentum Calculations: {stats['momentum_calculations']}")
    print(f"Rest Advantages: {stats['rest_advantages_calculated']}")
    print(f"Analysis Rate: {(stats['trends_analyzed'] / max(uptime, 1)):.1f} trends/sec")
    
    print("\n✅ TASK 24 COMPLETE - Temporal Intelligence System DELIVERED!")


if __name__ == "__main__":
    asyncio.run(main())
