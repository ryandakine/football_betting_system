#!/usr/bin/env python3
"""
NCAA Conference Crew Analyzer
Analyzes officiating patterns by conference rather than individual refs

NCAA officiating structure:
- Each conference has its own officiating staff
- SEC crews officiate SEC games
- Big Ten crews officiate Big Ten games
- etc.

Cross-conference games use either:
- Home conference's crew
- Neutral crew assignments
- Specific bowl game crews

Key differences from NFL:
- Conference loyalty/bias
- Home team advantage varies by conference
- Different interpretations of rules by conference
- Varying penalty strictness
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class OfficiatingEvent:
    """NCAA officiating event"""
    game_id: str
    home_team: str
    away_team: str
    home_conference: str
    away_conference: str
    officiating_conference: str  # Which conference's crew
    penalty_type: str
    team_penalized: str  # 'home' or 'away'
    quarter: int
    game_time_remaining: int
    yards: int
    score_differential: int  # At time of penalty
    is_critical_play: bool  # Late game, close score
    is_conference_game: bool
    is_rivalry: bool
    game_outcome: str  # 'home_win' or 'away_win'


class ConferenceCrewAnalyzer:
    """
    Analyzes NCAA officiating patterns by conference
    
    Key analyses:
    1. Home bias by conference
    2. Conference protection (favoring own teams)
    3. Cross-conference game patterns
    4. Critical call tendencies
    5. Rivalry game differences
    """

    def __init__(self, data_dir="data/football/ncaaf/officiating"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Conference-specific priors
        self.conference_priors = {
            'SEC': {
                'home_bias': 0.54,  # SEC known for home cooking
                'penalty_rate': 12.3,  # Penalties per game
                'holding_strictness': 0.62,
                'pi_strictness': 0.58
            },
            'Big Ten': {
                'home_bias': 0.52,
                'penalty_rate': 11.8,
                'holding_strictness': 0.68,  # Big Ten calls lots of holds
                'pi_strictness': 0.52
            },
            'Big 12': {
                'home_bias': 0.51,
                'penalty_rate': 13.5,  # Most penalties
                'holding_strictness': 0.58,
                'pi_strictness': 0.65  # Big 12 DBs get away with more
            },
            'ACC': {
                'home_bias': 0.53,
                'penalty_rate': 11.2,
                'holding_strictness': 0.60,
                'pi_strictness': 0.60
            },
            'Pac-12': {
                'home_bias': 0.50,  # Most balanced
                'penalty_rate': 10.9,  # Fewest penalties
                'holding_strictness': 0.55,
                'pi_strictness': 0.55
            }
        }

    def analyze_conference_patterns(self, events: List[OfficiatingEvent]) -> Dict:
        """
        Main analysis function - analyzes all conference patterns
        """
        results = {
            'home_bias_by_conference': self._analyze_home_bias(events),
            'conference_protection': self._analyze_conference_protection(events),
            'cross_conference_patterns': self._analyze_cross_conference(events),
            'critical_call_analysis': self._analyze_critical_calls(events),
            'rivalry_officiating': self._analyze_rivalry_games(events),
            'penalty_timing_patterns': self._analyze_penalty_timing(events),
            'statistical_anomalies': self._detect_statistical_anomalies(events)
        }
        
        return results

    def _analyze_home_bias(self, events: List[OfficiatingEvent]) -> Dict:
        """
        Analyze home bias by conference
        
        Expected: ~52% of penalties on away team (home field advantage)
        Anomaly: Significantly above or below this
        """
        conference_stats = defaultdict(lambda: {
            'home_penalties': 0,
            'away_penalties': 0,
            'total_games': 0,
            'conference_games': 0,
            'non_conference_games': 0
        })

        for event in events:
            conf = event.officiating_conference
            stats_dict = conference_stats[conf]
            
            stats_dict['total_games'] += 1
            
            if event.team_penalized == 'away':
                stats_dict['away_penalties'] += 1
            else:
                stats_dict['home_penalties'] += 1
            
            if event.is_conference_game:
                stats_dict['conference_games'] += 1
            else:
                stats_dict['non_conference_games'] += 1

        # Calculate bias scores
        results = {}
        
        for conf, stats_dict in conference_stats.items():
            total_penalties = stats_dict['home_penalties'] + stats_dict['away_penalties']
            
            if total_penalties < 10:  # Need minimum sample
                continue
            
            away_penalty_rate = stats_dict['away_penalties'] / total_penalties
            
            # Statistical test
            p_value = stats.binomtest(
                stats_dict['away_penalties'],
                total_penalties,
                0.52  # Expected home advantage
            ).pvalue
            
            # Z-score for standardized comparison
            expected_away = total_penalties * 0.52
            std_dev = np.sqrt(total_penalties * 0.52 * 0.48)
            z_score = (stats_dict['away_penalties'] - expected_away) / std_dev if std_dev > 0 else 0
            
            # Determine bias level
            if z_score > 2.0:
                bias_level = "HIGH_HOME_BIAS"
                risk_score = 0.85
            elif z_score > 1.5:
                bias_level = "MODERATE_HOME_BIAS"
                risk_score = 0.65
            elif z_score < -2.0:
                bias_level = "AWAY_FAVORING"
                risk_score = 0.80
            elif z_score < -1.5:
                bias_level = "MODERATE_AWAY_BIAS"
                risk_score = 0.60
            else:
                bias_level = "BALANCED"
                risk_score = 0.30
            
            results[conf] = {
                'away_penalty_rate': away_penalty_rate,
                'home_penalty_rate': stats_dict['home_penalties'] / total_penalties,
                'total_penalties': total_penalties,
                'p_value': p_value,
                'z_score': z_score,
                'bias_level': bias_level,
                'risk_score': risk_score,
                'sample_size': stats_dict['total_games'],
                'statistical_significance': 'significant' if p_value < 0.05 else 'not_significant'
            }

        return results

    def _analyze_conference_protection(self, events: List[OfficiatingEvent]) -> Dict:
        """
        Analyze if conferences protect their own teams in cross-conference games
        
        Example: SEC crew officiating SEC team vs ACC team
        Do they favor the SEC team?
        """
        protection_stats = defaultdict(lambda: {
            'own_team_penalties': 0,
            'opponent_penalties': 0,
            'games': 0
        })

        for event in events:
            if event.is_conference_game:
                continue  # Only look at cross-conference games
            
            crew_conf = event.officiating_conference
            home_conf = event.home_conference
            away_conf = event.away_conference
            
            # Determine if crew's conference team is involved
            if crew_conf == home_conf and crew_conf != away_conf:
                # Crew's conference = home team
                if event.team_penalized == 'home':
                    protection_stats[crew_conf]['own_team_penalties'] += 1
                else:
                    protection_stats[crew_conf]['opponent_penalties'] += 1
                protection_stats[crew_conf]['games'] += 1
                
            elif crew_conf == away_conf and crew_conf != home_conf:
                # Crew's conference = away team
                if event.team_penalized == 'away':
                    protection_stats[crew_conf]['own_team_penalties'] += 1
                else:
                    protection_stats[crew_conf]['opponent_penalties'] += 1
                protection_stats[crew_conf]['games'] += 1

        results = {}
        
        for conf, stats_dict in protection_stats.items():
            total = stats_dict['own_team_penalties'] + stats_dict['opponent_penalties']
            
            if total < 5:  # Minimum sample
                continue
            
            opponent_penalty_rate = stats_dict['opponent_penalties'] / total
            
            # Expected: 52% penalties on opponent (normal home advantage)
            # If significantly higher = protection
            p_value = stats.binomtest(
                stats_dict['opponent_penalties'],
                total,
                0.52
            ).pvalue
            
            z_score = (opponent_penalty_rate - 0.52) / (np.sqrt(0.52 * 0.48 / total)) if total > 0 else 0
            
            if z_score > 2.0:
                protection_level = "HIGH_PROTECTION"
                risk_score = 0.90
            elif z_score > 1.5:
                protection_level = "MODERATE_PROTECTION"
                risk_score = 0.70
            else:
                protection_level = "BALANCED"
                risk_score = 0.35
            
            results[conf] = {
                'opponent_penalty_rate': opponent_penalty_rate,
                'own_team_penalty_rate': stats_dict['own_team_penalties'] / total,
                'protection_level': protection_level,
                'risk_score': risk_score,
                'p_value': p_value,
                'z_score': z_score,
                'sample_size': stats_dict['games']
            }

        return results

    def _analyze_cross_conference(self, events: List[OfficiatingEvent]) -> Dict:
        """
        Analyze patterns in cross-conference matchups
        """
        matchup_stats = defaultdict(lambda: {
            'home_wins': 0,
            'away_wins': 0,
            'home_penalties': 0,
            'away_penalties': 0,
            'total_games': 0
        })

        for event in events:
            if event.is_conference_game:
                continue
            
            matchup_key = f"{event.home_conference}_vs_{event.away_conference}"
            stats_dict = matchup_stats[matchup_key]
            
            stats_dict['total_games'] += 1
            
            if event.team_penalized == 'home':
                stats_dict['home_penalties'] += 1
            else:
                stats_dict['away_penalties'] += 1
            
            if event.game_outcome == 'home_win':
                stats_dict['home_wins'] += 1
            else:
                stats_dict['away_wins'] += 1

        results = {}
        
        for matchup, stats_dict in matchup_stats.items():
            if stats_dict['total_games'] < 3:
                continue
            
            total_penalties = stats_dict['home_penalties'] + stats_dict['away_penalties']
            away_penalty_rate = stats_dict['away_penalties'] / total_penalties if total_penalties > 0 else 0.5
            
            home_win_rate = stats_dict['home_wins'] / stats_dict['total_games']
            
            results[matchup] = {
                'away_penalty_rate': away_penalty_rate,
                'home_win_rate': home_win_rate,
                'sample_size': stats_dict['total_games'],
                'total_penalties': total_penalties
            }

        return results

    def _analyze_critical_calls(self, events: List[OfficiatingEvent]) -> Dict:
        """
        Analyze officiating in critical situations
        - Late in game
        - Close score
        - High stakes plays
        """
        critical_events = [e for e in events if e.is_critical_play]
        
        if not critical_events:
            return {}
        
        conference_critical = defaultdict(lambda: {
            'home_favored': 0,
            'away_favored': 0,
            'neutral': 0,
            'total': 0
        })

        for event in critical_events:
            conf = event.officiating_conference
            stats_dict = conference_critical[conf]
            stats_dict['total'] += 1
            
            # Determine if call favored home or away
            # Penalty on away = favors home
            if event.team_penalized == 'away':
                stats_dict['home_favored'] += 1
            else:
                stats_dict['away_favored'] += 1

        results = {}
        
        for conf, stats_dict in conference_critical.items():
            if stats_dict['total'] < 5:
                continue
            
            home_favor_rate = stats_dict['home_favored'] / stats_dict['total']
            
            # Expected: 52% home favoring
            z_score = (home_favor_rate - 0.52) / (np.sqrt(0.52 * 0.48 / stats_dict['total'])) if stats_dict['total'] > 0 else 0
            
            if abs(z_score) > 2.0:
                significance = "HIGH"
                risk_score = 0.85
            elif abs(z_score) > 1.5:
                significance = "MODERATE"
                risk_score = 0.65
            else:
                significance = "LOW"
                risk_score = 0.35
            
            results[conf] = {
                'home_favor_rate': home_favor_rate,
                'significance': significance,
                'risk_score': risk_score,
                'z_score': z_score,
                'sample_size': stats_dict['total']
            }

        return results

    def _analyze_rivalry_games(self, events: List[OfficiatingEvent]) -> Dict:
        """
        Analyze officiating in rivalry games
        Often called tighter or differently
        """
        rivalry_events = [e for e in events if e.is_rivalry]
        
        if not rivalry_events:
            return {}
        
        conference_rivalry = defaultdict(lambda: {
            'home_penalties': 0,
            'away_penalties': 0,
            'games': 0,
            'total_penalties': 0
        })

        for event in rivalry_events:
            conf = event.officiating_conference
            stats_dict = conference_rivalry[conf]
            
            stats_dict['games'] += 1
            stats_dict['total_penalties'] += 1
            
            if event.team_penalized == 'home':
                stats_dict['home_penalties'] += 1
            else:
                stats_dict['away_penalties'] += 1

        results = {}
        
        for conf, stats_dict in conference_rivalry.items():
            if stats_dict['games'] < 2:
                continue
            
            total = stats_dict['home_penalties'] + stats_dict['away_penalties']
            away_penalty_rate = stats_dict['away_penalties'] / total if total > 0 else 0.5
            
            penalties_per_game = stats_dict['total_penalties'] / stats_dict['games']
            
            results[conf] = {
                'away_penalty_rate': away_penalty_rate,
                'penalties_per_game': penalties_per_game,
                'sample_size': stats_dict['games']
            }

        return results

    def _analyze_penalty_timing(self, events: List[OfficiatingEvent]) -> Dict:
        """
        Analyze when penalties are called
        - Early game vs late game
        - Before vs after halftime
        """
        timing_stats = defaultdict(lambda: {
            'q1': {'home': 0, 'away': 0},
            'q2': {'home': 0, 'away': 0},
            'q3': {'home': 0, 'away': 0},
            'q4': {'home': 0, 'away': 0}
        })

        for event in events:
            conf = event.officiating_conference
            quarter = f"q{event.quarter}"
            
            if event.team_penalized == 'home':
                timing_stats[conf][quarter]['home'] += 1
            else:
                timing_stats[conf][quarter]['away'] += 1

        results = {}
        
        for conf, quarters in timing_stats.items():
            results[conf] = {}
            
            for quarter, penalties in quarters.items():
                total = penalties['home'] + penalties['away']
                if total > 0:
                    away_rate = penalties['away'] / total
                    results[conf][quarter] = {
                        'away_penalty_rate': away_rate,
                        'total_penalties': total
                    }

        return results

    def _detect_statistical_anomalies(self, events: List[OfficiatingEvent]) -> List[Dict]:
        """
        Detect statistical anomalies that warrant attention
        """
        anomalies = []
        
        home_bias = self._analyze_home_bias(events)
        
        for conf, stats_dict in home_bias.items():
            if stats_dict['statistical_significance'] == 'significant':
                if abs(stats_dict['z_score']) > 2.5:
                    anomalies.append({
                        'type': 'EXTREME_HOME_BIAS',
                        'conference': conf,
                        'z_score': stats_dict['z_score'],
                        'p_value': stats_dict['p_value'],
                        'severity': 'HIGH',
                        'recommendation': 'AVOID_BETTING_ROAD_TEAMS' if stats_dict['z_score'] > 0 else 'TARGET_ROAD_TEAMS'
                    })

        return anomalies

    def get_betting_adjustments(self, home_conf: str, away_conf: str, 
                                officiating_conf: str, is_rivalry: bool = False) -> Dict:
        """
        Get betting adjustments based on officiating analysis
        
        Returns adjustments to apply to predictions
        """
        adjustments = {
            'spread_adjustment': 0.0,
            'confidence_adjustment': 0.0,
            'risk_score': 0.5,
            'recommendation': 'NEUTRAL'
        }

        # Load analysis results (would be pre-computed)
        # For now, return based on priors
        
        if officiating_conf in self.conference_priors:
            priors = self.conference_priors[officiating_conf]
            
            # Home bias adjustment
            home_bias_deviation = priors['home_bias'] - 0.52
            adjustments['spread_adjustment'] = home_bias_deviation * 3.0  # Convert to points
            
            # Cross-conference protection
            if officiating_conf == home_conf and home_conf != away_conf:
                adjustments['spread_adjustment'] += 1.5  # Favor home team
                adjustments['risk_score'] = 0.70
                adjustments['recommendation'] = 'FAVOR_HOME'
            elif officiating_conf == away_conf and home_conf != away_conf:
                adjustments['spread_adjustment'] -= 1.5  # Favor away team
                adjustments['risk_score'] = 0.70
                adjustments['recommendation'] = 'FAVOR_AWAY'
            
            # Rivalry adjustment
            if is_rivalry:
                adjustments['confidence_adjustment'] = -0.05  # Lower confidence (more chaos)
                adjustments['risk_score'] += 0.10

        return adjustments
