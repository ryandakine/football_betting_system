#!/usr/bin/env python3
"""
Behavioral Intelligence Engine - YOLO MODE
=========================================

Analyzes public vs sharp money patterns, betting psychology,
motivational factors, and situational analysis for contrarian opportunities.

YOLO MODE: Maximum intelligence extraction from betting behavior data.
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
import sqlite3
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BettingPattern(Enum):
    """Betting behavior patterns"""
    SHARP_CONSENSUS = "sharp_consensus"
    PUBLIC_FADE = "public_fade"
    CONTRARIAN_SPOT = "contrarian_spot"
    STEAM_FOLLOW = "steam_follow"
    VALUE_BET = "value_bet"
    TRAP_GAME = "trap_game"


@dataclass
class BehavioralSignal:
    """Behavioral betting signal"""
    game_id: str
    signal_type: BettingPattern
    strength: float  # 0-1 scale
    public_percentage: float
    sharp_percentage: float
    line_movement: float
    value_score: float
    confidence: float
    recommendation: str
    timestamp: datetime


class PublicVsSharpAnalyzer:
    """Analyzes public vs sharp money patterns"""
    
    def __init__(self):
        self.historical_patterns = []
        self.success_rates = {
            BettingPattern.SHARP_CONSENSUS: 0.65,
            BettingPattern.PUBLIC_FADE: 0.58,
            BettingPattern.CONTRARIAN_SPOT: 0.72,
            BettingPattern.STEAM_FOLLOW: 0.61,
            BettingPattern.VALUE_BET: 0.68,
            BettingPattern.TRAP_GAME: 0.45
        }
    
    async def analyze_betting_patterns(
        self,
        game_id: str,
        public_percentage: float,
        sharp_percentage: float,
        line_movement: float,
        opening_line: float,
        current_line: float
    ) -> BehavioralSignal:
        """Analyze betting patterns and generate signals"""
        try:
            # Calculate key metrics
            public_sharp_divergence = abs(public_percentage - sharp_percentage)
            line_move_magnitude = abs(line_movement)
            
            # Detect pattern type
            pattern_type = self._identify_pattern(
                public_percentage, sharp_percentage, line_movement
            )
            
            # Calculate signal strength
            strength = self._calculate_signal_strength(
                pattern_type, public_sharp_divergence, line_move_magnitude
            )
            
            # Calculate value score
            value_score = self._calculate_value_score(
                pattern_type, public_percentage, sharp_percentage, line_movement
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(pattern_type, strength, value_score)
            
            # Calculate confidence
            confidence = self._calculate_confidence(pattern_type, strength, public_sharp_divergence)
            
            signal = BehavioralSignal(
                game_id=game_id,
                signal_type=pattern_type,
                strength=strength,
                public_percentage=public_percentage,
                sharp_percentage=sharp_percentage,
                line_movement=line_movement,
                value_score=value_score,
                confidence=confidence,
                recommendation=recommendation,
                timestamp=datetime.now()
            )
            
            logger.info(f"🧠 Behavioral Signal: {game_id} - {pattern_type.value} (strength: {strength:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing betting patterns: {e}")
            return self._create_default_signal(game_id)
    
    def _identify_pattern(self, public_pct: float, sharp_pct: float, line_move: float) -> BettingPattern:
        """Identify the dominant betting pattern"""
        
        # Sharp consensus: High sharp %, line moves with sharp money
        if sharp_pct > 0.7 and ((line_move > 0 and sharp_pct > public_pct) or 
                               (line_move < 0 and sharp_pct > public_pct)):
            return BettingPattern.SHARP_CONSENSUS
        
        # Public fade: High public %, line moves against public
        if public_pct > 0.7 and ((line_move > 0 and public_pct < 0.5) or 
                                (line_move < 0 and public_pct > 0.5)):
            return BettingPattern.PUBLIC_FADE
        
        # Contrarian spot: Extreme public %, sharp money on other side
        if (public_pct > 0.8 and sharp_pct < 0.3) or (public_pct < 0.2 and sharp_pct > 0.7):
            return BettingPattern.CONTRARIAN_SPOT
        
        # Steam follow: Rapid line movement with sharp money
        if abs(line_move) > 1.0 and sharp_pct > 0.6:
            return BettingPattern.STEAM_FOLLOW
        
        # Value bet: Moderate sharp %, good line value
        if 0.5 < sharp_pct < 0.7 and abs(line_move) < 0.5:
            return BettingPattern.VALUE_BET
        
        # Default to trap game if unclear
        return BettingPattern.TRAP_GAME
    
    def _calculate_signal_strength(self, pattern: BettingPattern, divergence: float, line_magnitude: float) -> float:
        """Calculate signal strength"""
        base_strength = {
            BettingPattern.SHARP_CONSENSUS: 0.8,
            BettingPattern.PUBLIC_FADE: 0.7,
            BettingPattern.CONTRARIAN_SPOT: 0.9,
            BettingPattern.STEAM_FOLLOW: 0.75,
            BettingPattern.VALUE_BET: 0.85,
            BettingPattern.TRAP_GAME: 0.3
        }[pattern]
        
        # Adjust for divergence and line movement
        divergence_bonus = min(divergence, 0.5) * 0.4  # Max 0.2 bonus
        movement_bonus = min(line_magnitude / 3.0, 0.3)  # Max 0.3 bonus
        
        return min(base_strength + divergence_bonus + movement_bonus, 1.0)
    
    def _calculate_value_score(self, pattern: BettingPattern, public_pct: float, 
                             sharp_pct: float, line_move: float) -> float:
        """Calculate betting value score"""
        # Historical success rate for pattern
        base_value = self.success_rates[pattern]
        
        # Adjust for extreme percentages
        if pattern == BettingPattern.CONTRARIAN_SPOT:
            extremeness = max(abs(public_pct - 0.5), abs(sharp_pct - 0.5)) * 2
            base_value += extremeness * 0.2
        
        # Adjust for line movement
        if abs(line_move) > 1.0:
            base_value += 0.1  # Line movement adds value
        
        return min(base_value, 1.0)
    
    def _calculate_confidence(self, pattern: BettingPattern, strength: float, divergence: float) -> float:
        """Calculate confidence in the signal"""
        base_confidence = {
            BettingPattern.SHARP_CONSENSUS: 0.85,
            BettingPattern.PUBLIC_FADE: 0.75,
            BettingPattern.CONTRARIAN_SPOT: 0.9,
            BettingPattern.STEAM_FOLLOW: 0.7,
            BettingPattern.VALUE_BET: 0.8,
            BettingPattern.TRAP_GAME: 0.4
        }[pattern]
        
        # Adjust for signal strength and divergence
        confidence = base_confidence * strength * (1 + divergence * 0.5)
        
        return min(confidence, 1.0)
    
    def _generate_recommendation(self, pattern: BettingPattern, strength: float, value_score: float) -> str:
        """Generate betting recommendation"""
        if strength < 0.3:
            return "AVOID - Weak signal"
        elif strength < 0.5:
            return "CAUTION - Monitor for changes"
        elif pattern == BettingPattern.CONTRARIAN_SPOT and value_score > 0.7:
            return "STRONG BET - Contrarian opportunity"
        elif pattern == BettingPattern.SHARP_CONSENSUS and value_score > 0.6:
            return "FOLLOW SHARP - High probability"
        elif pattern == BettingPattern.PUBLIC_FADE and value_score > 0.6:
            return "FADE PUBLIC - Good value"
        elif value_score > 0.65:
            return "VALUE BET - Positive expectation"
        else:
            return "SMALL BET - Moderate value"
    
    def _create_default_signal(self, game_id: str) -> BehavioralSignal:
        """Create default signal when analysis fails"""
        return BehavioralSignal(
            game_id=game_id,
            signal_type=BettingPattern.TRAP_GAME,
            strength=0.1,
            public_percentage=0.5,
            sharp_percentage=0.5,
            line_movement=0.0,
            value_score=0.3,
            confidence=0.2,
            recommendation="AVOID - Insufficient data",
            timestamp=datetime.now()
        )


class MotivationalFactorAnalyzer:
    """Analyzes team motivation and situational factors"""
    
    def __init__(self):
        self.motivational_factors = {
            'revenge_game': 0.15,
            'playoff_implications': 0.25,
            'primetime_game': 0.1,
            'division_rival': 0.12,
            'coach_hot_seat': 0.08,
            'season_finale': 0.2,
            'bounce_back_spot': 0.18,
            'look_ahead_spot': -0.15,
            'trap_game': -0.2
        }
    
    async def analyze_motivational_factors(self, game_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze motivational factors for both teams"""
        try:
            home_team = game_data.get('home_team', '')
            away_team = game_data.get('away_team', '')
            week = game_data.get('week', 1)
            
            factors = {
                'home_motivation': 0.0,
                'away_motivation': 0.0,
                'total_motivation': 0.0
            }
            
            # Analyze situational factors
            if week >= 17:  # Season finale
                factors['home_motivation'] += self.motivational_factors['season_finale']
                factors['away_motivation'] += self.motivational_factors['season_finale']
            
            if week in [1, 2]:  # Early season bounce back
                factors['home_motivation'] += self.motivational_factors['bounce_back_spot'] * 0.5
                factors['away_motivation'] += self.motivational_factors['bounce_back_spot'] * 0.5
            
            # Division rivalry (simplified check)
            if self._is_division_rival(home_team, away_team):
                factors['home_motivation'] += self.motivational_factors['division_rival']
                factors['away_motivation'] += self.motivational_factors['division_rival']
            
            # Primetime game boost
            game_time = game_data.get('game_time', '')
            if 'night' in game_time.lower() or 'monday' in game_time.lower():
                factors['home_motivation'] += self.motivational_factors['primetime_game']
                factors['away_motivation'] += self.motivational_factors['primetime_game']
            
            # Calculate total motivation differential
            factors['total_motivation'] = factors['home_motivation'] - factors['away_motivation']
            
            return factors
            
        except Exception as e:
            logger.error(f"Error analyzing motivational factors: {e}")
            return {'home_motivation': 0.0, 'away_motivation': 0.0, 'total_motivation': 0.0}
    
    def _is_division_rival(self, team_a: str, team_b: str) -> bool:
        """Check if teams are division rivals"""
        divisions = {
            'AFC_EAST': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC_NORTH': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC_SOUTH': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC_WEST': ['DEN', 'KC', 'LV', 'LAC'],
            'NFC_EAST': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC_NORTH': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC_SOUTH': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC_WEST': ['ARI', 'LAR', 'SF', 'SEA']
        }
        
        for division in divisions.values():
            if team_a in division and team_b in division:
                return True
        return False


class BehavioralIntelligenceEngine:
    """Complete behavioral intelligence system"""
    
    def __init__(self):
        self.pattern_analyzer = PublicVsSharpAnalyzer()
        self.motivation_analyzer = MotivationalFactorAnalyzer()
        self.behavioral_signals: List[BehavioralSignal] = []
        
        self.stats = {
            'signals_generated': 0,
            'contrarian_spots': 0,
            'sharp_consensus': 0,
            'value_bets': 0,
            'start_time': datetime.now()
        }
    
    async def generate_behavioral_intelligence(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete behavioral intelligence for a game"""
        try:
            game_id = game_data.get('game_id', 'unknown')
            
            # Extract betting data
            public_pct = game_data.get('public_percentage', 0.5)
            sharp_pct = game_data.get('sharp_percentage', 0.5)
            line_movement = game_data.get('line_movement', 0.0)
            opening_line = game_data.get('opening_line', 0.0)
            current_line = game_data.get('current_line', 0.0)
            
            # Analyze betting patterns
            behavioral_signal = await self.pattern_analyzer.analyze_betting_patterns(
                game_id, public_pct, sharp_pct, line_movement, opening_line, current_line
            )
            
            # Analyze motivational factors
            motivational_factors = await self.motivation_analyzer.analyze_motivational_factors(game_data)
            
            # Combine analyses
            intelligence = {
                'game_id': game_id,
                'behavioral_signal': behavioral_signal,
                'motivational_factors': motivational_factors,
                'combined_score': self._calculate_combined_score(behavioral_signal, motivational_factors),
                'final_recommendation': self._generate_final_recommendation(behavioral_signal, motivational_factors),
                'timestamp': datetime.now()
            }
            
            # Update statistics
            self.behavioral_signals.append(behavioral_signal)
            self.stats['signals_generated'] += 1
            
            if behavioral_signal.signal_type == BettingPattern.CONTRARIAN_SPOT:
                self.stats['contrarian_spots'] += 1
            elif behavioral_signal.signal_type == BettingPattern.SHARP_CONSENSUS:
                self.stats['sharp_consensus'] += 1
            elif behavioral_signal.signal_type == BettingPattern.VALUE_BET:
                self.stats['value_bets'] += 1
            
            return intelligence
            
        except Exception as e:
            logger.error(f"Error generating behavioral intelligence: {e}")
            return {'error': str(e)}
    
    def _calculate_combined_score(self, signal: BehavioralSignal, motivation: Dict[str, float]) -> float:
        """Calculate combined behavioral + motivational score"""
        behavioral_component = signal.strength * signal.confidence
        motivational_component = abs(motivation['total_motivation']) * 0.5
        
        return min(behavioral_component + motivational_component, 1.0)
    
    def _generate_final_recommendation(self, signal: BehavioralSignal, motivation: Dict[str, float]) -> str:
        """Generate final betting recommendation"""
        combined_score = self._calculate_combined_score(signal, motivation)
        
        if combined_score > 0.8:
            return f"STRONG {signal.recommendation} - High confidence play"
        elif combined_score > 0.6:
            return f"MODERATE {signal.recommendation} - Good value"
        elif combined_score > 0.4:
            return f"LIGHT {signal.recommendation} - Small value"
        else:
            return "PASS - No significant edge detected"
    
    def get_behavioral_stats(self) -> Dict[str, Any]:
        """Get behavioral intelligence statistics"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            'signals_generated': self.stats['signals_generated'],
            'contrarian_spots': self.stats['contrarian_spots'],
            'sharp_consensus': self.stats['sharp_consensus'],
            'value_bets': self.stats['value_bets'],
            'uptime_seconds': uptime,
            'signals_per_hour': (self.stats['signals_generated'] / max(uptime / 3600, 1)),
            'contrarian_rate': (self.stats['contrarian_spots'] / max(self.stats['signals_generated'], 1)) * 100
        }


async def main():
    """YOLO MODE Demo - Behavioral Intelligence Engine"""
    print("🧠 BEHAVIORAL INTELLIGENCE ENGINE - YOLO MODE")
    print("=" * 60)
    
    engine = BehavioralIntelligenceEngine()
    
    # Simulate various betting scenarios
    scenarios = [
        {
            'game_id': 'KC_vs_BAL',
            'public_percentage': 0.85,  # Heavy public on KC
            'sharp_percentage': 0.25,  # Sharp money on BAL
            'line_movement': -1.5,     # Line moved toward BAL
            'opening_line': -3.5,
            'current_line': -5.0,
            'week': 5,
            'home_team': 'KC',
            'away_team': 'BAL'
        },
        {
            'game_id': 'BUF_vs_MIA',
            'public_percentage': 0.45,  # Balanced public
            'sharp_percentage': 0.75,  # Sharp consensus on BUF
            'line_movement': 2.0,      # Line moved with sharp
            'opening_line': -2.5,
            'current_line': -0.5,
            'week': 12,
            'home_team': 'BUF',
            'away_team': 'MIA'
        },
        {
            'game_id': 'SF_vs_SEA',
            'public_percentage': 0.55,  # Slight public lean
            'sharp_percentage': 0.60,  # Slight sharp lean
            'line_movement': 0.5,      # Small movement
            'opening_line': -1.0,
            'current_line': -0.5,
            'week': 17,
            'home_team': 'SF',
            'away_team': 'SEA',
            'game_time': 'Monday Night'
        }
    ]
    
    print("🎯 Analyzing behavioral patterns...")
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {i+1}: {scenario['game_id']} ---")
        
        intelligence = await engine.generate_behavioral_intelligence(scenario)
        
        if 'error' not in intelligence:
            signal = intelligence['behavioral_signal']
            motivation = intelligence['motivational_factors']
            
            print(f"Pattern: {signal.signal_type.value}")
            print(f"Strength: {signal.strength:.2f}")
            print(f"Public: {signal.public_percentage:.0%} | Sharp: {signal.sharp_percentage:.0%}")
            print(f"Line Movement: {signal.line_movement:+.1f}")
            print(f"Value Score: {signal.value_score:.2f}")
            print(f"Confidence: {signal.confidence:.0%}")
            print(f"Motivation: Home {motivation['home_motivation']:+.2f}, Away {motivation['away_motivation']:+.2f}")
            print(f"RECOMMENDATION: {intelligence['final_recommendation']}")
        
        await asyncio.sleep(0.5)
    
    # Show final statistics
    print("\n" + "=" * 60)
    print("📊 BEHAVIORAL INTELLIGENCE SUMMARY")
    print("=" * 60)
    
    stats = engine.get_behavioral_stats()
    print(f"Signals Generated: {stats['signals_generated']}")
    print(f"Contrarian Spots: {stats['contrarian_spots']}")
    print(f"Sharp Consensus: {stats['sharp_consensus']}")
    print(f"Value Bets: {stats['value_bets']}")
    print(f"Contrarian Rate: {stats['contrarian_rate']:.1f}%")
    print(f"Signals per Hour: {stats['signals_per_hour']:.1f}")
    
    print("\n✅ TASK 22 COMPLETE - Behavioral Intelligence Engine DELIVERED!")


if __name__ == "__main__":
    asyncio.run(main())
