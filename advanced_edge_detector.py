#!/usr/bin/env python3
"""
Advanced Edge Detection Algorithms
Improved algorithms for identifying exploitable prop pricing inefficiencies
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class AdvancedEdgeDetector:
    """Advanced edge detection with multiple algorithms"""
    
    def __init__(self):
        self.min_edge_threshold = 0.02  # 2% minimum edge
    
    def correlation_edge(self, player: Dict, related_players: List[Dict]) -> Dict[str, Any]:
        """
        Detect correlated prop opportunities
        Vegas prices props independently but many are correlated
        """
        player_name = player.get('name', 'Unknown')
        
        # Calculate implied correlation from related player props
        if not related_players or len(related_players) < 2:
            return {'edge_type': 'none', 'edge_strength': 0}
        
        # High-variance player in shooting game -> more volume for teammates
        try:
            # Get baseline consistency
            player_consistency = player.get('consistency_score', 0.65)
            player_ceiling = player.get('ceiling_performance', player.get('line', 0))
            player_line = player.get('line', 0)
            
            # If this player has high ceiling and low consistency, teammates benefit
            ceiling_ratio = player_ceiling / max(player_line, 1) if player_line > 0 else 1.0
            
            if ceiling_ratio > 1.25 and player_consistency < 0.70:
                # In blowout games, bench players get fewer touches
                # But with high variance player, game stays close -> more volume
                edge = min(0.08, (ceiling_ratio - 1.2) * 0.5)
                
                logger.info(f"🔗 CORRELATION EDGE: {player_name} high ceiling attracts defensive attention, teammates benefit")
                return {
                    'edge_type': 'correlation',
                    'edge_strength': edge,
                    'signal': f'Game flow dependency: {ceiling_ratio:.2f}x ceiling ratio',
                    'bet_type': 'OVER (volume from game flow)'
                }
        except Exception as e:
            logger.debug(f"Correlation calc error: {e}")
        
        return {'edge_type': 'none', 'edge_strength': 0}
    
    def implied_volatility_edge(self, player: Dict) -> Dict[str, Any]:
        """
        Calculate implied volatility from historical performance
        High volatility players have +EV OVER bets when variance is underpriced
        """
        player_name = player.get('name', 'Unknown')
        
        try:
            avg_performance = player.get('avg_performance', player.get('line', 0))
            std_dev = player.get('std_dev', 15)
            line = player.get('line', 0)
            
            if line <= 0 or avg_performance <= 0:
                return {'edge_type': 'none', 'edge_strength': 0}
            
            # Calculate coefficient of variation
            cv = std_dev / avg_performance
            
            # Volatility regimes:
            # Low vol (CV < 0.12): Consistent players, edges in reliability
            # Medium vol (0.12-0.20): Normal variance, standard pricing
            # High vol (0.20-0.35): Explosive upside, Vegas underprices ceiling
            # Extreme vol (>0.35): Lottery ticket syndrome, public overprices overs
            
            if 0.20 <= cv <= 0.35:
                # Sweet spot for variance exploitation
                # Vegas prices at mean, but volatility favors overs
                
                # Calculate skewness effect
                # Right-skewed distributions favor overs
                ceiling = player.get('ceiling_performance', avg_performance * 1.3)
                floor = player.get('floor_performance', avg_performance * 0.7)
                
                skew = (ceiling - avg_performance) - (avg_performance - floor)
                skew_ratio = skew / std_dev if std_dev > 0 else 0
                
                if skew_ratio > 0.1:  # Positive skew
                    edge = 0.06 + (cv - 0.20) * 0.5 + min(0.03, skew_ratio * 0.1)
                    
                    logger.info(f"💥 IMPLIED VOL EDGE: {player_name} - CV:{cv:.2f}, Skew:{skew_ratio:.2f}")
                    return {
                        'edge_type': 'implied_volatility',
                        'edge_strength': min(0.10, edge),
                        'signal': f'High volatility with positive skew: CV={cv:.2f}',
                        'bet_type': 'OVER (ceiling outweighs floor)'
                    }
            
            elif cv < 0.12:
                # Ultra-consistent players have edge in UNDER when underrated
                edge = 0.04 + (0.12 - cv) * 0.2
                
                logger.info(f"✅ CONSISTENCY EDGE: {player_name} - very reliable, CV={cv:.2f}")
                return {
                    'edge_type': 'implied_volatility',
                    'edge_strength': edge,
                    'signal': f'Ultra-consistent: CV={cv:.2f}',
                    'bet_type': 'UNDER (safe floor)'
                }
        
        except Exception as e:
            logger.debug(f"IV calc error: {e}")
        
        return {'edge_type': 'none', 'edge_strength': 0}
    
    def matchup_advantage_edge(self, player: Dict, opponent: Dict) -> Dict[str, Any]:
        """
        Advanced matchup analysis
        Specific defender-receiver/RB combinations create edges
        """
        player_name = player.get('name', 'Unknown')
        position = player.get('position', 'Unknown')
        
        try:
            # Matchup-specific weakness detection
            player_avg = player.get('avg_performance', player.get('line', 0))
            defense_rank = opponent.get('defense_rank', 16)
            
            # Defense quality: 1=elite, 16=worst
            # Better matchup (higher defense rank) = easier
            
            if defense_rank > 14:  # Bottom 2 defenses
                # Expect significant outperformance
                expected_bonus = 1.2 + (defense_rank - 14) * 0.05
                ceiling = player.get('ceiling_performance', player_avg * 1.3)
                line = player.get('line', 0)
                
                if line > 0 and ceiling > line:
                    edge = min(0.08, (expected_bonus - 1.0) * 0.5)
                    
                    if edge > self.min_edge_threshold:
                        logger.info(f"🎯 MATCHUP EDGE: {player_name} vs bottom-5 defense")
                        return {
                            'edge_type': 'matchup_advantage',
                            'edge_strength': edge,
                            'signal': f'Elite matchup: Defense Rank {defense_rank}',
                            'bet_type': 'OVER (weak defense)'
                        }
            
            elif defense_rank <= 5:  # Elite defenses
                # Expect significant underperformance
                expected_penalty = 0.75 - (5 - defense_rank) * 0.05
                floor = player.get('floor_performance', player_avg * 0.7)
                line = player.get('line', 0)
                
                if line > floor:
                    edge = min(0.07, (1.0 - expected_penalty) * 0.4)
                    
                    if edge > self.min_edge_threshold:
                        logger.info(f"🛡️  ELITE DEFENSE EDGE: {player_name} vs elite D")
                        return {
                            'edge_type': 'matchup_advantage',
                            'edge_strength': edge,
                            'signal': f'Elite defense: Rank {defense_rank}',
                            'bet_type': 'UNDER (elite defense)'
                        }
        
        except Exception as e:
            logger.debug(f"Matchup calc error: {e}")
        
        return {'edge_type': 'none', 'edge_strength': 0}
    
    def momentum_edge(self, player: Dict, team_data: Dict = None) -> Dict[str, Any]:
        """
        Detect momentum-based edges
        Recent performance changes create temporary mispricings
        """
        player_name = player.get('name', 'Unknown')
        
        try:
            avg_performance = player.get('avg_performance', player.get('line', 0))
            recent_trend = player.get('recent_trend', 0)  # -1 to +1
            
            if abs(recent_trend) < 0.1 or recent_trend is None:
                return {'edge_type': 'none', 'edge_strength': 0}
            
            # Overreaction to recent trends creates edges
            if recent_trend > 0.2:  # Positive recent trend
                # Vegas may have slowly incorporated recent success
                # Sharp bettors already priced it in, but public lags
                edge = min(0.05, abs(recent_trend) * 0.2)
                
                logger.info(f"📈 MOMENTUM EDGE: {player_name} - positive recent trend")
                return {
                    'edge_type': 'momentum',
                    'edge_strength': edge,
                    'signal': f'Positive momentum: {recent_trend:+.1%}',
                    'bet_type': 'OVER (riding momentum)'
                }
            
            elif recent_trend < -0.2:  # Negative recent trend
                # Overreaction to recent bad games
                edge = min(0.04, abs(recent_trend) * 0.15)
                
                logger.info(f"📉 REGRESSION EDGE: {player_name} - negative trend reversion")
                return {
                    'edge_type': 'momentum',
                    'edge_strength': edge,
                    'signal': f'Negative momentum (reversion play): {recent_trend:+.1%}',
                    'bet_type': 'OVER (mean reversion)'
                }
        
        except Exception as e:
            logger.debug(f"Momentum calc error: {e}")
        
        return {'edge_type': 'none', 'edge_strength': 0}
    
    def line_movement_edge(self, player: Dict) -> Dict[str, Any]:
        """
        Detect sharp vs public line divergence
        Closing line value indicates where informed bettors are
        """
        player_name = player.get('name', 'Unknown')
        
        try:
            opening_line = player.get('opening_line')
            closing_line = player.get('line', opening_line)
            
            if not opening_line or not closing_line:
                return {'edge_type': 'none', 'edge_strength': 0}
            
            line_move = closing_line - opening_line
            
            # Significant line movements signal smart money
            if abs(line_move) > 5:
                direction = 'OVER' if line_move > 0 else 'UNDER'
                edge = min(0.06, abs(line_move) / 50 * 0.1)
                
                logger.info(f"📊 LINE MOVE EDGE: {player_name} - {line_move:+.1f} point move")
                return {
                    'edge_type': 'line_movement',
                    'edge_strength': edge,
                    'signal': f'Significant line move: {line_move:+.1f} points',
                    'bet_type': direction
                }
        
        except Exception as e:
            logger.debug(f"Line movement calc error: {e}")
        
        return {'edge_type': 'none', 'edge_strength': 0}
    
    def detect_all_edges(self, player: Dict, opponent: Dict = None, 
                        related_players: List[Dict] = None) -> Tuple[str, float, Dict]:
        """Run all edge detection methods and return strongest edge"""
        
        edges = [
            self.implied_volatility_edge(player),
            self.matchup_advantage_edge(player, opponent or {}),
            self.correlation_edge(player, related_players or []),
            self.momentum_edge(player),
            self.line_movement_edge(player),
        ]
        
        # Find strongest edge
        best_edge = max(edges, key=lambda x: x.get('edge_strength', 0))
        
        if best_edge.get('edge_strength', 0) < self.min_edge_threshold:
            return 'skip', 0, {}
        
        edge_type = best_edge.get('edge_type', 'unknown')
        edge_strength = best_edge.get('edge_strength', 0)
        
        return edge_type, edge_strength, best_edge
