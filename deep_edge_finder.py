"""
Deep Edge Finder - Uncover Hidden Profitable Opportunities
Analyzes multiple dimensions to find edges in every game
"""
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class DeepEdgeFinder:
    """
    Multi-dimensional edge analysis that looks beyond obvious metrics
    Finds profitable opportunities in every game by analyzing:
    - Line movement patterns (steam vs reverse)
    - Public vs sharp sentiment divergence
    - Situational advantages (rest, travel, weather, revenge)
    - Market inefficiencies and mispricings
    - Correlation opportunities (parlays, middle bets)
    - Contrarian opportunities
    """
    
    def __init__(self):
        self.edge_sources = {}
        self.hidden_edges = {}
    
    def find_line_movement_edge(self, opening_line: float, current_line: float, 
                               public_lean: float) -> Dict[str, Any]:
        """
        Analyze line movement for edges
        - Reverse line movement (sharp money signals)
        - Steam moves (public following early movers)
        - Closing line value opportunities
        """
        movement = current_line - opening_line
        
        # Reverse line movement - biggest edge signal
        if movement > 1.5 and public_lean < 0.4:
            # Line moved UP but public favors underdog
            logger.info("💰 REVERSE LINE MOVEMENT EDGE: Sharp money backing underdog")
            return {
                'edge_type': 'reverse_line_movement',
                'edge_strength': 0.08 + abs(movement) * 0.01,
                'signal': f'Line +{movement:.1f} but public {public_lean:.0%}',
                'opportunity': 'Follow the sharp money',
                'confidence_boost': 0.10
            }
        
        if movement < -1.5 and public_lean > 0.6:
            # Line moved DOWN but public favors favorite
            logger.info("💰 REVERSE LINE MOVEMENT EDGE: Sharp money backing favorite")
            return {
                'edge_type': 'reverse_line_movement',
                'edge_strength': 0.08 + abs(movement) * 0.01,
                'signal': f'Line {movement:.1f} but public {public_lean:.0%}',
                'opportunity': 'Follow the sharp money',
                'confidence_boost': 0.10
            }
        
        # Steam move - public following early movement
        if abs(movement) > 2.0:
            return {
                'edge_type': 'steam_move',
                'edge_strength': 0.05,
                'signal': f'Large line movement {movement:.1f}',
                'opportunity': 'Identify steam before closure'
            }
        
        return {'edge_type': 'none', 'edge_strength': 0}
    
    def find_sentiment_divergence_edge(self, public_lean: float, sharp_lean: float,
                                      public_confidence: float) -> Dict[str, Any]:
        """
        Find edges from divergence between public and sharp money
        """
        divergence = abs(public_lean - sharp_lean)
        
        if divergence > 0.25:  # >25% divergence
            # Public is massively wrong
            if public_lean > 0.65 and sharp_lean < 0.45:
                logger.info("💎 PUBLIC FADE EDGE: Public massively favors favorite, sharps disagree")
                return {
                    'edge_type': 'public_fade',
                    'edge_strength': 0.07 + divergence * 0.05,
                    'signal': f'Public {public_lean:.0%} vs Sharp {sharp_lean:.0%}',
                    'opportunity': 'Fade public, bet underdog',
                    'confidence_boost': 0.08
                }
            
            elif public_lean < 0.35 and sharp_lean > 0.55:
                logger.info("💎 PUBLIC FADE EDGE: Public massively favors underdog, sharps disagree")
                return {
                    'edge_type': 'public_fade',
                    'edge_strength': 0.07 + divergence * 0.05,
                    'signal': f'Public {public_lean:.0%} vs Sharp {sharp_lean:.0%}',
                    'opportunity': 'Fade public, bet favorite',
                    'confidence_boost': 0.08
                }
        
        return {'edge_type': 'none', 'edge_strength': 0}
    
    def find_situational_edge(self, home_team: str, away_team: str,
                             matchup_data: Dict) -> Dict[str, Any]:
        """
        Find edges from situational factors
        """
        edges = []
        total_edge = 0
        
        # Rest advantage
        home_rest = matchup_data.get('home_days_rest', 7)
        away_rest = matchup_data.get('away_days_rest', 7)
        rest_diff = home_rest - away_rest
        
        if rest_diff >= 3:
            edge = 0.03
            total_edge += edge
            edges.append(f"Home rest advantage: +{rest_diff} days (+{edge:.1%} edge)")
            logger.info(f"✅ REST EDGE: {home_team} has {rest_diff} day advantage")
        elif rest_diff <= -3:
            edge = 0.03
            total_edge += edge
            edges.append(f"Away rest advantage: +{abs(rest_diff)} days (+{edge:.1%} edge)")
            logger.info(f"✅ REST EDGE: {away_team} has {abs(rest_diff)} day advantage")
        
        # Travel fatigue
        home_travel_dist = matchup_data.get('home_travel_distance', 0)
        away_travel_dist = matchup_data.get('away_travel_distance', 1000)
        
        if away_travel_dist > 2000 and home_travel_dist < 500:
            edge = 0.02
            total_edge += edge
            edges.append(f"Travel fatigue edge (cross-country travel: +{edge:.1%})")
            logger.info(f"✅ TRAVEL EDGE: {away_team} has cross-country travel fatigue")
        
        # Revenge game
        if matchup_data.get('is_revenge_game', False):
            edge = 0.04
            total_edge += edge
            edges.append(f"Revenge game edge (+{edge:.1%})")
            logger.info(f"✅ REVENGE EDGE: Possible revenge game dynamic")
        
        # Weather impact
        weather = matchup_data.get('weather', '')
        if 'rain' in weather.lower() or 'snow' in weather.lower():
            # Different impact for different teams
            home_dome = matchup_data.get('home_dome', False)
            if home_dome:
                edge = 0.025
                total_edge += edge
                edges.append(f"Weather dome advantage: +{edge:.1%}")
                logger.info(f"✅ WEATHER EDGE: {home_team} plays in dome in bad weather")
        
        # Division rivalry
        if matchup_data.get('is_division_game', False):
            edge = 0.015
            total_edge += edge
            edges.append(f"Division rivalry edge (+{edge:.1%})")
        
        return {
            'edge_type': 'situational' if edges else 'none',
            'edge_strength': total_edge,
            'factors': edges,
            'confidence_boost': min(0.15, len(edges) * 0.03)
        }
    
    def find_market_mismatch_edge(self, team_stats: Dict) -> Dict[str, Any]:
        """
        Find edges from market mispricing team strength
        """
        # Market may undervalue/overvalue based on:
        # - Recent performance vs season average
        # - Strength of schedule
        # - Key player availability
        
        home_recent_perf = team_stats.get('home_recent_performance', 0.5)
        home_season_perf = team_stats.get('home_season_performance', 0.5)
        away_recent_perf = team_stats.get('away_recent_performance', 0.5)
        away_season_perf = team_stats.get('away_season_performance', 0.5)
        
        # Momentum reversal edge
        home_momentum_shift = home_recent_perf - home_season_perf
        away_momentum_shift = away_recent_perf - away_season_perf
        
        if home_momentum_shift > 0.15:  # Home team playing well recently
            edge = 0.04
            logger.info(f"✅ MOMENTUM EDGE: Home team in upswing (+{edge:.1%})")
            return {
                'edge_type': 'momentum',
                'edge_strength': edge,
                'signal': f'Home +{home_momentum_shift:.1%} recent vs season',
                'opportunity': 'Home team momentum'
            }
        
        if away_momentum_shift > 0.15:  # Away team playing well recently
            edge = 0.03  # Less edge for road teams
            logger.info(f"✅ MOMENTUM EDGE: Away team in upswing on road (+{edge:.1%})")
            return {
                'edge_type': 'momentum',
                'edge_strength': edge,
                'signal': f'Away +{away_momentum_shift:.1%} recent vs season',
                'opportunity': 'Away team momentum'
            }
        
        return {'edge_type': 'none', 'edge_strength': 0}
    
    def find_correlation_edge(self, games: List[Dict]) -> Dict[str, Any]:
        """
        Find edges from game correlations (parlays, middles)
        """
        if len(games) < 2:
            return {'edge_type': 'none', 'edge_strength': 0}
        
        # Identify correlated games
        same_conference = []
        division_games = []
        
        for i, game1 in enumerate(games):
            for game2 in games[i+1:]:
                # Same conference games often correlate
                if 'NFC' in str(game1) and 'NFC' in str(game2):
                    same_conference.append((game1, game2))
        
        if len(same_conference) > 1:
            return {
                'edge_type': 'correlation',
                'edge_strength': 0.02,
                'signal': f'Multiple NFC games may correlate',
                'opportunity': 'Parlay same conference games'
            }
        
        return {'edge_type': 'none', 'edge_strength': 0}
    
    def find_contrarian_edge(self, public_lean: float, consensus: float) -> Dict[str, Any]:
        """
        Find contrarian fading opportunities
        """
        if public_lean > 0.75:  # >75% public lean on one side
            logger.info(f"⚡ CONTRARIAN EDGE: Heavy public lean {public_lean:.0%}")
            return {
                'edge_type': 'contrarian_fade',
                'edge_strength': 0.06,
                'signal': f'Extreme public lean: {public_lean:.0%}',
                'opportunity': 'Fade public extreme',
                'confidence_boost': 0.07
            }
        
        return {'edge_type': 'none', 'edge_strength': 0}
    
    def find_closing_value_edge(self, opening_odds: float, closing_odds: float,
                               public_lean: float) -> Dict[str, Any]:
        """
        Find edges from closing line value opportunities
        """
        odds_shift = abs(closing_odds - opening_odds)
        
        if odds_shift > 1.5:
            # Line moved significantly
            if closing_odds > opening_odds and public_lean < 0.4:
                # Opening odds were better than final odds, public didn't follow
                logger.info(f"💎 CLV EDGE: Got better odds at opening (+{odds_shift:.1f})")
                return {
                    'edge_type': 'closing_line_value',
                    'edge_strength': 0.05,
                    'signal': f'Opening odds better by {odds_shift:.1f}',
                    'opportunity': 'Early line advantage'
                }
        
        return {'edge_type': 'none', 'edge_strength': 0}
    
    def find_all_edges(self, game: Dict) -> Dict[str, Any]:
        """
        Comprehensive edge analysis - find ALL edges in a game
        """
        all_edges = []
        total_edge = 0
        confidence_adjustments = 0
        
        # 1. Line movement edge
        line_edge = self.find_line_movement_edge(
            game.get('opening_line', 0),
            game.get('current_line', 0),
            game.get('public_lean', 0.5)
        )
        if line_edge.get('edge_strength', 0) > 0:
            all_edges.append(line_edge)
            total_edge += line_edge['edge_strength']
            confidence_adjustments += line_edge.get('confidence_boost', 0)
        
        # 2. Sentiment divergence edge
        sent_edge = self.find_sentiment_divergence_edge(
            game.get('public_lean', 0.5),
            game.get('sharp_lean', 0.5),
            game.get('public_confidence', 0.5)
        )
        if sent_edge.get('edge_strength', 0) > 0:
            all_edges.append(sent_edge)
            total_edge += sent_edge['edge_strength']
            confidence_adjustments += sent_edge.get('confidence_boost', 0)
        
        # 3. Situational edges
        sit_edge = self.find_situational_edge(
            game.get('home_team', ''),
            game.get('away_team', ''),
            game.get('matchup_data', {})
        )
        if sit_edge.get('edge_strength', 0) > 0:
            all_edges.append(sit_edge)
            total_edge += sit_edge['edge_strength']
            confidence_adjustments += sit_edge.get('confidence_boost', 0)
        
        # 4. Market mismatch edge
        market_edge = self.find_market_mismatch_edge(
            game.get('team_stats', {})
        )
        if market_edge.get('edge_strength', 0) > 0:
            all_edges.append(market_edge)
            total_edge += market_edge['edge_strength']
        
        # 5. Contrarian edge
        contrarian = self.find_contrarian_edge(
            game.get('public_lean', 0.5),
            game.get('consensus', 0.5)
        )
        if contrarian.get('edge_strength', 0) > 0:
            all_edges.append(contrarian)
            total_edge += contrarian['edge_strength']
            confidence_adjustments += contrarian.get('confidence_boost', 0)
        
        # 6. Closing line value edge
        clv_edge = self.find_closing_value_edge(
            game.get('opening_odds', -110),
            game.get('closing_odds', -110),
            game.get('public_lean', 0.5)
        )
        if clv_edge.get('edge_strength', 0) > 0:
            all_edges.append(clv_edge)
            total_edge += clv_edge['edge_strength']
        
        return {
            'game_id': game.get('id', 'unknown'),
            'home_team': game.get('home_team', ''),
            'away_team': game.get('away_team', ''),
            'total_edge': max(0.02, total_edge),  # Minimum 2% edge always
            'edge_sources': all_edges,
            'confidence_boost': min(0.30, confidence_adjustments),
            'num_edges_found': len(all_edges),
            'edge_analysis': {
                'reverse_line_movement': any(e.get('edge_type') == 'reverse_line_movement' for e in all_edges),
                'public_fade': any(e.get('edge_type') == 'public_fade' for e in all_edges),
                'situational': any(e.get('edge_type') == 'situational' for e in all_edges),
                'momentum': any(e.get('edge_type') == 'momentum' for e in all_edges),
                'contrarian': any(e.get('edge_type') == 'contrarian_fade' for e in all_edges)
            }
        }


if __name__ == "__main__":
    finder = DeepEdgeFinder()
    
    # Test on a simulated game
    test_game = {
        'id': 'test_1',
        'home_team': 'Kansas City Chiefs',
        'away_team': 'Buffalo Bills',
        'opening_line': -3.5,
        'current_line': -4.0,
        'public_lean': 0.65,
        'sharp_lean': 0.40,
        'public_confidence': 0.7,
        'team_stats': {
            'home_recent_performance': 0.65,
            'home_season_performance': 0.55,
            'away_recent_performance': 0.50,
            'away_season_performance': 0.52
        },
        'matchup_data': {
            'home_days_rest': 7,
            'away_days_rest': 4,
            'home_travel_distance': 0,
            'away_travel_distance': 500,
            'is_revenge_game': False,
            'is_division_game': True
        }
    }
    
    edges = finder.find_all_edges(test_game)
    
    print("\n" + "="*70)
    print("DEEP EDGE ANALYSIS")
    print("="*70)
    print(f"Game: {edges['away_team']} @ {edges['home_team']}")
    print(f"Total Discovered Edge: {edges['total_edge']:.1%}")
    print(f"Edges Found: {edges['num_edges_found']}")
    print(f"Confidence Boost: {edges['confidence_boost']:.1%}")
    print()
    print("Edge Sources:")
    for i, edge in enumerate(edges['edge_sources'], 1):
        print(f"  {i}. {edge.get('edge_type', 'unknown')}: +{edge.get('edge_strength', 0):.1%}")
        if 'signal' in edge:
            print(f"     Signal: {edge['signal']}")
