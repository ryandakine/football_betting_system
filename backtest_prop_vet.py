#!/usr/bin/env python3
"""
Prop Vet Exploit Engine Backtest
Tests the engine's ability to identify exploitable prop pricing inefficiencies
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from datetime import datetime
import sys

from prop_vet_exploit_engine import PropVetExploitEngine
from real_data_loader import RealDataLoader, RealDataLoadError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PropVetBacktester:
    """Backtests prop vet exploit engine against historical data"""
    
    def __init__(self):
        self.engine = PropVetExploitEngine()
        self.results = []
        self.stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pushes': 0})
        
    def load_historical_games(self) -> List[Dict[str, Any]]:
        """Load real scraped player prop data"""
        try:
            # Load the most recent scraped props parquet file
            import glob
            import os
            prop_files = glob.glob('./data/player_props/nfl_props_advanced_*.parquet')
            if not prop_files:
                logger.error(f"❌ No scraped prop files found in ./data/player_props/")
                sys.exit(1)
            
            latest_file = max(prop_files, key=os.path.getctime)
            logger.info(f"✅ Loading real scraped props from: {latest_file}")
            
            import pandas as pd
            df = pd.read_parquet(latest_file)
            games_data = []
            
            # Group props by game
            for game_id, game_group in df.groupby('game_id'):
                game = {
                    'id': game_id,
                    'matchup': game_group.iloc[0].get('game_matchup', 'Unknown') if len(game_group) > 0 else 'Unknown',
                    'home_props': game_group.to_dict('records'),
                    'away_props': [],
                    'total': 48,
                    'spread': 0,
                }
                games_data.append(game)
            
            logger.info(f"✅ Loaded {len(df)} real prop records from {len(games_data)} games")
            return games_data
        except Exception as e:
            logger.error(f"❌ Error loading props: {e}")
            sys.exit(1)
    
    def extract_player_data(self, game: Dict) -> List[Dict[str, Any]]:
        """Extract player prop data from game"""
        players = []
        
        # Get game metadata
        game_meta = {
            'game_id': game.get('id') or game.get('game_id'),
            'date': game.get('date'),
            'home_team': game.get('home_team'),
            'away_team': game.get('away_team'),
            'game_total': game.get('total') or game.get('game_total', 48),
            'spread': game.get('spread', 0),
            'is_primetime': game.get('is_primetime', False),
        }
        
        # Extract home and away props
        for team_key, team_name in [('home', game.get('home_team')), ('away', game.get('away_team'))]:
            team_props = game.get(f'{team_key}_props', [])
            
            for prop in team_props:
                # Ensure all required fields exist with defaults
                player_data = {
                    **game_meta,
                    'team': team_name or prop.get('team', 'Unknown'),
                    'is_home': team_key == 'home',
                    'name': prop.get('player_name'),
                    'position': prop.get('position'),
                    'prop_type': prop.get('prop_type'),
                    'line': prop.get('line'),
                    'over_odds': prop.get('over_odds'),
                    'under_odds': prop.get('under_odds'),
                    'actual_value': prop.get('actual_value'),
                    'implied_prob': self.odds_to_prob(prop.get('over_odds')),
                    'vs_specific_defense': prop.get('vs_defense_stats', {}),
                    'avg_performance': prop.get('avg_performance', prop.get('line', 0)),
                    'std_dev': prop.get('std_dev', 15),  # Default variance
                    'ceiling_performance': prop.get('ceiling_performance', prop.get('line', 0) * 1.25),
                    'consistency_score': prop.get('consistency_score', 0.65),
                    'win_probability': prop.get('win_probability') or prop.get('team_win_prob', 0.5),
                    'implied_team_score': prop.get('implied_team_score') or prop.get('implied_score', 24),
                }
                players.append(player_data)
        
        return players
    
    def odds_to_prob(self, odds: float) -> float:
        """Convert American odds to implied probability"""
        if odds is None:
            return 0.5
        if odds > 0:
            return 100 / (100 + odds)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    def evaluate_prop_edge(self, player: Dict) -> Tuple[str, float, Dict]:
        """Run all edge detection methods on a player prop"""
        if not player.get('actual_value') or not player.get('line'):
            return 'skip', 0, {}
        
        # Get opponent defense
        opponent_name = player.get('away_team') if player.get('is_home') else player.get('home_team')
        opponent = {'name': opponent_name, 'defense_rank': 16}
        
        # Run edge detection
        matchup_edge = self.engine.find_matchup_anomaly_edge(player, opponent)
        variance_edge = self.engine.find_variance_exploitation_edge(player)
        script_edge = self.engine.find_game_script_edge(
            {'win_probability': player.get('win_probability'), 
             'implied_team_score': player.get('implied_team_score'),
             'game_total': player.get('game_total')},
            opponent,
            player
        )
        
        # Select strongest edge
        edges = [
            (matchup_edge, 'matchup'),
            (variance_edge, 'variance'),
            (script_edge, 'script')
        ]
        
        best_edge = max(edges, key=lambda x: x[0].get('edge_strength', 0))
        
        if best_edge[0].get('edge_strength', 0) < 0.01:
            return 'skip', 0, {}
        
        return best_edge[1], best_edge[0].get('edge_strength', 0), best_edge[0]
    
    def evaluate_bet(self, player: Dict, edge_type: str, edge_strength: float, 
                    edge_info: Dict) -> Dict[str, Any]:
        """Determine bet direction and if it hit"""
        if edge_type == 'skip':
            return {'result': 'skip', 'correct': False}
        
        actual = player.get('actual_value')
        line = player.get('line')
        
        if actual is None or line is None:
            return {'result': 'skip', 'correct': False}
        
        # Determine bet direction from edge
        bet_type = edge_info.get('bet_type', 'OVER').split()[0]  # Extract OVER/UNDER
        
        # Check if bet was correct
        hit = (bet_type == 'OVER' and actual > line) or (bet_type == 'UNDER' and actual < line)
        push = actual == line
        
        if push:
            result = 'push'
            correct = True  # Pushes are neutral
        else:
            result = 'win' if hit else 'loss'
            correct = hit
        
        return {
            'result': result,
            'correct': correct,
            'bet_type': bet_type,
            'line': line,
            'actual': actual,
            'hit': hit,
            'edge_strength': edge_strength,
            'edge_type': edge_type
        }
    
    def backtest(self) -> Dict[str, Any]:
        """Run full backtest"""
        logger.info("🎯 Starting Prop Vet Exploit Engine Backtest")
        
        games = self.load_historical_games()
        if not games:
            logger.error("❌ No games loaded")
            return {}
        
        total_bets = 0
        total_wins = 0
        total_losses = 0
        total_pushes = 0
        roi = 0
        
        for game in games:
            game_date = game.get('date', 'unknown')
            players = self.extract_player_data(game)
            
            for player in players:
                edge_type, edge_strength, edge_info = self.evaluate_prop_edge(player)
                
                if edge_type == 'skip':
                    continue
                
                bet_result = self.evaluate_bet(player, edge_type, edge_strength, edge_info)
                
                if bet_result['result'] == 'skip':
                    continue
                
                total_bets += 1
                player_name = player.get('name', 'Unknown')
                prop_type = player.get('prop_type', 'unknown')
                
                if bet_result['result'] == 'win':
                    total_wins += 1
                    logger.info(f"✅ WIN: {player_name} {prop_type} ({edge_type})")
                elif bet_result['result'] == 'loss':
                    total_losses += 1
                    logger.warning(f"❌ LOSS: {player_name} {prop_type} ({edge_type})")
                else:
                    total_pushes += 1
                
                self.stats[edge_type]['wins' if bet_result['correct'] else 'losses'] += 1
                
                self.results.append({
                    'game_date': game_date,
                    'player': player_name,
                    'position': player.get('position'),
                    'prop_type': prop_type,
                    'line': bet_result['line'],
                    'actual': bet_result['actual'],
                    'bet_type': bet_result['bet_type'],
                    'result': bet_result['result'],
                    'edge_type': edge_type,
                    'edge_strength': edge_strength,
                })
        
        # Calculate metrics
        if total_bets > 0:
            win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
            # Assume -110 odds (standard)
            roi = ((total_wins * 100) - (total_losses * 110)) / (total_bets * 110) * 100
        else:
            win_rate = 0
        
        summary = {
            'total_games': len(games),
            'total_bets': total_bets,
            'wins': total_wins,
            'losses': total_losses,
            'pushes': total_pushes,
            'win_rate': f"{win_rate:.1%}",
            'roi': f"{roi:.1f}%",
            'edge_stats': dict(self.stats),
        }
        
        return summary
    
    def print_report(self, summary: Dict) -> None:
        """Print detailed backtest report"""
        print("\n" + "="*80)
        print("PROP VET EXPLOIT ENGINE BACKTEST REPORT")
        print("="*80)
        print(f"Games Analyzed: {summary['total_games']}")
        print(f"Total Bets: {summary['total_bets']}")
        print(f"Wins: {summary['wins']}")
        print(f"Losses: {summary['losses']}")
        print(f"Pushes: {summary['pushes']}")
        print(f"Win Rate: {summary['win_rate']}")
        print(f"ROI: {summary['roi']}")
        print("\nEdge Type Breakdown:")
        for edge_type, stats in summary['edge_stats'].items():
            total = stats['wins'] + stats['losses']
            if total > 0:
                edge_win_rate = stats['wins'] / total
                print(f"  {edge_type}: {stats['wins']}/{total} ({edge_win_rate:.1%})")
        print("="*80 + "\n")


def main():
    backtester = PropVetBacktester()
    summary = backtester.backtest()
    
    if summary:
        backtester.print_report(summary)
        
        # Save results
        output_file = 'prop_vet_backtest_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': summary,
                'detailed_results': backtester.results[:100]  # First 100 for brevity
            }, f, indent=2)
        logger.info(f"📊 Results saved to {output_file}")
    else:
        logger.error("❌ Backtest failed - no data")
        sys.exit(1)


if __name__ == '__main__':
    main()
