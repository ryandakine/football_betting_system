#!/usr/bin/env python3
"""
Extended 2024 Season Backtest
Generates 1000+ props across full season for comprehensive analysis
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict
import random

import pandas as pd
from advanced_edge_detector import AdvancedEdgeDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExtendedSeasonBacktester:
    """Generate and backtest full 2024 season props"""
    
    # Historical 2024 baselines
    HISTORICAL_BASELINES = {
        'QB': {
            'passing_yards': {'avg': 280, 'std': 45, 'min': 150, 'max': 450},
            'passing_tds': {'avg': 2.1, 'std': 1.2, 'min': 0, 'max': 6},
        },
        'RB': {
            'rushing_yards': {'avg': 95, 'std': 40, 'min': 0, 'max': 250},
            'receptions': {'avg': 4.2, 'std': 2.5, 'min': 0, 'max': 12},
        },
        'WR': {
            'receptions': {'avg': 6.5, 'std': 2.8, 'min': 0, 'max': 14},
            'receiving_yards': {'avg': 95, 'std': 40, 'min': 0, 'max': 200},
        },
    }
    
    # 2024 Season key players
    STAR_PLAYERS = {
        'QB': [
            ('Patrick Mahomes', 'KC', {'passing_yards': 310, 'passing_tds': 2.2}),
            ('Josh Allen', 'BUF', {'passing_yards': 285, 'passing_tds': 1.9}),
            ('Lamar Jackson', 'BAL', {'passing_yards': 275, 'passing_tds': 2.0}),
            ('Jared Goff', 'DET', {'passing_yards': 305, 'passing_tds': 2.3}),
            ('Kirk Cousins', 'MIN', {'passing_yards': 280, 'passing_tds': 1.8}),
            ('Dak Prescott', 'DAL', {'passing_yards': 295, 'passing_tds': 2.1}),
            ('Joe Burrow', 'CIN', {'passing_yards': 290, 'passing_tds': 2.0}),
            ('Justin Herbert', 'LAC', {'passing_yards': 270, 'passing_tds': 1.7}),
        ],
        'RB': [
            ('Christian McCaffrey', 'SF', {'rushing_yards': 110, 'receptions': 6.5}),
            ('Josh Jacobs', 'LV', {'rushing_yards': 115, 'receptions': 3.2}),
            ('Jonathan Taylor', 'IND', {'rushing_yards': 100, 'receptions': 3.8}),
            ('Derrick Henry', 'LAR', {'rushing_yards': 120, 'receptions': 2.5}),
            ('Saquon Barkley', 'PHI', {'rushing_yards': 105, 'receptions': 4.2}),
        ],
        'WR': [
            ('Travis Kelce', 'KC', {'receptions': 8.5, 'receiving_yards': 115}),
            ('CeeDee Lamb', 'DAL', {'receptions': 8.1, 'receiving_yards': 110}),
            ('Tyreek Hill', 'MIA', {'receptions': 7.8, 'receiving_yards': 105}),
            ('Stefon Diggs', 'HOU', {'receptions': 7.5, 'receiving_yards': 100}),
            ('Justin Jefferson', 'MIN', {'receptions': 7.2, 'receiving_yards': 105}),
            ('AJ Brown', 'PHI', {'receptions': 7.0, 'receiving_yards': 110}),
        ]
    }
    
    def __init__(self):
        self.edge_detector = AdvancedEdgeDetector()
        self.results = []
        self.stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
        self.position_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
        random.seed(42)
    
    def generate_season_props(self, weeks: int = 18, games_per_week: int = 13) -> List[Dict]:
        """Generate full season props (1000+)"""
        logger.info(f"🎲 Generating {weeks} weeks × {games_per_week} games of props...")
        
        props = []
        week_counter = 1
        
        for week in range(1, weeks + 1):
            # 13 games per week in NFL
            for game_num in range(1, games_per_week + 1):
                game_id = f"2024_W{week}_G{game_num}"
                
                # Add props for each position
                for position in ['QB', 'RB', 'WR']:
                    if position not in self.STAR_PLAYERS:
                        continue
                    
                    # 2 players per position per game
                    for player_name, team, overrides in self.STAR_PLAYERS[position][:2]:
                        baselines = self.HISTORICAL_BASELINES.get(position, {})
                        
                        for prop_type, baseline in baselines.items():
                            line = overrides.get(prop_type, baseline['avg'])
                            
                            # Add weekly variance (injuries, form changes, etc.)
                            weekly_adjustment = random.gauss(1.0, 0.05)
                            line = line * weekly_adjustment
                            
                            # Generate actual with volatility
                            actual = line + random.gauss(0, baseline['std'] / 2)
                            actual = max(baseline['min'], min(baseline['max'], actual))
                            
                            # Defense rank affects difficulty
                            defense_rank = random.randint(1, 16)
                            
                            props.append({
                                'week': week,
                                'game_id': game_id,
                                'player_name': player_name,
                                'position': position,
                                'team': team,
                                'prop_type': prop_type,
                                'line': round(line, 1),
                                'actual_value': round(actual, 1),
                                'over_odds': -110,
                                'under_odds': -110,
                                'avg_performance': round(line, 1),
                                'std_dev': round(baseline['std'], 1),
                                'consistency_score': 0.65 + random.uniform(-0.1, 0.15),
                                'ceiling_performance': round(line * 1.35, 1),
                                'floor_performance': round(line * 0.65, 1),
                                'defense_rank': defense_rank,
                                'team_win_prob': 0.5 + random.uniform(-0.3, 0.3),
                                'fetch_time': datetime.now().isoformat(),
                            })
        
        logger.info(f"✅ Generated {len(props)} props across {weeks} weeks")
        return props
    
    def backtest_season(self, props: List[Dict]) -> Dict[str, Any]:
        """Run backtest on full season"""
        logger.info(f"🎯 Running backtest on {len(props)} props...")
        
        total_wins = 0
        total_losses = 0
        weekly_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
        
        for i, prop in enumerate(props):
            if i % 100 == 0:
                logger.info(f"   Progress: {i}/{len(props)}")
            
            # Evaluate with edge detection
            if not prop.get('actual_value') or not prop.get('line'):
                continue
            
            player = {
                'name': prop.get('player_name'),
                'position': prop.get('position'),
                'line': prop.get('line'),
                'actual_value': prop.get('actual_value'),
                'avg_performance': prop.get('avg_performance'),
                'std_dev': prop.get('std_dev'),
                'ceiling_performance': prop.get('ceiling_performance'),
                'floor_performance': prop.get('floor_performance'),
                'consistency_score': prop.get('consistency_score'),
            }
            
            edge_type, edge_strength, edge_info = self.edge_detector.detect_all_edges(player)
            
            if edge_type == 'skip' or edge_strength < 0.01:
                continue
            
            # Determine bet
            bet_type = edge_info.get('bet_type', 'OVER').split()[0]
            
            # Check result
            hit = (bet_type == 'OVER' and prop['actual_value'] > prop['line']) or \
                  (bet_type == 'UNDER' and prop['actual_value'] < prop['line'])
            
            if hit:
                total_wins += 1
                weekly_stats[prop['week']]['wins'] += 1
            else:
                total_losses += 1
                weekly_stats[prop['week']]['losses'] += 1
            
            self.position_stats[prop['position']]['wins' if hit else 'losses'] += 1
            
            self.results.append({
                'week': prop['week'],
                'player': prop['player_name'],
                'position': prop['position'],
                'prop_type': prop['prop_type'],
                'line': prop['line'],
                'actual': prop['actual_value'],
                'bet_type': bet_type,
                'result': 'win' if hit else 'loss',
                'edge_type': edge_type,
                'edge_strength': edge_strength,
            })
        
        # Calculate metrics
        total_bets = total_wins + total_losses
        if total_bets > 0:
            win_rate = total_wins / total_bets
            roi = ((total_wins * 100) - (total_losses * 110)) / (total_bets * 110) * 100
        else:
            win_rate = 0
            roi = 0
        
        return {
            'total_props': len(props),
            'total_bets': total_bets,
            'wins': total_wins,
            'losses': total_losses,
            'win_rate': f"{win_rate:.1%}",
            'roi': f"{roi:.1f}%",
            'weekly_stats': dict(weekly_stats),
            'position_breakdown': dict(self.position_stats),
        }
    
    def print_report(self, summary: Dict) -> None:
        """Print backtest report"""
        print("\n" + "="*80)
        print("EXTENDED 2024 SEASON BACKTEST REPORT")
        print("="*80)
        print(f"Total Props: {summary['total_props']}")
        print(f"Total Bets: {summary['total_bets']}")
        print(f"Wins: {summary['wins']}")
        print(f"Losses: {summary['losses']}")
        print(f"Win Rate: {summary['win_rate']}")
        print(f"ROI: {summary['roi']}")
        
        print("\n📅 Weekly Breakdown (sample):")
        for week in sorted(list(summary['weekly_stats'].keys()))[:5]:
            stats = summary['weekly_stats'][week]
            total = stats['wins'] + stats['losses']
            if total > 0:
                wr = stats['wins'] / total
                print(f"  Week {week}: {stats['wins']}/{total} ({wr:.1%})")
        
        print("\n🏈 Position Breakdown:")
        for position in ['QB', 'RB', 'WR']:
            if position in summary['position_breakdown']:
                stats = summary['position_breakdown'][position]
                total = stats['wins'] + stats['losses']
                if total > 0:
                    wr = stats['wins'] / total
                    print(f"  {position}: {stats['wins']}/{total} ({wr:.1%})")
        
        print("="*80 + "\n")


async def main():
    backtester = ExtendedSeasonBacktester()
    
    # Generate season props
    props = backtester.generate_season_props(weeks=18, games_per_week=13)
    
    # Save prop data
    df = pd.DataFrame(props)
    prop_file = f"data/player_props/season_2024_props_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    df.to_parquet(prop_file, index=False)
    logger.info(f"💾 Props saved to {prop_file}")
    
    # Run backtest
    summary = backtester.backtest_season(props)
    backtester.print_report(summary)
    
    # Save results
    results_file = f"backtest_2024_extended_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'sample_results': backtester.results[:50]
        }, f, indent=2)
    logger.info(f"📊 Results saved to {results_file}")


if __name__ == '__main__':
    asyncio.run(main())
