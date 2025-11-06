#!/usr/bin/env python3
"""
Enhanced Prop Vet Backtest
Supports stratified analysis by position and prop type
Longer backtests with historical data
"""

import json
import logging
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime
import sys
import glob
import os

import pandas as pd
from advanced_edge_detector import AdvancedEdgeDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedBacktester:
    """Enhanced backtest with advanced edge detection and stratified analysis"""
    
    def __init__(self):
        self.edge_detector = AdvancedEdgeDetector()
        self.results = []
        self.stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pushes': 0})
        self.position_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
        self.prop_type_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
        self.edge_type_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
    
    def load_props_data(self) -> List[Dict[str, Any]]:
        """Load prop data from latest files"""
        prop_files = glob.glob('./data/player_props/nfl_props*multisource*.parquet')
        
        if not prop_files:
            prop_files = glob.glob('./data/player_props/nfl_props*advanced*.parquet')
        
        if not prop_files:
            logger.error("❌ No prop files found")
            return []
        
        latest_file = max(prop_files, key=os.path.getctime)
        logger.info(f"✅ Loading props from: {latest_file}")
        
        df = pd.read_parquet(latest_file)
        return df.to_dict('records')
    
    def evaluate_prop(self, prop: Dict) -> Dict[str, Any]:
        """Evaluate a single prop with advanced edge detection"""
        
        if not prop.get('actual_value') or not prop.get('line'):
            return {'result': 'skip', 'correct': False}
        
        # Set default values for required fields
        player = {
            'name': prop.get('player_name', 'Unknown'),
            'position': prop.get('position', 'Unknown'),
            'line': prop.get('line'),
            'actual_value': prop.get('actual_value'),
            'avg_performance': prop.get('avg_performance', prop.get('line')),
            'std_dev': prop.get('std_dev', 15),
            'ceiling_performance': prop.get('ceiling_performance', prop.get('line', 0) * 1.25),
            'floor_performance': prop.get('floor_performance', prop.get('line', 0) * 0.7),
            'consistency_score': prop.get('consistency_score', 0.65),
        }
        
        # Detect edge
        edge_type, edge_strength, edge_info = self.edge_detector.detect_all_edges(player)
        
        if edge_type == 'skip' or edge_strength < 0.01:
            return {'result': 'skip', 'correct': False, 'edge_type': 'none'}
        
        # Determine bet
        bet_type = edge_info.get('bet_type', 'OVER').split()[0]
        
        # Evaluate result
        line = prop.get('line')
        actual = prop.get('actual_value')
        
        hit = (bet_type == 'OVER' and actual > line) or (bet_type == 'UNDER' and actual < line)
        push = actual == line
        
        if push:
            result = 'push'
            correct = True
        else:
            result = 'win' if hit else 'loss'
            correct = hit
        
        return {
            'result': result,
            'correct': correct,
            'edge_type': edge_type,
            'edge_strength': edge_strength,
            'bet_type': bet_type,
            'line': line,
            'actual': actual,
            'position': prop.get('position'),
            'prop_type': prop.get('prop_type'),
        }
    
    def backtest(self) -> Dict[str, Any]:
        """Run full backtest"""
        logger.info("🎯 Starting Enhanced Prop Vet Backtest")
        logger.info("=" * 70)
        
        props = self.load_props_data()
        if not props:
            logger.error("❌ No props loaded")
            return {}
        
        total_bets = 0
        total_wins = 0
        total_losses = 0
        total_pushes = 0
        
        for prop in props:
            result = self.evaluate_prop(prop)
            
            if result['result'] == 'skip':
                continue
            
            total_bets += 1
            position = prop.get('position', 'Unknown')
            prop_type = prop.get('prop_type', 'Unknown')
            edge_type = result.get('edge_type', 'unknown')
            
            if result['result'] == 'win':
                total_wins += 1
                logger.info(f"✅ WIN: {prop.get('player_name')} {prop_type} ({edge_type})")
            elif result['result'] == 'loss':
                total_losses += 1
                logger.warning(f"❌ LOSS: {prop.get('player_name')} {prop_type} ({edge_type})")
            else:
                total_pushes += 1
            
            # Track stats
            self.stats[edge_type]['wins' if result['correct'] else 'losses'] += 1
            self.position_stats[position]['wins' if result['result'] == 'win' else 'losses'] += 1
            self.prop_type_stats[prop_type]['wins' if result['result'] == 'win' else 'losses'] += 1
            self.edge_type_stats[edge_type]['wins' if result['result'] == 'win' else 'losses'] += 1
            
            self.results.append({
                'player': prop.get('player_name'),
                'position': position,
                'prop_type': prop_type,
                'line': result.get('line'),
                'actual': result.get('actual'),
                'bet_type': result.get('bet_type'),
                'result': result['result'],
                'edge_type': edge_type,
                'edge_strength': result.get('edge_strength'),
            })
        
        # Calculate metrics
        if total_bets > 0:
            win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
            roi = ((total_wins * 100) - (total_losses * 110)) / (total_bets * 110) * 100
        else:
            win_rate = 0
            roi = 0
        
        summary = {
            'total_props': len(props),
            'total_bets': total_bets,
            'wins': total_wins,
            'losses': total_losses,
            'pushes': total_pushes,
            'win_rate': f"{win_rate:.1%}",
            'roi': f"{roi:.1f}%",
            'edge_stats': dict(self.stats),
            'position_breakdown': dict(self.position_stats),
            'prop_type_breakdown': dict(self.prop_type_stats),
        }
        
        return summary
    
    def print_report(self, summary: Dict) -> None:
        """Print detailed backtest report"""
        print("\n" + "="*80)
        print("ENHANCED PROP VET BACKTEST REPORT")
        print("="*80)
        print(f"Total Props: {summary['total_props']}")
        print(f"Total Bets: {summary['total_bets']}")
        print(f"Wins: {summary['wins']}")
        print(f"Losses: {summary['losses']}")
        print(f"Pushes: {summary['pushes']}")
        print(f"Win Rate: {summary['win_rate']}")
        print(f"ROI: {summary['roi']}")
        
        print("\n📊 Edge Type Breakdown:")
        for edge_type, stats in summary['edge_stats'].items():
            total = stats['wins'] + stats['losses']
            if total > 0:
                edge_win_rate = stats['wins'] / total
                print(f"  {edge_type}: {stats['wins']}/{total} ({edge_win_rate:.1%})")
        
        print("\n🏈 Position Breakdown:")
        for position, stats in summary['position_breakdown'].items():
            total = stats['wins'] + stats['losses']
            if total > 0:
                position_win_rate = stats['wins'] / total
                print(f"  {position}: {stats['wins']}/{total} ({position_win_rate:.1%})")
        
        print("\n🎯 Prop Type Breakdown:")
        for prop_type, stats in summary['prop_type_breakdown'].items():
            total = stats['wins'] + stats['losses']
            if total > 0:
                prop_win_rate = stats['wins'] / total
                print(f"  {prop_type}: {stats['wins']}/{total} ({prop_win_rate:.1%})")
        
        print("="*80 + "\n")


def main():
    backtester = EnhancedBacktester()
    summary = backtester.backtest()
    
    if summary and summary.get('total_bets', 0) > 0:
        backtester.print_report(summary)
        
        # Save results
        output_file = 'backtest_enhanced_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': summary,
                'detailed_results': backtester.results[:100]
            }, f, indent=2)
        logger.info(f"📊 Results saved to {output_file}")
    else:
        logger.error("❌ Backtest produced no bets")
        sys.exit(1)


if __name__ == '__main__':
    main()
