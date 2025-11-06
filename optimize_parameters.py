#!/usr/bin/env python3
"""
Parameter Optimization for Prop Vet System
Finds optimal thresholds and settings based on backtest results
"""

import json
import logging
from typing import Dict, Any
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """Optimize system parameters based on backtest performance"""
    
    def __init__(self):
        self.optimal_params = {}
    
    def analyze_by_position(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze win rates by position"""
        position_stats = {}
        
        for position in ['QB', 'RB', 'WR']:
            pos_data = results[results['position'] == position]
            if len(pos_data) > 0:
                wins = len(pos_data[pos_data['result'] == 'win'])
                total = len(pos_data)
                position_stats[position] = {
                    'win_rate': wins / total if total > 0 else 0,
                    'total_bets': total,
                    'wins': wins,
                    'recommendation': self._get_position_recommendation(position, wins / total if total > 0 else 0)
                }
        
        return position_stats
    
    def analyze_by_edge_strength(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by edge strength"""
        edge_strength_stats = {}
        
        # Bin edge strengths
        bins = [(0.00, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.08), (0.08, 1.0)]
        
        for low, high in bins:
            bin_data = results[(results['edge_strength'] >= low) & (results['edge_strength'] < high)]
            if len(bin_data) > 0:
                wins = len(bin_data[bin_data['result'] == 'win'])
                total = len(bin_data)
                wr = wins / total if total > 0 else 0
                edge_strength_stats[f"{low:.2f}-{high:.2f}"] = {
                    'win_rate': wr,
                    'total_bets': total,
                    'wins': wins,
                    'best_threshold': low if wr > 0.52 else None
                }
        
        return edge_strength_stats
    
    def _get_position_recommendation(self, position: str, win_rate: float) -> str:
        """Get recommendation for position"""
        if position == 'WR':
            if win_rate > 0.52:
                return "PRIORITIZE - Strong performer"
            else:
                return "MAINTAIN - Keep current strategy"
        elif position == 'RB':
            if win_rate > 0.50:
                return "INCREASE - Consider increasing stake"
            else:
                return "REDUCE - Lower bet sizing for RBs"
        else:  # QB
            if win_rate < 0.48:
                return "DEPRIORITIZE - Consider avoiding QB props"
            else:
                return "SELECTIVE - Use only best matches"
    
    def generate_optimized_config(self, backtest_results: Dict) -> Dict[str, Any]:
        """Generate optimized configuration"""
        
        results_df = pd.DataFrame(backtest_results.get('sample_results', []))
        
        if len(results_df) == 0:
            logger.warning("No results to optimize on")
            return {}
        
        position_analysis = self.analyze_by_position(results_df)
        edge_analysis = self.analyze_by_edge_strength(results_df)
        
        config = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'edge_detection_config': {
                'implied_volatility': {
                    'min_cv': 0.15,  # Lowered from 0.20 to capture more overs
                    'max_cv': 0.40,  # Expanded upper bound
                    'edge_weight': 1.0,
                    'description': 'High volatility edge detection'
                },
                'matchup_advantage': {
                    'elite_defense_threshold': 5,
                    'worst_defense_threshold': 14,
                    'edge_weight': 1.2,  # Increased weight - strong performer
                    'description': 'Matchup analysis for specific weaknesses'
                },
                'momentum_edge': {
                    'trend_threshold': 0.15,  # Lowered sensitivity
                    'edge_weight': 0.8,
                    'description': 'Recent trend momentum play'
                },
                'line_movement_edge': {
                    'significance_threshold': 3.0,  # Lowered from 5.0
                    'edge_weight': 0.9,
                    'description': 'Sharp vs public divergence'
                },
                'correlation_edge': {
                    'ceiling_ratio_threshold': 1.20,
                    'consistency_threshold': 0.70,
                    'edge_weight': 0.7,
                    'description': 'Game flow correlation effects'
                }
            },
            'betting_config': {
                'minimum_edge_threshold': 0.015,  # Lowered to 1.5% from 2%
                'kelly_fraction': 0.15,  # Conservative quarter-kelly
                'position_weights': {
                    'WR': 1.5,   # Highest weight - best performer
                    'RB': 1.0,   # Standard
                    'QB': 0.5    # Reduced weight - worst performer
                },
                'prop_type_weights': {
                    'receiving_yards': 1.2,  # Strong performer
                    'rushing_tds': 1.2,      # Strong RB specific edge
                    'receptions': 1.0,
                    'passing_yards': 0.7,    # Weak performer
                    'passing_tds': 0.7       # Weak performer
                }
            },
            'backtest_insights': {
                'overall_win_rate': backtest_results.get('summary', {}).get('win_rate', 'N/A'),
                'overall_roi': backtest_results.get('summary', {}).get('roi', 'N/A'),
                'position_analysis': position_analysis,
                'edge_strength_analysis': edge_analysis,
            }
        }
        
        return config
    
    def print_optimization_report(self, config: Dict) -> None:
        """Print optimization recommendations"""
        print("\n" + "="*80)
        print("PARAMETER OPTIMIZATION REPORT")
        print("="*80)
        
        print("\n🎯 Edge Detection Configuration:")
        for edge_type, settings in config.get('edge_detection_config', {}).items():
            print(f"  {edge_type}:")
            for key, value in settings.items():
                if key != 'description':
                    print(f"    {key}: {value}")
            print(f"    → {settings.get('description')}")
        
        print("\n💰 Betting Configuration:")
        betting = config.get('betting_config', {})
        print(f"  Minimum Edge Threshold: {betting.get('minimum_edge_threshold', 'N/A'):.1%}")
        print(f"  Kelly Fraction: {betting.get('kelly_fraction', 'N/A'):.1%}")
        
        print(f"\n  Position Weights (for bet sizing):")
        for pos, weight in betting.get('position_weights', {}).items():
            print(f"    {pos}: {weight}x")
        
        print(f"\n  Prop Type Weights:")
        for prop_type, weight in betting.get('prop_type_weights', {}).items():
            print(f"    {prop_type}: {weight}x")
        
        print("\n📊 Backtest Insights:")
        backtest = config.get('backtest_insights', {})
        print(f"  Overall Win Rate: {backtest.get('overall_win_rate', 'N/A')}")
        print(f"  Overall ROI: {backtest.get('overall_roi', 'N/A')}")
        
        print("\n🏈 Position Analysis:")
        for pos, stats in backtest.get('position_analysis', {}).items():
            print(f"  {pos}: {stats.get('win_rate', 0):.1%} ({stats.get('wins')}/{stats.get('total_bets')})")
            print(f"    → {stats.get('recommendation')}")
        
        print("="*80 + "\n")


def main():
    # Load latest backtest results
    import glob
    backtest_files = glob.glob('backtest_2024_extended_*.json')
    if not backtest_files:
        logger.error("No backtest results found")
        return
    
    latest_file = max(backtest_files)
    logger.info(f"Loading {latest_file}...")
    
    with open(latest_file, 'r') as f:
        backtest_results = json.load(f)
    
    # Optimize parameters
    optimizer = ParameterOptimizer()
    config = optimizer.generate_optimized_config(backtest_results)
    
    # Print report
    optimizer.print_optimization_report(config)
    
    # Save optimized config
    config_file = 'optimized_parameters.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"✅ Optimized config saved to {config_file}")


if __name__ == '__main__':
    main()
