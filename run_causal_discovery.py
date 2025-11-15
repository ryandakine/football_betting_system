#!/usr/bin/env python3
"""
NCAA Causal Discovery Runner
=============================
Discovers causal relationships in NCAA game data using PC/GES algorithms.
Run this once to analyze your historical data (2015-2024).
Results are cached for instant loading on future runs.
"""

import sys
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from college_football_system.causal_discovery.ncaa_causal_learner import (
    NCAACousalLearner,
    discover_causality_from_file
)


def main():
    """Run causal discovery on NCAA data."""
    print("\n" + "="*70)
    print("🔍 NCAA CAUSAL DISCOVERY")
    print("="*70 + "\n")
    
    # Try to load historical data
    data_path = Path('data/ncaa_history_2015_2024.csv')
    
    if not data_path.exists():
        print(f"⚠️  Historical data not found at {data_path}")
        print("\nTo run causal discovery, you need:")
        print("  1. CSV file with 10 years of NCAA games (2015-2024)")
        print("  2. Columns: weather, injuries, rest_days, team_stats, game_outcome")
        print("\nRunning demo with synthetic data instead...\n")
        run_demo()
        return
    
    print(f"📂 Loading data from {data_path}...")
    
    try:
        # Run causal discovery
        results = discover_causality_from_file(
            str(data_path),
            output_dir='models/causal_models',
            method='pc'  # PC is faster than GES
        )
        
        print("\n" + "="*70)
        print("✅ CAUSAL DISCOVERY COMPLETE")
        print("="*70)
        
        print(f"\nDiscovered Relationships:")
        print(f"  Direct Causal Edges: {results.get('edge_count', 0)}")
        print(f"  Confounders: {results.get('confounder_count', 0)}")
        print(f"  Mediation Paths: {results.get('mediator_count', 0)}")
        print(f"  Total: {results.get('total_relationships', 0)}")
        
        if results.get('direct_causal_edges'):
            print(f"\nTop Causal Edges:")
            for edge in results['direct_causal_edges'][:5]:
                print(f"  {edge['cause']} → {edge['effect']}")
        
        print(f"\n💾 Results cached to models/causal_models/")
        print(f"   - causal_graph.pkl")
        print(f"   - causal_paths.json\n")
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        run_demo()


def run_demo():
    """Run demo with synthetic NCAA data."""
    print("📊 Running Demo with Synthetic Data...\n")
    
    # Create synthetic NCAA data
    np_data = {
        'temperature': [45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
        'wind_speed': [5, 8, 10, 12, 15, 18, 20, 22, 25, 28],
        'home_team_rest': [3, 4, 7, 3, 4, 7, 3, 4, 7, 3],
        'away_team_rest': [3, 3, 3, 4, 4, 4, 7, 7, 7, 3],
        'key_injuries_home': [0, 1, 2, 0, 1, 0, 2, 1, 0, 1],
        'key_injuries_away': [1, 0, 1, 1, 0, 2, 1, 0, 1, 0],
        'home_passing_yards': [250, 280, 200, 300, 220, 260, 180, 290, 310, 270],
        'away_passing_yards': [240, 260, 240, 220, 300, 240, 220, 280, 250, 310],
        'home_total_yards': [380, 420, 300, 450, 350, 400, 280, 440, 480, 420],
        'away_total_yards': [360, 380, 360, 340, 420, 360, 340, 420, 390, 450],
        'total_points': [45, 52, 38, 56, 48, 52, 42, 64, 70, 58],
        'home_score': [24, 28, 20, 32, 26, 29, 22, 38, 42, 35],
    }
    
    data = pd.DataFrame(np_data)
    
    learner = NCAACousalLearner(cache_dir='models/causal_models')
    
    print("Running PC algorithm...")
    results = learner.discover_causality(data, method='pc', confidence_level=0.05)
    
    print("\n" + "="*70)
    print("✅ DEMO CAUSAL DISCOVERY COMPLETE")
    print("="*70)
    
    print(f"\nDiscovered Relationships:")
    print(f"  Direct Causal Edges: {results.get('edge_count', 0)}")
    print(f"  Confounders: {results.get('confounder_count', 0)}")
    print(f"  Mediation Paths: {results.get('mediator_count', 0)}")
    
    if results.get('direct_causal_edges'):
        print(f"\nExample Causal Edges:")
        for edge in results['direct_causal_edges'][:5]:
            print(f"  {edge['cause']} → {edge['effect']}")
    
    print(f"\n💾 Demo results cached to models/causal_models/\n")


if __name__ == '__main__':
    main()
