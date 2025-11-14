#!/usr/bin/env python3
"""
NCAA Causal Discovery Runner
=============================
Runs causal discovery on 2015-2024 NCAA historical data.

This is a one-time setup (or run once per season) to discover
causal relationships in your 10 years of NCAA game data.
"""

import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Direct import to avoid __init__.py dependencies
sys.path.insert(0, str(Path(__file__).parent))
from college_football_system.causal_discovery.ncaa_causal_learner import NCAACousalLearner

print("🔍 NCAA CAUSAL DISCOVERY")
print("=" * 80)
print()


def load_ncaa_games_data(start_year=2015, end_year=2024):
    """Load and combine NCAA game data from multiple years."""
    print(f"📂 Loading NCAA data from {start_year} to {end_year}...")

    all_games = []
    data_dir = Path("data/football/historical/ncaaf")

    for year in range(start_year, end_year + 1):
        games_file = data_dir / f"ncaaf_{year}_games.json"
        stats_file = data_dir / f"ncaaf_{year}_stats.json"

        if not games_file.exists():
            print(f"   ⚠️  {year}: Games file not found, skipping")
            continue

        try:
            with open(games_file) as f:
                games = json.load(f)

            # Load stats if available
            stats = {}
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)

            print(f"   ✅ {year}: Loaded {len(games)} games")
            all_games.extend(games)

        except Exception as e:
            print(f"   ❌ {year}: Error loading - {e}")
            continue

    print(f"\n✅ Total games loaded: {len(all_games)}")
    return all_games


def extract_causal_features(games):
    """
    Extract features relevant to causal discovery.

    Features:
    - Weather: temperature, wind_speed, precipitation
    - Rest: days_rest_home, days_rest_away
    - Performance: points_scored, points_allowed, total_yards
    - Game context: spread, total, is_rivalry, conference_game
    - Outcomes: final_score_home, final_score_away, cover
    """
    print("🔧 Extracting causal features...")

    data = []

    for game in games:
        # Skip games with missing critical data
        if not game.get('home_team') or not game.get('away_team'):
            continue

        # Extract features
        feature_dict = {
            # Weather (if available)
            'temperature': game.get('weather', {}).get('temperature', 65),
            'wind_speed': game.get('weather', {}).get('wind_speed', 0),
            'precipitation': 1 if game.get('weather', {}).get('condition', '').lower() in ['rain', 'snow'] else 0,

            # Home team performance
            'home_points': game.get('home_points', 0),
            'home_yards': game.get('home_stats', {}).get('total_yards', 0),
            'home_turnovers': game.get('home_stats', {}).get('turnovers', 0),
            'home_penalties': game.get('home_stats', {}).get('penalties', 0),

            # Away team performance
            'away_points': game.get('away_points', 0),
            'away_yards': game.get('away_stats', {}).get('total_yards', 0),
            'away_turnovers': game.get('away_stats', {}).get('turnovers', 0),
            'away_penalties': game.get('away_stats', {}).get('penalties', 0),

            # Game context
            'spread': game.get('spread', 0),
            'total': game.get('total', 0),
            'is_conference_game': 1 if game.get('conference_game', False) else 0,
            'is_rivalry': 1 if game.get('rivalry', False) else 0,

            # Outcomes
            'total_points': game.get('home_points', 0) + game.get('away_points', 0),
            'point_differential': game.get('home_points', 0) - game.get('away_points', 0),
            'home_covered': 1 if (game.get('home_points', 0) - game.get('away_points', 0)) > game.get('spread', 0) else 0,
        }

        data.append(feature_dict)

    df = pd.DataFrame(data)

    # Remove rows with too many missing values
    df = df.dropna(thresh=len(df.columns) * 0.7)

    # Fill remaining NaNs with 0
    df = df.fillna(0)

    print(f"   ✅ Extracted {len(df)} valid game records")
    print(f"   Features: {list(df.columns)}")

    return df


def run_causal_discovery(df, method='pc'):
    """Run causal discovery algorithm on the data."""

    print(f"\n🔬 Running causal discovery ({method.upper()} algorithm)...")
    print(f"   Data shape: {df.shape}")
    print()

    learner = NCAACousalLearner(cache_dir='models/causal_models')

    # Run discovery
    results = learner.discover_causality(df, method=method, confidence_level=0.05)

    print("\n📊 CAUSAL DISCOVERY RESULTS")
    print("=" * 80)
    print(f"Direct causal edges: {results['edge_count']}")
    print(f"Confounders: {results['confounder_count']}")
    print(f"Mediation paths: {results['mediator_count']}")
    print()

    # Show sample causal edges
    if results['direct_causal_edges']:
        print("Sample Causal Relationships:")
        for edge in results['direct_causal_edges'][:10]:
            print(f"  {edge['cause']} → {edge['effect']}")
        print()

    # Show predictive paths
    predictive = learner.get_predictive_paths()
    if predictive['causal_edges']:
        print("Causal Paths to Game Outcomes:")
        for path in predictive['causal_edges']:
            print(f"  {path['predictor']} → {path['target']}")
            print(f"    Action: {path['action']}")
        print()

    if predictive['mediation_paths']:
        print("Mediation Paths to Game Outcomes:")
        for path in predictive['mediation_paths'][:5]:
            print(f"  {path['path']}")
            print(f"    Action: {path['action']}")
        print()

    return results, learner


def main():
    """Main execution."""

    # Step 1: Load data
    games = load_ncaa_games_data(start_year=2015, end_year=2024)

    if not games:
        print("❌ No games loaded. Cannot run causal discovery.")
        return

    # Step 2: Extract features
    df = extract_causal_features(games)

    if df.empty:
        print("❌ No valid features extracted. Cannot run causal discovery.")
        return

    # Optional: Save combined data for future use
    data_file = Path("data/ncaa_causal_discovery_data.csv")
    df.to_csv(data_file, index=False)
    print(f"💾 Saved feature data to: {data_file}")
    print()

    # Step 3: Run PC algorithm (fast)
    print("🚀 Running PC Algorithm (fast, exploratory)...")
    pc_results, pc_learner = run_causal_discovery(df, method='pc')

    # Step 4: Optionally run GES (slower but more conservative)
    run_ges = input("\n❓ Run GES algorithm too? (slower, more conservative) [y/N]: ").lower()

    if run_ges == 'y':
        print("\n🐢 Running GES Algorithm (slower, conservative)...")
        ges_results, ges_learner = run_causal_discovery(df, method='ges')

        # Compare results
        print("\n📊 ALGORITHM COMPARISON")
        print("=" * 80)
        print(f"PC Algorithm:")
        print(f"  Causal edges: {pc_results['edge_count']}")
        print(f"  Confounders: {pc_results['confounder_count']}")
        print(f"  Mediators: {pc_results['mediator_count']}")
        print()
        print(f"GES Algorithm:")
        print(f"  Causal edges: {ges_results['edge_count']}")
        print(f"  Confounders: {ges_results['confounder_count']}")
        print(f"  Mediators: {ges_results['mediator_count']}")
        print()

    print("✅ CAUSAL DISCOVERY COMPLETE!")
    print()
    print("📁 Results cached in: models/causal_models/")
    print("   - causal_graph.pkl")
    print("   - causal_paths.json")
    print()
    print("🎯 Next steps:")
    print("   1. Review causal_paths.json to see discovered relationships")
    print("   2. Use predict_ncaa_world_models.py for predictions")
    print("   3. Results will be loaded automatically from cache")


if __name__ == "__main__":
    main()
