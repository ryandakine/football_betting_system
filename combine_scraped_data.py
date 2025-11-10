#!/usr/bin/env python3
"""
Combine Scraped Market Data
============================

After running scrapers locally, this combines all sources into clean datasets.

USAGE:
    python combine_scraped_data.py
"""

import pandas as pd
from pathlib import Path
import glob
import json


def combine_all_spreads():
    """Combine all scraped spreads into single dataset per year"""

    print("="*80)
    print("🔗 COMBINING SCRAPED MARKET DATA")
    print("="*80)
    print()

    # Directories
    cache_dir = Path("data/wayback_spreads")
    market_dir = Path("data/market_spreads")
    output_dir = Path("data")

    all_games = []

    # 1. Load Archive.org data
    if cache_dir.exists():
        print("📦 Loading Archive.org data...")
        json_files = list(cache_dir.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                games = data.get('games', [])
                for game in games:
                    all_games.append({
                        'year': game.get('year'),
                        'week': game.get('week'),
                        'home_team': game.get('home_team'),
                        'away_team': game.get('away_team'),
                        'market_spread': game.get('spread'),
                        'source': game.get('source', 'archive_org'),
                    })
            except Exception as e:
                print(f"   ⚠️  Error loading {json_file}: {e}")

        print(f"   ✅ Loaded {len(all_games)} games from Archive.org")

    # 2. Load direct scraper data
    if market_dir.exists():
        print("📦 Loading direct scraper data...")
        csv_files = list(market_dir.glob("*.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                for _, row in df.iterrows():
                    all_games.append({
                        'year': row.get('year'),
                        'week': row.get('week'),
                        'home_team': row.get('home_team'),
                        'away_team': row.get('away_team'),
                        'market_spread': row.get('market_spread') or row.get('spread'),
                        'source': row.get('source', 'direct_scraper'),
                    })
            except Exception as e:
                print(f"   ⚠️  Error loading {csv_file}: {e}")

        print(f"   ✅ Loaded {len(all_games)} total games")

    if not all_games:
        print("\n❌ No data found!")
        print("   Run scrapers first on your local machine")
        return

    # 3. Convert to DataFrame
    df = pd.DataFrame(all_games)

    # 4. Clean data
    print("\n🧹 Cleaning data...")

    # Remove rows with missing data
    df = df.dropna(subset=['year', 'week', 'home_team', 'away_team', 'market_spread'])

    # Normalize team names
    df['home_team'] = df['home_team'].str.strip().str.title()
    df['away_team'] = df['away_team'].str.strip().str.title()

    # Convert types
    df['year'] = df['year'].astype(int)
    df['week'] = df['week'].astype(int)
    df['market_spread'] = df['market_spread'].astype(float)

    # Remove duplicates (keep first occurrence = most reliable source)
    print(f"   Before dedup: {len(df):,} games")
    df = df.drop_duplicates(
        subset=['year', 'week', 'home_team', 'away_team'],
        keep='first'
    )
    print(f"   After dedup: {len(df):,} games")

    # 5. Save by year
    print("\n💾 Saving datasets...")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        output_file = output_dir / f"market_spreads_{year}.csv"
        year_data.to_csv(output_file, index=False)
        print(f"   ✅ {year}: {len(year_data):,} games → {output_file}")
        total_saved += len(year_data)

    # 6. Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print()
    print(f"Total games with market spreads: {total_saved:,}")
    print(f"Years covered: {df['year'].min()}-{df['year'].max()}")
    print(f"Weeks per year: {df.groupby('year')['week'].nunique().mean():.1f}")
    print()

    # Coverage by year
    print("Coverage by year:")
    year_coverage = df.groupby('year').size()
    for year, count in year_coverage.items():
        weeks = df[df['year'] == year]['week'].nunique()
        print(f"   {year}: {count:,} games across {weeks} weeks")

    print()
    print("="*80)
    print("✅ DATA READY FOR BACKTESTING!")
    print("="*80)
    print()
    print("Next step: python backtest_ncaa_parlays_REALISTIC.py")
    print()

    return df


def check_coverage(df: pd.DataFrame):
    """Check data coverage and identify gaps"""

    print("="*80)
    print("🔍 COVERAGE ANALYSIS")
    print("="*80)
    print()

    # Expected games per week (approximate)
    expected_per_week = {
        1: 50, 2: 60, 3: 60, 4: 60, 5: 60,
        6: 60, 7: 60, 8: 60, 9: 60, 10: 60,
        11: 60, 12: 60, 13: 60, 14: 40, 15: 10  # Championship week
    }

    gaps = []

    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]

        for week in range(1, 16):
            week_data = year_data[year_data['week'] == week]
            actual = len(week_data)
            expected = expected_per_week.get(week, 60)

            coverage_pct = (actual / expected) * 100

            if coverage_pct < 50:
                gaps.append({
                    'year': year,
                    'week': week,
                    'actual': actual,
                    'expected': expected,
                    'coverage': coverage_pct
                })

    if gaps:
        print("⚠️  Gaps found (< 50% coverage):")
        print()
        for gap in gaps[:20]:  # Show first 20
            print(f"   {gap['year']} Week {gap['week']}: "
                  f"{gap['actual']}/{gap['expected']} games "
                  f"({gap['coverage']:.0f}%)")

        if len(gaps) > 20:
            print(f"   ... and {len(gaps) - 20} more gaps")

        print()
        print("💡 Re-run scrapers for these specific year/week combos")
    else:
        print("✅ No major gaps found! Coverage looks good.")

    print()


def main():
    """Main entry point"""

    # Combine all data
    df = combine_all_spreads()

    if df is not None:
        # Check coverage
        check_coverage(df)


if __name__ == "__main__":
    main()
