#!/usr/bin/env python3
"""
Backtest Day-of-Week Edge Patterns
===================================

HYPOTHESIS TO TEST:
- Tuesday: Softer lines (Vegas lets public win to hook them)
- Wednesday: Sharper lines (Vegas takes money back)
- Thursday-Saturday: Mixed

We'll analyze 2015-2024 data to see if this pattern exists.

METRICS BY DAY:
1. Model prediction accuracy by day
2. Average prediction error by day
3. Edge distribution by day
4. Win rate against spread by day

If hypothesis is correct, we should see:
- Tuesday: Lowest prediction error, highest edge, best win rate
- Wednesday: Highest prediction error, lowest edge, worst win rate
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator


class DayOfWeekEdgeBacktest:
    """Backtest to see if certain days have more edge"""

    def __init__(self):
        self.models_dir = Path("models/ncaa")
        self.engineer = NCAAFeatureEngineer()
        self.orchestrator = SuperIntelligenceOrchestrator(models_dir=str(self.models_dir))

        self._load_models()

        print("="*80)
        print("📊 DAY-OF-WEEK EDGE PATTERN BACKTEST")
        print("="*80)
        print()
        print("Testing hypothesis:")
        print("  • Tuesday = Softer lines (let public win)")
        print("  • Wednesday = Sharper lines (take money back)")
        print("  • Thursday-Saturday = Mixed")
        print()

    def _load_models(self):
        """Load trained models"""
        model_names = ['xgboost_super', 'neural_net_deep', 'alt_spread']

        for model_name in model_names:
            model_path = self.models_dir / f"{model_name}.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        model = self.orchestrator.models[model_name]

                        if 'model' in model_data:
                            if isinstance(model_data['model'], dict):
                                model.models = model_data['model']
                            else:
                                model.model = model_data['model']

                        if 'scaler' in model_data:
                            model.scaler = model_data['scaler']

                        model.is_trained = True

                except Exception as e:
                    print(f"⚠️  Error loading {model_name}: {e}")

    def analyze_day_of_week_patterns(self, start_year: int = 2015, end_year: int = 2024):
        """
        Analyze prediction accuracy and edge by day of week
        """
        print(f"📅 Analyzing {start_year}-{end_year} seasons...")
        print()

        # Results by day of week
        results_by_day = {
            'Tuesday': {'predictions': [], 'actuals': [], 'errors': []},
            'Wednesday': {'predictions': [], 'actuals': [], 'errors': []},
            'Thursday': {'predictions': [], 'actuals': [], 'errors': []},
            'Friday': {'predictions': [], 'actuals': [], 'errors': []},
            'Saturday': {'predictions': [], 'actuals': [], 'errors': []},
            'Sunday': {'predictions': [], 'actuals': [], 'errors': []},  # Rare but exists
        }

        total_games = 0
        processed_games = 0

        for year in range(start_year, end_year + 1):
            print(f"   Processing {year}...", end=" ")

            try:
                games = self.engineer.load_season_data(year)
                print(f"{len(games)} games")

                for game in games:
                    total_games += 1

                    # Get day of week
                    start_date = game.get('startDate')
                    if not start_date:
                        continue

                    try:
                        # Parse date
                        game_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        day_of_week = game_date.strftime('%A')

                        if day_of_week not in results_by_day:
                            continue

                        # Get actual spread
                        home_points = game.get('homePoints')
                        away_points = game.get('awayPoints')

                        if home_points is None or away_points is None:
                            continue

                        actual_spread = home_points - away_points

                        # Generate prediction
                        features = self.engineer.engineer_features(game, year)
                        if not features:
                            continue

                        feature_array = np.array([list(features.values())])

                        # Get predictions from models
                        spread_preds = []
                        for model_name in ['xgboost_super', 'neural_net_deep', 'alt_spread']:
                            model = self.orchestrator.models[model_name]
                            if model.is_trained:
                                try:
                                    pred = model.predict(feature_array)
                                    spread_preds.append(pred[0] if isinstance(pred, np.ndarray) else pred)
                                except:
                                    pass

                        if not spread_preds:
                            continue

                        predicted_spread = np.mean(spread_preds)
                        error = abs(predicted_spread - actual_spread)

                        # Store results
                        results_by_day[day_of_week]['predictions'].append(predicted_spread)
                        results_by_day[day_of_week]['actuals'].append(actual_spread)
                        results_by_day[day_of_week]['errors'].append(error)

                        processed_games += 1

                    except Exception as e:
                        continue

            except Exception as e:
                print(f"Error: {e}")
                continue

        print()
        print(f"✅ Processed {processed_games:,} / {total_games:,} games")
        print()

        return results_by_day

    def calculate_day_statistics(self, results_by_day: Dict) -> pd.DataFrame:
        """Calculate statistics for each day"""
        stats = []

        for day, data in results_by_day.items():
            if len(data['predictions']) == 0:
                continue

            predictions = np.array(data['predictions'])
            actuals = np.array(data['actuals'])
            errors = np.array(data['errors'])

            # Basic stats
            count = len(predictions)
            mean_error = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)

            # Win rate (within 7 points = "correct")
            within_7 = np.sum(errors <= 7.0)
            win_rate = (within_7 / count) * 100

            # Edge calculation
            # Edge = how much closer we are to actual than a random guess
            # Random guess error would be ~14 points (half of typical spread range)
            edge = ((14.0 - mean_error) / 14.0) * 100

            stats.append({
                'Day': day,
                'Games': count,
                'Mean Error': round(mean_error, 2),
                'Median Error': round(median_error, 2),
                'Std Error': round(std_error, 2),
                'Win Rate %': round(win_rate, 1),
                'Edge %': round(edge, 1),
            })

        df = pd.DataFrame(stats)

        # Sort by typical week order
        day_order = {'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        df['sort_order'] = df['Day'].map(day_order)
        df = df.sort_values('sort_order').drop('sort_order', axis=1)

        return df

    def test_hypothesis(self, df: pd.DataFrame):
        """Test if Tuesday is actually softer than Wednesday"""
        print("="*80)
        print("🔬 HYPOTHESIS TEST")
        print("="*80)
        print()

        if 'Tuesday' not in df['Day'].values or 'Wednesday' not in df['Day'].values:
            print("❌ Not enough Tuesday/Wednesday games to test hypothesis")
            return

        tuesday = df[df['Day'] == 'Tuesday'].iloc[0]
        wednesday = df[df['Day'] == 'Wednesday'].iloc[0]

        print("Hypothesis: Tuesday lines are SOFTER than Wednesday")
        print()
        print("Expected if TRUE:")
        print("  • Tuesday mean error < Wednesday mean error")
        print("  • Tuesday win rate > Wednesday win rate")
        print("  • Tuesday edge > Wednesday edge")
        print()

        print("RESULTS:")
        print("-" * 80)

        # Test 1: Mean error
        tuesday_error = tuesday['Mean Error']
        wednesday_error = wednesday['Mean Error']
        test1_pass = tuesday_error < wednesday_error

        print(f"1. Mean Error:")
        print(f"   Tuesday:    {tuesday_error:.2f} points")
        print(f"   Wednesday:  {wednesday_error:.2f} points")
        print(f"   Difference: {wednesday_error - tuesday_error:+.2f} points")
        print(f"   Result:     {'✅ PASS' if test1_pass else '❌ FAIL'} - Tuesday {'IS' if test1_pass else 'IS NOT'} easier")
        print()

        # Test 2: Win rate
        tuesday_wr = tuesday['Win Rate %']
        wednesday_wr = wednesday['Win Rate %']
        test2_pass = tuesday_wr > wednesday_wr

        print(f"2. Win Rate:")
        print(f"   Tuesday:    {tuesday_wr:.1f}%")
        print(f"   Wednesday:  {wednesday_wr:.1f}%")
        print(f"   Difference: {tuesday_wr - wednesday_wr:+.1f}%")
        print(f"   Result:     {'✅ PASS' if test2_pass else '❌ FAIL'} - Tuesday {'IS' if test2_pass else 'IS NOT'} more profitable")
        print()

        # Test 3: Edge
        tuesday_edge = tuesday['Edge %']
        wednesday_edge = wednesday['Edge %']
        test3_pass = tuesday_edge > wednesday_edge

        print(f"3. Edge:")
        print(f"   Tuesday:    {tuesday_edge:.1f}%")
        print(f"   Wednesday:  {wednesday_edge:.1f}%")
        print(f"   Difference: {tuesday_edge - wednesday_edge:+.1f}%")
        print(f"   Result:     {'✅ PASS' if test3_pass else '❌ FAIL'} - Tuesday {'HAS' if test3_pass else 'DOES NOT HAVE'} more edge")
        print()

        # Overall verdict
        tests_passed = sum([test1_pass, test2_pass, test3_pass])

        print("="*80)
        if tests_passed >= 2:
            print("✅ HYPOTHESIS CONFIRMED!")
            print("="*80)
            print()
            print("Tuesday IS softer than Wednesday!")
            print("Recommendation: BET MORE AGGRESSIVELY ON TUESDAY")
        else:
            print("❌ HYPOTHESIS REJECTED")
            print("="*80)
            print()
            print("Tuesday is NOT significantly softer than Wednesday")
            print("Recommendation: Use same strategy all week")
        print()

    def display_results(self, df: pd.DataFrame):
        """Display results table"""
        print("="*80)
        print("📊 DAY-OF-WEEK EDGE ANALYSIS RESULTS")
        print("="*80)
        print()
        print(df.to_string(index=False))
        print()

        # Find best and worst days
        best_day = df.loc[df['Edge %'].idxmax()]
        worst_day = df.loc[df['Edge %'].idxmin()]

        print("="*80)
        print("🏆 BEST DAY:")
        print("="*80)
        print(f"   Day: {best_day['Day']}")
        print(f"   Edge: {best_day['Edge %']:.1f}%")
        print(f"   Win Rate: {best_day['Win Rate %']:.1f}%")
        print(f"   Mean Error: {best_day['Mean Error']:.2f} points")
        print(f"   Games: {best_day['Games']:,}")
        print()

        print("="*80)
        print("⚠️  WORST DAY:")
        print("="*80)
        print(f"   Day: {worst_day['Day']}")
        print(f"   Edge: {worst_day['Edge %']:.1f}%")
        print(f"   Win Rate: {worst_day['Win Rate %']:.1f}%")
        print(f"   Mean Error: {worst_day['Mean Error']:.2f} points")
        print(f"   Games: {worst_day['Games']:,}")
        print()

    def recommend_strategy(self, df: pd.DataFrame):
        """Recommend betting strategy based on results"""
        print("="*80)
        print("💡 RECOMMENDED BETTING STRATEGY")
        print("="*80)
        print()

        for _, row in df.iterrows():
            day = row['Day']
            edge = row['Edge %']
            win_rate = row['Win Rate %']

            # Determine aggressiveness
            if edge >= 40 and win_rate >= 70:
                stance = "VERY AGGRESSIVE"
                threshold = "3-5%"
                stake = "3%"
            elif edge >= 35 and win_rate >= 65:
                stance = "AGGRESSIVE"
                threshold = "4-6%"
                stake = "2-3%"
            elif edge >= 30 and win_rate >= 60:
                stance = "STANDARD"
                threshold = "5-7%"
                stake = "2%"
            elif edge >= 25 and win_rate >= 55:
                stance = "CONSERVATIVE"
                threshold = "7-10%"
                stake = "1-2%"
            else:
                stance = "VERY CONSERVATIVE"
                threshold = "10%+"
                stake = "1%"

            print(f"{day}:")
            print(f"   Stance: {stance}")
            print(f"   Edge Threshold: {threshold}")
            print(f"   Stake Size: {stake} of bankroll")
            print(f"   Reason: {edge:.1f}% edge, {win_rate:.1f}% win rate")
            print()


def main():
    """Main backtest"""
    backtest = DayOfWeekEdgeBacktest()

    # Run analysis
    results_by_day = backtest.analyze_day_of_week_patterns(2015, 2024)

    # Calculate statistics
    df = backtest.calculate_day_statistics(results_by_day)

    # Display results
    backtest.display_results(df)

    # Test hypothesis
    backtest.test_hypothesis(df)

    # Recommend strategy
    backtest.recommend_strategy(df)

    # Save results
    output_dir = Path("data/backtests")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "day_of_week_edge_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"💾 Results saved to {output_file}")


if __name__ == "__main__":
    main()
