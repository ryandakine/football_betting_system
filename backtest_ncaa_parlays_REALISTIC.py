#!/usr/bin/env python3
"""
NCAA Parlay Backtesting - REALISTIC VERSION
============================================

Fixes the inflated win rate by:
1. Using market spreads (when available)
2. Proper bet evaluation (did we beat the spread?)
3. Realistic edge calculation
4. No generous ±7 point margins

Expected results: 52-55% win rate, 5-10% ROI
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator
from college_football_system.parlay_optimizer import ParlayOptimizer, ParlayBet


# Configuration
STARTING_BANKROLL = 10000
FRACTIONAL_KELLY = 0.25
YEARS = list(range(2015, 2025))

# Strategies
STRATEGIES = {
    'conservative': {
        'max_parlay_size': 2,
        'min_confidence': 0.70,
        'min_edge': 0.03,  # 3% edge minimum (realistic)
        'bankroll_allocation': 0.02,  # 2% per parlay
        'max_parlays_per_week': 5,
    },
    'balanced': {
        'max_parlay_size': 3,
        'min_confidence': 0.65,
        'min_edge': 0.025,  # 2.5% edge
        'bankroll_allocation': 0.03,  # 3% per parlay
        'max_parlays_per_week': 8,
    },
}


class RealisticNCAAParlayBacktester:
    """Realistic parlay backtesting with proper spread evaluation"""

    def __init__(self, models_dir="models/ncaa"):
        self.models_dir = Path(models_dir)
        self.engineer = NCAAFeatureEngineer()
        self.orchestrator = SuperIntelligenceOrchestrator(models_dir=str(models_dir))

        # Load trained models
        self._load_models()

        # Load market spreads if available
        self.market_spreads = self._load_market_spreads()

        # Results storage
        self.results = {
            'conservative': {'bets': [], 'bankroll_history': [STARTING_BANKROLL]},
            'balanced': {'bets': [], 'bankroll_history': [STARTING_BANKROLL]},
        }

        print(f"✅ Realistic Backtester initialized")
        print(f"   Models loaded: {self.models_loaded}/3")
        print(f"   Market spreads available: {len(self.market_spreads)} games")

    def _load_models(self):
        """Load trained models"""
        model_names = ['xgboost_super', 'neural_net_deep', 'alt_spread']

        self.models_loaded = 0
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
                        self.models_loaded += 1

                except Exception as e:
                    print(f"⚠️  Error loading {model_name}: {e}")

    def _load_market_spreads(self) -> Dict:
        """Load historical market spreads from CSV files"""
        market_data = {}

        for year in YEARS:
            csv_file = Path(f"data/market_spreads_{year}.csv")
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    game_id = row['game_id']
                    market_data[game_id] = {
                        'market_spread': row['market_spread'],
                        'source': row['source'],
                    }

        return market_data

    def backtest_all_strategies(self):
        """Run realistic backtest"""
        print("\n" + "="*80)
        print("🎰 NCAA PARLAY BACKTESTING - REALISTIC (10 YEARS)")
        print("="*80)

        for strategy_name, config in STRATEGIES.items():
            print(f"\n{'='*80}")
            print(f"📊 STRATEGY: {strategy_name.upper()}")
            print(f"   Max Size: {config['max_parlay_size']} legs")
            print(f"   Min Confidence: {config['min_confidence']*100:.0f}%")
            print(f"   Min Edge: {config['min_edge']*100:.1f}%")
            print(f"{'='*80}")

            self._backtest_strategy(strategy_name, config)

        self._display_results()

    def _backtest_strategy(self, strategy_name: str, config: Dict):
        """Backtest a single strategy"""
        bankroll = STARTING_BANKROLL
        total_bets = 0
        total_wins = 0

        for year in YEARS:
            print(f"\n📅 {year} Season...")

            try:
                games = self.engineer.load_season_data(year)
            except:
                print(f"   ⚠️  No data for {year}")
                continue

            # Group by week
            weeks = defaultdict(list)
            for game in games:
                if game.get('completed', False):
                    week = game.get('week', 0)
                    weeks[week].append(game)

            year_bets_start = len(self.results[strategy_name]['bets'])

            # Process each week
            for week_num in sorted(weeks.keys()):
                week_games = weeks[week_num]

                # Get predictions
                predictions = self._predict_week_realistic(week_games, year)

                # Filter by thresholds
                eligible = [
                    p for p in predictions
                    if p['confidence'] >= config['min_confidence']
                    and p.get('edge_value', 0) >= config['min_edge']
                    and p.get('has_market_spread', False)  # MUST have market spread
                ]

                if len(eligible) < config['max_parlay_size']:
                    continue

                # Build parlays
                parlays = self._build_parlays(eligible, config, bankroll)
                parlays = parlays[:config['max_parlays_per_week']]

                # Evaluate each parlay
                for parlay in parlays:
                    result = self._evaluate_parlay_realistic(parlay)

                    total_bets += 1
                    if result['won']:
                        total_wins += 1
                        profit = result['payout'] - result['stake']
                        bankroll += profit
                    else:
                        bankroll -= result['stake']

                    self.results[strategy_name]['bets'].append({
                        'year': year,
                        'week': week_num,
                        'legs': len(parlay.games),
                        'stake': result['stake'],
                        'won': result['won'],
                        'profit': result['payout'] - result['stake'] if result['won'] else -result['stake'],
                        'bankroll': bankroll,
                    })

                    self.results[strategy_name]['bankroll_history'].append(bankroll)

            # Year summary
            year_bets_end = len(self.results[strategy_name]['bets'])
            year_bets_count = year_bets_end - year_bets_start

            if year_bets_count > 0:
                year_bets = self.results[strategy_name]['bets'][year_bets_start:year_bets_end]
                year_wins = sum(1 for b in year_bets if b['won'])
                year_profit = sum(b['profit'] for b in year_bets)
                print(f"   {year}: {year_wins}/{year_bets_count} ({year_wins/year_bets_count*100:.1f}%) | "
                      f"Profit: ${year_profit:+,.0f} | Bankroll: ${bankroll:,.0f}")
            else:
                print(f"   {year}: 0 parlays (no market data)")

        # Strategy summary
        if total_bets > 0:
            win_rate = total_wins / total_bets
            final_profit = bankroll - STARTING_BANKROLL
            roi = (final_profit / STARTING_BANKROLL) * 100

            print(f"\n✅ {strategy_name.upper()} COMPLETE:")
            print(f"   Total Parlays: {total_bets}")
            print(f"   Win Rate: {win_rate*100:.2f}%")
            print(f"   Final Bankroll: ${bankroll:,.0f}")
            print(f"   Total Profit: ${final_profit:+,.0f}")
            print(f"   ROI: {roi:+.2f}%")

    def _predict_week_realistic(self, games: List[Dict], year: int) -> List[Dict]:
        """Get predictions with realistic edge calculation"""
        predictions = []

        for game in games:
            try:
                game_id = game.get('id')

                # Check if we have market spread
                if game_id not in self.market_spreads:
                    continue

                market_spread = self.market_spreads[game_id]['market_spread']

                # Engineer features
                features = self.engineer.engineer_features(game, year)
                if not features:
                    continue

                feature_array = np.array([list(features.values())])

                # Get predictions
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
                spread_std = np.std(spread_preds)

                # Confidence
                raw_confidence = max(0.5, min(0.95, 1 - (spread_std / 15.0)))
                confidence = raw_confidence * 0.90

                # REALISTIC EDGE: Difference from market
                edge = abs(predicted_spread - market_spread) / 7.0  # Normalize by ~1 TD

                # Get actual result for evaluation
                home_score = game.get('homePoints', 0) or 0
                away_score = game.get('awayPoints', 0) or 0
                actual_spread = home_score - away_score

                predictions.append({
                    'game_id': game_id,
                    'home_team': game.get('homeTeam', ''),
                    'away_team': game.get('awayTeam', ''),
                    'conference': game.get('homeConference', ''),
                    'predicted_spread': predicted_spread,
                    'market_spread': market_spread,
                    'actual_spread': actual_spread,
                    'confidence': confidence,
                    'edge_value': edge,
                    'odds': -110,
                    'has_market_spread': True,
                })

            except Exception as e:
                continue

        return predictions

    def _build_parlays(self, predictions: List[Dict], config: Dict, bankroll: float) -> List[ParlayBet]:
        """Build parlays using optimizer"""
        optimizer_config = {
            'max_parlay_size': config['max_parlay_size'],
            'min_confidence_threshold': config['min_confidence'],
            'min_edge_threshold': config['min_edge'],
            'max_correlation_penalty': 0.15,
            'bankroll_allocation': config['bankroll_allocation'],
            'risk_tolerance': 'medium',
            'conference_correlations': {
                'same_conference': 0.3,
                'rival_games': 0.2,
                'neutral_site': -0.1,
            },
        }
        optimizer = ParlayOptimizer(optimizer_config)
        parlays = optimizer.optimize_parlays(predictions, bankroll)
        return parlays

    def _evaluate_parlay_realistic(self, parlay: ParlayBet) -> Dict:
        """
        Evaluate parlay using REALISTIC betting rules:
        - Did our prediction beat the market spread?
        - No generous ±7 margins
        """
        all_legs_won = True
        leg_results = []

        for game_pred in parlay.games:
            our_prediction = game_pred['predicted_spread']
            market_spread = game_pred['market_spread']
            actual_spread = game_pred['actual_spread']

            # Determine which side we bet
            if our_prediction > market_spread:
                # We think home wins by more than market
                # Bet on home to cover
                bet_home = True
                bet_won = actual_spread > market_spread
            else:
                # We think home wins by less (or loses)
                # Bet on away to cover
                bet_home = False
                bet_won = actual_spread < market_spread

            leg_results.append(bet_won)

            if not bet_won:
                all_legs_won = False

        # Calculate payout
        if all_legs_won:
            if parlay.combined_odds >= 0:
                decimal_odds = (parlay.combined_odds / 100) + 1
            else:
                decimal_odds = (100 / abs(parlay.combined_odds)) + 1

            payout = parlay.recommended_stake * decimal_odds
        else:
            payout = 0

        return {
            'won': all_legs_won,
            'stake': parlay.recommended_stake,
            'payout': payout,
            'leg_results': leg_results,
        }

    def _display_results(self):
        """Display final results"""
        print("\n" + "="*80)
        print("📊 FINAL RESULTS - REALISTIC BACKTEST")
        print("="*80)

        for strategy_name in ['conservative', 'balanced']:
            bets = self.results[strategy_name]['bets']

            if not bets:
                continue

            wins = sum(1 for b in bets if b['won'])
            total = len(bets)
            win_rate = wins / total if total > 0 else 0

            final_bankroll = self.results[strategy_name]['bankroll_history'][-1]
            total_profit = final_bankroll - STARTING_BANKROLL
            roi = (total_profit / STARTING_BANKROLL) * 100

            print(f"\n{strategy_name.upper()}:")
            print(f"  Total Parlays: {total}")
            print(f"  Win Rate: {win_rate*100:.2f}%")
            print(f"  Final Bankroll: ${final_bankroll:,.0f}")
            print(f"  Total Profit: ${total_profit:+,.0f}")
            print(f"  ROI: {roi:+.2f}%")

            # By parlay size
            by_size = defaultdict(lambda: {'wins': 0, 'total': 0})
            for bet in bets:
                by_size[bet['legs']]['total'] += 1
                if bet['won']:
                    by_size[bet['legs']]['wins'] += 1

            print(f"\n  By Parlay Size:")
            for size in sorted(by_size.keys()):
                size_wins = by_size[size]['wins']
                size_total = by_size[size]['total']
                size_rate = size_wins / size_total if size_total > 0 else 0
                print(f"    {size}-leg: {size_wins}/{size_total} ({size_rate*100:.1f}%)")


def main():
    """Main entry point"""
    backtester = RealisticNCAAParlayBacktester()

    if len(backtester.market_spreads) == 0:
        print("\n" + "="*80)
        print("❌ NO MARKET SPREAD DATA AVAILABLE")
        print("="*80)
        print()
        print("To run realistic backtest, you need historical market spreads.")
        print()
        print("Options:")
        print("  1. Run: python scrape_historical_ncaa_spreads.py 2024")
        print("  2. Purchase data from Sports Insights / SportsDataIO")
        print("  3. Use The Odds API historical endpoint")
        print()
        print("Once you have market_spreads_YEAR.csv files in data/,")
        print("rerun this backtest.")
        print()
        return

    backtester.backtest_all_strategies()


if __name__ == "__main__":
    main()
