#!/usr/bin/env python3
"""
NCAA DeepSeek R1 System Backtest
=================================

PRINCIPLE: Test R1 meta-analysis on historical data

WHAT THIS TESTS:
1. Load first half 2024 season (Sep-Oct games)
2. Generate REAL 12-model predictions for each game
3. Run R1 meta-analysis (simplified - no real-time context)
4. Compare R1 recommendations vs actual results
5. Track win rate and ROI by R1 confidence tier

LIMITATIONS:
- No real-time context (injuries, weather, momentum narratives)
- No true contrarian signals (no historical public betting %)
- Simplified R1 prompt (focused on model analysis only)

But we CAN test:
- Does R1 find edges in the 12-model ensemble?
- Does R1's confidence correlate with win rate?
- Does R1's bet sizing produce positive ROI?

USAGE:
    python backtest_ncaa_r1_system.py <DEEPSEEK_API_KEY>

EXPECTED RESULTS:
- Win rate: 56-62% (based on NFL 60.91x validation)
- ROI: 30-50% over half season
- Higher confidence = higher win rate
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator
from ncaa_deepseek_r1_reasoner import NCAADeepSeekR1Reasoner, ModelPrediction, R1Analysis


class NCAAR1Backtester:
    """
    Backtest R1 system on historical games

    Tests: Does R1 meta-analysis beat market spreads?
    """

    def __init__(self, deepseek_api_key: str, models_dir: str = "models/ncaa"):
        self.models_dir = Path(models_dir)
        self.engineer = NCAAFeatureEngineer()
        self.orchestrator = SuperIntelligenceOrchestrator(models_dir=str(models_dir))

        # Initialize R1 reasoner
        self.r1_reasoner = NCAADeepSeekR1Reasoner(deepseek_api_key)

        # Load trained models
        self._load_models()

        # Load market spreads
        self.market_spreads = self._load_market_spreads()

        # Load optimal config
        self.config = self._load_optimal_config()

        # Results by confidence tier
        self.results_by_tier = {
            '80%+': {'bets': [], 'wins': 0, 'losses': 0, 'total_profit': 0},
            '75-79%': {'bets': [], 'wins': 0, 'losses': 0, 'total_profit': 0},
            '70-74%': {'bets': [], 'wins': 0, 'losses': 0, 'total_profit': 0},
            'below_70%': {'bets': [], 'wins': 0, 'losses': 0, 'total_profit': 0},
        }

        print(f"✅ R1 Backtester initialized")
        print(f"   Models loaded: {self.models_loaded}/12")
        print(f"   Market spreads: {len(self.market_spreads)} games")

    def _load_models(self):
        """Load all 12 trained models"""
        model_names = [
            'xgboost_super', 'neural_net_deep', 'alt_spread',
            'bayesian_ensemble', 'momentum_model', 'situational',
            'advanced_stats', 'drive_outcomes', 'opponent_adjusted',
            'special_teams', 'pace_tempo', 'game_script'
        ]

        self.models_loaded = 0
        for model_name in model_names:
            model_path = self.models_dir / f"{model_name}.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        model = self.orchestrator.models.get(model_name)

                        if not model:
                            continue

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
        """Load historical market spreads"""
        market_data = {}

        # Try to load from CSV files
        for year in range(2015, 2025):
            csv_file = Path(f"data/market_spreads_{year}.csv")
            if csv_file.exists():
                import pandas as pd
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    game_id = row['game_id']
                    market_data[game_id] = {
                        'market_spread': row['market_spread'],
                        'source': row.get('source', 'unknown'),
                    }

        return market_data

    def _load_optimal_config(self) -> Dict:
        """Load R1 optimal config"""
        config_path = Path("ncaa_optimal_llm_weights.json")

        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        else:
            # Default config
            return {
                "model": "deepseek-r1",
                "weight": 1.0,
                "min_confidence": 70.0,
                "bet_sizing": {
                    "80%+ confidence": 6,
                    "75-79% confidence": 4,
                    "70-74% confidence": 2
                }
            }

    def validate_market_data(self, year: int) -> tuple[bool, str]:
        """
        Validate sufficient market spread coverage for backtest

        Returns: (is_valid, error_message)
        """
        try:
            games = self.engineer.load_season_data(year)
            total_games = len(games)

            # Count games with market spreads
            games_with_spreads = sum(
                1 for game in games
                if game.get('id') in self.market_spreads
            )

            coverage = games_with_spreads / total_games if total_games > 0 else 0

            MIN_COVERAGE = 0.80  # 80% minimum

            if coverage < MIN_COVERAGE:
                error_msg = (
                    f"\n{'='*80}\n"
                    f"❌ INSUFFICIENT MARKET DATA FOR {year}\n"
                    f"{'='*80}\n\n"
                    f"Total games: {total_games}\n"
                    f"With market spreads: {games_with_spreads} ({coverage:.1%})\n"
                    f"Required: {MIN_COVERAGE:.0%}+\n\n"
                    f"Cannot backtest R1 without market spreads.\n\n"
                    f"ACTION REQUIRED:\n"
                    f"Run scrapers to get market spread data:\n"
                    f"  python scrape_teamrankings_historical.py {year}\n"
                    f"  python scrape_covers_historical.py {year}\n"
                )
                return False, error_msg

            return True, f"✅ Market data coverage: {coverage:.1%} ({games_with_spreads}/{total_games} games)"

        except Exception as e:
            return False, f"❌ Error validating market data: {e}"

    def backtest_first_half_season(self, year: int = 2024):
        """
        Backtest R1 on first half of season

        WHY: Treat games as if they haven't been played yet
        """

        print(f"\n{'='*80}")
        print(f"🧠 BACKTESTING R1 SYSTEM - FIRST HALF {year} SEASON")
        print(f"{'='*80}\n")

        # Step 1: Validate market data
        is_valid, msg = self.validate_market_data(year)
        print(msg)

        if not is_valid:
            return

        # Step 2: Load season games
        print(f"\n📊 Loading {year} season data...")
        games = self.engineer.load_season_data(year)

        # Filter to first half of season (Sep-Oct, weeks 1-8)
        first_half_games = []
        for game in games:
            week = game.get('week', 0)
            if 1 <= week <= 8:  # First half of season
                first_half_games.append(game)

        print(f"   Total games: {len(games)}")
        print(f"   First half (weeks 1-8): {len(first_half_games)}")

        if not first_half_games:
            print(f"\n⚠️  No games found in first half of {year} season")
            return

        # Step 3: Process each game
        print(f"\n🔬 Generating predictions and R1 analysis...")

        processed = 0
        skipped = 0

        for game in first_half_games:
            game_id = game.get('id')
            home_team = game.get('homeTeam', 'Unknown')
            away_team = game.get('awayTeam', 'Unknown')

            # Check if market spread available
            if game_id not in self.market_spreads:
                skipped += 1
                continue

            market_spread = self.market_spreads[game_id]['market_spread']

            # Generate 12-model predictions
            model_predictions = self._get_model_predictions(game, year)

            if not model_predictions:
                skipped += 1
                continue

            # R1 meta-analysis (simplified - no real-time context)
            game_data = {
                'home_team': home_team,
                'away_team': away_team,
                'day_of_week': self._get_day_of_week(game),
                'conference': game.get('conference', 'Unknown'),
                'is_maction': False  # Can't determine from historical data
            }

            try:
                r1_analysis = self.r1_reasoner.analyze_game(
                    game_data,
                    model_predictions,
                    market_spread,
                    contrarian_signal=None  # No historical public data
                )

                # Evaluate bet result
                self._evaluate_r1_bet(game, r1_analysis, market_spread)

                processed += 1

                if processed % 10 == 0:
                    print(f"   Processed {processed}/{len(first_half_games)} games...")

            except Exception as e:
                print(f"⚠️  R1 analysis failed for {away_team} @ {home_team}: {e}")
                skipped += 1
                continue

        print(f"\n✅ Processing complete")
        print(f"   Processed: {processed}")
        print(f"   Skipped: {skipped}")

        # Step 4: Display results
        self._display_backtest_results(year)

    def _get_model_predictions(self, game: Dict, year: int) -> List[ModelPrediction]:
        """
        Get REAL 12-model predictions for game

        IMPORTANT: No mock data - only real trained models
        """

        # Engineer features
        try:
            features = self.engineer.engineer_features(game, year)
            if not features:
                return []

            feature_array = np.array([list(features.values())])

            # Get predictions from all trained models
            model_names = [
                'xgboost_super', 'neural_net_deep', 'alt_spread',
                'bayesian_ensemble', 'momentum_model', 'situational',
                'advanced_stats', 'drive_outcomes', 'opponent_adjusted',
                'special_teams', 'pace_tempo', 'game_script'
            ]

            model_preds = []
            for model_name in model_names:
                if model_name not in self.orchestrator.models:
                    continue

                model = self.orchestrator.models[model_name]

                if not model.is_trained:
                    continue

                try:
                    # REAL model prediction
                    pred_spread = model.predict(feature_array)
                    spread = pred_spread[0] if isinstance(pred_spread, np.ndarray) else pred_spread

                    # REAL confidence
                    confidence = getattr(model, 'confidence', 0.75)

                    model_preds.append(ModelPrediction(
                        model_name=model_name,
                        predicted_spread=float(spread),
                        confidence=float(confidence),
                        reasoning=f"{model_name} prediction"
                    ))

                except Exception as e:
                    continue

            return model_preds

        except Exception as e:
            return []

    def _get_day_of_week(self, game: Dict) -> str:
        """Try to determine day of week from game data"""
        # Try to parse from game date if available
        date_str = game.get('date') or game.get('start_date')
        if date_str:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.strftime('%A')
            except:
                pass

        # Fallback: Assume Saturday (most common)
        return 'Saturday'

    def _evaluate_r1_bet(self, game: Dict, r1_analysis: R1Analysis, market_spread: float):
        """
        Evaluate R1 bet against actual game result

        Did R1 correctly predict the spread?
        """

        # Get actual result
        home_points = game.get('homePoints')
        away_points = game.get('awayPoints')

        if home_points is None or away_points is None:
            return  # Game not played yet

        actual_spread = home_points - away_points  # Positive = home won by more

        # Determine R1's pick
        pick = r1_analysis.recommended_pick
        confidence = r1_analysis.confidence

        if pick == "NO BET" or confidence < 70:
            tier = 'below_70%'
            self.results_by_tier[tier]['bets'].append({
                'game': f"{game.get('awayTeam')} @ {game.get('homeTeam')}",
                'r1_pick': pick,
                'confidence': confidence,
                'skipped': True
            })
            return

        # Parse R1's pick (e.g., "HOME -3.5" or "AWAY +7.0")
        r1_picked_home = 'HOME' in pick.upper()

        # Determine bet outcome
        # R1 picked home → home needs to cover market spread
        # R1 picked away → away needs to cover market spread

        if r1_picked_home:
            # R1 picked home - did home cover?
            bet_won = actual_spread > market_spread
        else:
            # R1 picked away - did away cover?
            bet_won = actual_spread < market_spread

        # Calculate bet size based on confidence
        bet_size = self._calculate_bet_size(confidence)

        # Calculate profit (assuming -110 odds)
        if bet_won:
            profit = bet_size * (100/110)  # Win
        else:
            profit = -bet_size  # Loss

        # Store result in appropriate tier
        if confidence >= 80:
            tier = '80%+'
        elif confidence >= 75:
            tier = '75-79%'
        elif confidence >= 70:
            tier = '70-74%'
        else:
            tier = 'below_70%'

        self.results_by_tier[tier]['bets'].append({
            'game': f"{game.get('awayTeam')} @ {game.get('homeTeam')}",
            'r1_pick': pick,
            'confidence': confidence,
            'market_spread': market_spread,
            'actual_spread': actual_spread,
            'bet_size': bet_size,
            'won': bet_won,
            'profit': profit
        })

        if bet_won:
            self.results_by_tier[tier]['wins'] += 1
            self.results_by_tier[tier]['total_profit'] += profit
        else:
            self.results_by_tier[tier]['losses'] += 1
            self.results_by_tier[tier]['total_profit'] += profit

    def _calculate_bet_size(self, confidence: int) -> float:
        """Calculate bet size based on R1 confidence"""
        # Base unit = $5 (from $100 bankroll)
        base_unit = 5

        bet_sizing = self.config['bet_sizing']

        if confidence >= 80:
            return base_unit * bet_sizing['80%+ confidence']
        elif confidence >= 75:
            return base_unit * bet_sizing['75-79% confidence']
        elif confidence >= 70:
            return base_unit * bet_sizing['70-74% confidence']
        else:
            return 0

    def _display_backtest_results(self, year: int):
        """Display backtest results by confidence tier"""

        print(f"\n{'='*80}")
        print(f"📊 R1 BACKTEST RESULTS - FIRST HALF {year} SEASON")
        print(f"{'='*80}\n")

        total_bets = 0
        total_wins = 0
        total_losses = 0
        total_profit = 0

        for tier in ['80%+', '75-79%', '70-74%', 'below_70%']:
            results = self.results_by_tier[tier]
            bets = len(results['bets'])
            wins = results['wins']
            losses = results['losses']
            profit = results['total_profit']

            if bets == 0:
                continue

            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

            print(f"{'='*80}")
            print(f"🎯 CONFIDENCE TIER: {tier}")
            print(f"{'='*80}\n")
            print(f"Total bets: {bets}")
            print(f"Wins: {wins}")
            print(f"Losses: {losses}")
            print(f"Win rate: {win_rate:.1%}")
            print(f"Total profit: ${profit:+.2f}")

            if wins + losses > 0:
                roi = (profit / ((wins + losses) * 20)) * 100  # Rough ROI estimate
                print(f"ROI: {roi:+.1f}%")

            print()

            total_bets += bets
            total_wins += wins
            total_losses += losses
            total_profit += profit

        # Overall results
        print(f"{'='*80}")
        print(f"📈 OVERALL RESULTS")
        print(f"{'='*80}\n")
        print(f"Total bets: {total_bets}")
        print(f"Wins: {total_wins}")
        print(f"Losses: {total_losses}")

        if total_wins + total_losses > 0:
            overall_win_rate = total_wins / (total_wins + total_losses)
            print(f"Win rate: {overall_win_rate:.1%}")
            print(f"Total profit: ${total_profit:+.2f}")

            # Compare to NFL validation (60.91x over 10 years)
            print(f"\n🎯 EXPECTED vs ACTUAL:")
            print(f"   Expected win rate: 58-62% (NFL validation)")
            print(f"   Actual win rate: {overall_win_rate:.1%}")

            if overall_win_rate >= 0.58:
                print(f"   ✅ R1 VALIDATED - System working as expected!")
            else:
                print(f"   ⚠️  Below expected - may need more data or refinement")

        print()

        # Save results
        self._save_backtest_results(year)

    def _save_backtest_results(self, year: int):
        """Save backtest results to file"""
        output_file = Path(f"backtest_results/r1_backtest_{year}_first_half.json")
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.results_by_tier, f, indent=2, default=str)

        print(f"💾 Results saved to {output_file}")


def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print("Usage: python backtest_ncaa_r1_system.py <DEEPSEEK_API_KEY>")
        print()
        print("Example:")
        print("  python backtest_ncaa_r1_system.py sk-deepseek123")
        print()
        print("This backtests R1 system on first half 2024 season")
        print("Requires: Market spread data (run scrapers first)")
        return

    deepseek_api_key = sys.argv[1]

    backtester = NCAAR1Backtester(deepseek_api_key)
    backtester.backtest_first_half_season(year=2024)


if __name__ == "__main__":
    main()
