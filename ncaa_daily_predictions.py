#!/usr/bin/env python3
"""
NCAA Daily Predictions - Run Every Day!
========================================

NCAA games happen Tuesday-Saturday, bringing fresh opportunities DAILY:
- Tuesday: MACtion, midweek specials (5-10 games)
- Wednesday: MACtion continues (5-10 games)
- Thursday: Conference USA, Sun Belt (10-15 games)
- Friday: Pac-12 after dark, Big Ten (15-20 games)
- Saturday: MAIN SLATE - All conferences (50-70 games!)

USAGE:
    # Run this DAILY during season:
    python ncaa_daily_predictions.py YOUR_API_KEY

    # Or set up cron job to run every day at 9am:
    0 9 * * * cd /path/to/football_betting_system && python ncaa_daily_predictions.py YOUR_KEY
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import requests
import sys

from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator


class NCAADailyPredictions:
    """
    Daily NCAA predictions - checks for games in next 48 hours
    Runs every day to catch Tuesday-Saturday games
    """

    def __init__(self, odds_api_key: str):
        self.odds_api_key = odds_api_key
        self.models_dir = Path("models/ncaa")
        self.predictions_dir = Path("data/daily_predictions")
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        self.engineer = NCAAFeatureEngineer()
        self.orchestrator = SuperIntelligenceOrchestrator(models_dir=str(self.models_dir))

        self._load_models()

        print(f"✅ Daily Prediction System initialized")
        print(f"   Models loaded: {self.models_loaded}/3")

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

    def fetch_upcoming_games(self) -> List[Dict]:
        """
        Fetch games happening in next 48 hours
        This catches Tuesday-Saturday games as they appear
        """
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds/"

        params = {
            'apiKey': self.odds_api_key,
            'regions': 'us',
            'markets': 'spreads,h2h',
            'oddsFormat': 'american',
            'bookmakers': 'fanduel',
        }

        print(f"\n📡 Fetching upcoming NCAA games...")

        try:
            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()

                remaining = response.headers.get('x-requests-remaining', 'Unknown')
                print(f"   API requests remaining: {remaining}")

                # Filter to games in next 48 hours
                now = datetime.utcnow()
                upcoming = []

                for game in data:
                    try:
                        # Parse commence time
                        commence_str = game['commence_time']
                        commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))

                        # Check if within next 48 hours
                        time_until = commence_time - now
                        hours_until = time_until.total_seconds() / 3600

                        if 0 < hours_until <= 48:
                            home_team = game['home_team']
                            away_team = game['away_team']

                            # Get spread
                            spread = None
                            odds_value = None

                            for bookmaker in game.get('bookmakers', []):
                                for market in bookmaker.get('markets', []):
                                    if market['key'] == 'spreads':
                                        for outcome in market['outcomes']:
                                            if outcome['name'] == home_team:
                                                spread = outcome['point']
                                                odds_value = outcome['price']
                                                break
                                        break
                                if spread is not None:
                                    break

                            if spread is not None:
                                # Determine day of week
                                day_of_week = commence_time.strftime('%A')

                                upcoming.append({
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'market_spread': spread,
                                    'odds': odds_value,
                                    'commence_time': commence_time,
                                    'hours_until': hours_until,
                                    'day_of_week': day_of_week,
                                    'source': 'the_odds_api'
                                })

                    except Exception as e:
                        continue

                # Group by day
                by_day = {}
                for game in upcoming:
                    day = game['day_of_week']
                    if day not in by_day:
                        by_day[day] = []
                    by_day[day].append(game)

                print(f"   ✅ Found {len(upcoming)} games in next 48 hours")
                print(f"\n   Games by day:")
                for day, games in sorted(by_day.items()):
                    print(f"      {day}: {len(games)} games")

                return upcoming

            elif response.status_code == 401:
                print(f"   ❌ Invalid API key")
                return []

            elif response.status_code == 429:
                print(f"   ❌ Rate limit exceeded")
                return []

            else:
                print(f"   ❌ Error: {response.status_code}")
                return []

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return []

    def generate_predictions_for_games(self, market_games: List[Dict], year: int) -> List[Dict]:
        """Generate predictions for specific games with market data"""
        print(f"\n🎯 Generating predictions...")

        predictions = []

        for market_game in market_games:
            try:
                home_team = market_game['home_team']
                away_team = market_game['away_team']

                # Load season data to find this game
                games = self.engineer.load_season_data(year)

                # Find matching game
                game_data = None
                for g in games:
                    if (self._team_match(g.get('homeTeam', ''), home_team) and
                        self._team_match(g.get('awayTeam', ''), away_team)):
                        game_data = g
                        break

                if not game_data:
                    # Game not in our data yet - might be too new
                    continue

                # Engineer features
                features = self.engineer.engineer_features(game_data, year)
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

                raw_confidence = max(0.5, min(0.95, 1 - (spread_std / 15.0)))
                confidence = raw_confidence * 0.90

                # Calculate edge
                market_spread = market_game['market_spread']
                edge = abs(predicted_spread - market_spread) / 7.0

                # Determine bet side
                if predicted_spread > market_spread:
                    bet_side = 'home'
                    bet_reasoning = f"We think home wins by {predicted_spread:.1f}, market only {market_spread:.1f}"
                else:
                    bet_side = 'away'
                    bet_reasoning = f"We think away covers, market has them at {market_spread:.1f}"

                predictions.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'our_spread': round(predicted_spread, 1),
                    'market_spread': market_spread,
                    'market_odds': market_game['odds'],
                    'edge': round(edge * 100, 1),
                    'confidence': round(confidence * 100, 0),
                    'bet_side': bet_side,
                    'bet_reasoning': bet_reasoning,
                    'commence_time': market_game['commence_time'].isoformat(),
                    'hours_until': round(market_game['hours_until'], 1),
                    'day_of_week': market_game['day_of_week'],
                })

            except Exception as e:
                continue

        print(f"   ✅ Generated {len(predictions)} predictions")
        return predictions

    def _team_match(self, name1: str, name2: str) -> bool:
        """Fuzzy team name matching"""
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()

        if name1 == name2:
            return True

        if name1 in name2 or name2 in name1:
            return True

        return False

    def display_daily_opportunities(self, predictions: List[Dict]):
        """Display today's and tomorrow's betting opportunities"""
        print(f"\n{'='*80}")
        print(f"📊 DAILY BETTING OPPORTUNITIES")
        print(f"{'='*80}\n")

        if not predictions:
            print("⚠️  No games with sufficient edge in next 48 hours")
            print("   Check back tomorrow!")
            return

        # Filter to games with edge
        bettable = [p for p in predictions if p['edge'] >= 5.0]

        if not bettable:
            print("⚠️  No games with 5%+ edge in next 48 hours")
            print(f"   Total games analyzed: {len(predictions)}")
            return

        # Sort by day, then edge
        bettable.sort(key=lambda x: (x['commence_time'], -x['edge']))

        # Group by day
        by_day = {}
        for pred in bettable:
            day = pred['day_of_week']
            if day not in by_day:
                by_day[day] = []
            by_day[day].append(pred)

        # Display by day
        for day, games in by_day.items():
            print(f"🗓️  {day.upper()} ({len(games)} opportunities)")
            print("-" * 80)

            for i, pred in enumerate(games, 1):
                print(f"\n{i}. {pred['away_team']} @ {pred['home_team']}")
                print(f"   Kickoff: {pred['hours_until']} hours")
                print(f"   Our Spread: {pred['our_spread']:+.1f}")
                print(f"   Market: {pred['market_spread']:+.1f} ({pred['market_odds']:+d})")
                print(f"   EDGE: {pred['edge']:.1f}% | Confidence: {pred['confidence']:.0f}%")
                print(f"   🎯 BET {pred['bet_side'].upper()}: {pred['bet_reasoning']}")

            print()

        print("="*80)
        print(f"💡 BETTING TIPS:")
        print("="*80)
        print()
        print("Edge Guidelines:")
        print("  • 5-10% edge → Small bet (1% bankroll)")
        print("  • 10-15% edge → Medium bet (2% bankroll)")
        print("  • 15%+ edge → Strong bet (3% bankroll)")
        print()
        print(f"Total opportunities today: {len(bettable)}")
        print(f"Recommended bets: {len([p for p in bettable if p['edge'] >= 10])}")
        print()

    def save_daily_predictions(self, predictions: List[Dict]):
        """Save predictions for tracking"""
        today = datetime.now().strftime("%Y%m%d")
        filename = self.predictions_dir / f"predictions_{today}.json"

        with open(filename, 'w') as f:
            json.dump({
                'date': today,
                'timestamp': datetime.now().isoformat(),
                'predictions': predictions,
                'count': len(predictions),
            }, f, indent=2)

        print(f"💾 Saved to {filename}")


def main():
    """Main entry point for daily predictions"""
    print("="*80)
    print("🏈 NCAA DAILY PREDICTIONS - Tuesday-Saturday Action!")
    print("="*80)
    print()
    print("NCAA games happen EVERY DAY Tue-Sat:")
    print("  • Tuesday: MACtion (5-10 games)")
    print("  • Wednesday: MACtion continues (5-10 games)")
    print("  • Thursday: Conference USA, Sun Belt (10-15 games)")
    print("  • Friday: Pac-12, Big Ten (15-20 games)")
    print("  • Saturday: MAIN SLATE - All conferences (50-70 games!)")
    print()

    if len(sys.argv) < 2:
        print("❌ No API key provided")
        print()
        print("Usage: python ncaa_daily_predictions.py YOUR_API_KEY")
        print()
        return

    api_key = sys.argv[1]
    year = 2025  # Current season

    system = NCAADailyPredictions(odds_api_key=api_key)

    # Fetch upcoming games (next 48 hours)
    upcoming_games = system.fetch_upcoming_games()

    if not upcoming_games:
        print("\n⚠️  No games in next 48 hours")
        print("   This is normal on Sunday/Monday")
        print("   Check back Tuesday for weekday action!")
        return

    # Generate predictions
    predictions = system.generate_predictions_for_games(upcoming_games, year)

    if not predictions:
        print("\n⚠️  No predictions available")
        print("   Games might be too new (not in our data yet)")
        return

    # Display opportunities
    system.display_daily_opportunities(predictions)

    # Save for tracking
    system.save_daily_predictions(predictions)

    print("\n" + "="*80)
    print("✅ DAILY CHECK COMPLETE")
    print("="*80)
    print()
    print("💡 Pro Tip: Run this script EVERY DAY at 9am to catch all opportunities!")
    print()
    print("Set up cron job:")
    print("0 9 * * * cd /path/to/football_betting_system && python ncaa_daily_predictions.py YOUR_KEY")
    print()


if __name__ == "__main__":
    main()
