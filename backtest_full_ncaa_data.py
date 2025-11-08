#!/usr/bin/env python3
"""
Full NCAA Backtest using collected College Football Data API data
Tests the system on 7,331+ real games from 2023-2024
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class FullNCAABacktester:
    """Comprehensive NCAA backtester using real collected data"""

    def __init__(self, bankroll=10000, unit_size=100, min_edge=0.03, min_confidence=0.60):
        self.bankroll_start = bankroll
        self.bankroll = bankroll
        self.unit_size = unit_size
        self.min_edge = min_edge
        self.min_confidence = min_confidence

        # Load SP+ ratings if available
        self.sp_ratings = {}

    def load_sp_ratings(self, year):
        """Load SP+ ratings for a season"""
        sp_file = Path(f"data/football/historical/ncaaf/ncaaf_{year}_sp_ratings.json")

        if sp_file.exists():
            with open(sp_file) as f:
                sp_data = json.load(f)

            # Create lookup dict
            ratings = {}
            for team in sp_data:
                team_name = team.get('team', '')
                rating = team.get('rating', 0)
                if team_name and rating:
                    ratings[team_name] = rating

            print(f"  ✅ Loaded SP+ ratings for {len(ratings)} teams")
            return ratings
        else:
            print(f"  ⚠️  No SP+ ratings found for {year}")
            return {}

    def calculate_game_edge(self, game, sp_ratings):
        """Calculate edge for a game using SP+ ratings"""
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')

        # Get SP+ ratings
        home_sp = sp_ratings.get(home_team, 0)
        away_sp = sp_ratings.get(away_team, 0)

        # Home field advantage (~3 points)
        home_advantage = 3.0

        # Predicted margin (positive = home favored)
        predicted_margin = (home_sp - away_sp) + home_advantage

        # Convert to win probability
        # SP+ margin roughly equals expected point spread
        # Use logistic function to convert to probability
        win_prob = 1 / (1 + np.exp(-predicted_margin / 10))

        # Calculate edge vs assumed market efficiency
        # Market is typically 52.4% efficient (vig removed)
        market_prob = 0.524
        edge = abs(win_prob - 0.5) - (market_prob - 0.5)
        edge = max(0, edge)

        # Confidence is based on margin magnitude
        confidence = min(0.95, 0.5 + abs(predicted_margin) / 50)

        return {
            'edge': edge,
            'confidence': confidence,
            'predicted_margin': predicted_margin,
            'win_prob': win_prob,
            'home_sp': home_sp,
            'away_sp': away_sp
        }

    def determine_winner(self, game):
        """Determine actual winner from game data"""
        home_score = game.get('home_points', game.get('home_score', 0))
        away_score = game.get('away_points', game.get('away_score', 0))

        if home_score and away_score:
            return int(home_score > away_score)

        # Check for winner field
        if 'winner' in game:
            return 1 if game['winner'] == game.get('home_team') else 0

        return None  # Can't determine

    def run_backtest(self, seasons=[2023, 2024]):
        """Run backtest across multiple seasons"""

        print("\n" + "="*70)
        print("🏈 COMPREHENSIVE NCAA BACKTEST")
        print("="*70)
        print(f"Seasons: {', '.join(map(str, seasons))}")
        print(f"Starting Bankroll: ${self.bankroll_start:,.2f}")
        print(f"Min Edge: {self.min_edge:.1%}")
        print(f"Min Confidence: {self.min_confidence:.0%}")

        all_bets = []
        total_games = 0
        games_with_complete_data = 0

        for year in seasons:
            print(f"\n{'='*70}")
            print(f"📊 SEASON {year}")
            print('='*70)

            # Load games
            games_file = Path(f"data/football/historical/ncaaf/ncaaf_{year}_games.json")
            if not games_file.exists():
                print(f"  ❌ No games file found: {games_file}")
                continue

            with open(games_file) as f:
                games = json.load(f)

            print(f"  ✅ Loaded {len(games)} games")
            total_games += len(games)

            # Load SP+ ratings
            sp_ratings = self.load_sp_ratings(year)

            # Process games
            season_bets = 0
            season_wins = 0

            for game in games:
                # Check if game is completed
                if game.get('completed') == False:
                    continue

                # Check if we have both teams in SP+
                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')

                if not home_team or not away_team:
                    continue

                if sp_ratings and (home_team not in sp_ratings or away_team not in sp_ratings):
                    continue

                games_with_complete_data += 1

                # Calculate edge
                prediction = self.calculate_game_edge(game, sp_ratings) if sp_ratings else {
                    'edge': 0.05,
                    'confidence': 0.65,
                    'predicted_margin': 0,
                    'win_prob': 0.5
                }

                edge = prediction['edge']
                confidence = prediction['confidence']

                # Skip if below thresholds
                if confidence < self.min_confidence or edge < self.min_edge:
                    continue

                # Determine actual result
                actual_winner = self.determine_winner(game)
                if actual_winner is None:
                    continue

                # Predict winner (bet on home if predicted margin positive)
                predicted_winner = 1 if prediction.get('predicted_margin', 0) > 0 else 0
                won = (predicted_winner == actual_winner)

                # Calculate stake using Kelly criterion
                kelly_fraction = edge / 0.5  # Simplified Kelly
                kelly_fraction = np.clip(kelly_fraction, 0.01, 0.15)  # Limit to 1-15% of bankroll
                stake = self.bankroll * kelly_fraction
                stake = min(stake, self.unit_size * 2.0)  # Cap at 2x unit
                stake = min(stake, self.bankroll * 0.15)  # Never bet more than 15% of bankroll

                if stake < 10:  # Minimum bet
                    continue

                # Calculate profit (assume -110 odds)
                profit = stake * 0.909 if won else -stake

                # Update bankroll
                self.bankroll += profit
                season_bets += 1
                if won:
                    season_wins += 1

                # Record bet
                all_bets.append({
                    'season': year,
                    'game': f"{away_team} @ {home_team}",
                    'home_team': home_team,
                    'away_team': away_team,
                    'predicted_margin': prediction.get('predicted_margin', 0),
                    'edge': edge,
                    'confidence': confidence,
                    'stake': stake,
                    'won': won,
                    'profit': profit,
                    'bankroll': self.bankroll,
                    'home_sp': prediction.get('home_sp', 0),
                    'away_sp': prediction.get('away_sp', 0)
                })

            # Season summary
            season_win_rate = season_wins / season_bets if season_bets > 0 else 0
            print(f"\n  📈 {year} Results:")
            print(f"     Bets Placed: {season_bets}")
            print(f"     Wins: {season_wins}")
            print(f"     Win Rate: {season_win_rate:.1%}")
            print(f"     Bankroll: ${self.bankroll:,.2f}")

        # Overall results
        self.display_results(all_bets, total_games, games_with_complete_data)

        return all_bets

    def display_results(self, bets, total_games, games_with_data):
        """Display comprehensive backtest results"""

        if not bets:
            print("\n❌ No bets placed! Check your thresholds.")
            return

        # Calculate metrics
        total_bets = len(bets)
        wins = sum(1 for b in bets if b['won'])
        losses = total_bets - wins
        win_rate = wins / total_bets

        total_profit = sum(b['profit'] for b in bets)
        roi = (total_profit / self.bankroll_start) * 100

        # Advanced metrics
        profits = [b['profit'] for b in bets]
        mean_profit = np.mean(profits)
        std_profit = np.std(profits, ddof=1)
        sharpe = (mean_profit / std_profit) * np.sqrt(len(profits)) if std_profit > 0 else 0

        # Max drawdown
        bankroll_curve = [b['bankroll'] for b in bets]
        peak = self.bankroll_start
        max_dd = 0
        max_dd_dollars = 0
        for value in bankroll_curve:
            peak = max(peak, value)
            dd_pct = (peak - value) / peak if peak > 0 else 0
            dd_dollars = peak - value
            if dd_pct > max_dd:
                max_dd = dd_pct
                max_dd_dollars = dd_dollars

        # Longest losing streak
        current_streak = 0
        max_losing_streak = 0
        for bet in bets:
            if not bet['won']:
                current_streak += 1
                max_losing_streak = max(max_losing_streak, current_streak)
            else:
                current_streak = 0

        # Results by season
        by_season = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit': 0})
        for bet in bets:
            season = bet['season']
            by_season[season]['bets'] += 1
            if bet['won']:
                by_season[season]['wins'] += 1
            by_season[season]['profit'] += bet['profit']

        # Display
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)

        print(f"\n📈 Game Coverage:")
        print(f"   Total Games Collected: {total_games:,}")
        print(f"   Games with Complete Data: {games_with_data:,}")
        print(f"   Bets Placed: {total_bets:,} ({total_bets/games_with_data*100:.1f}% of complete games)")

        print(f"\n💰 Performance:")
        print(f"   Starting Bankroll: ${self.bankroll_start:,.2f}")
        print(f"   Final Bankroll: ${self.bankroll:,.2f}")
        print(f"   Total Profit: ${total_profit:,.2f}")
        print(f"   ROI: {roi:.2f}%")

        print(f"\n🎯 Win Rate:")
        print(f"   Wins: {wins}")
        print(f"   Losses: {losses}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Break-even Rate: 52.4% (accounting for -110 vig)")

        print(f"\n📊 Risk Metrics:")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Max Drawdown: {max_dd:.1%} (${max_dd_dollars:,.2f})")
        print(f"   Max Losing Streak: {max_losing_streak} bets")
        print(f"   Avg Bet Size: ${np.mean([b['stake'] for b in bets]):,.2f}")

        print(f"\n📅 Results by Season:")
        for season in sorted(by_season.keys()):
            data = by_season[season]
            wr = data['wins'] / data['bets'] if data['bets'] > 0 else 0
            season_roi = (data['profit'] / self.bankroll_start) * 100
            print(f"   {season}: {data['bets']:3d} bets | Win Rate: {wr:.1%} | "
                  f"Profit: ${data['profit']:>8,.2f} | ROI: {season_roi:>6.2f}%")

        # Performance grade
        if roi > 18:
            grade = "🏆 EXCELLENT"
            desc = "Outstanding performance! System is highly profitable."
        elif roi > 10:
            grade = "🥇 GOOD"
            desc = "Strong performance. System shows consistent edge."
        elif roi > 5:
            grade = "🥈 AVERAGE"
            desc = "Modest profitability. Consider optimization."
        elif roi > 0:
            grade = "🥉 MARGINAL"
            desc = "Barely profitable. Needs significant improvement."
        else:
            grade = "❌ POOR"
            desc = "Losing system. Do not use for live betting."

        print(f"\n📈 Performance Grade: {grade}")
        print(f"   {desc}")

        # Recommendations
        print(f"\n💡 Recommendations:")

        if win_rate < 0.524:
            print("   ⚠️  Win rate below break-even. System needs improvement.")
        elif win_rate > 0.58:
            print("   ✅ Excellent win rate! System has strong predictive power.")

        if roi < 0:
            print("   ❌ System is losing money. DO NOT use for live betting.")
            print("   🔧 Suggestions:")
            print("      - Increase min_confidence threshold")
            print("      - Review SP+ rating integration")
            print("      - Focus on specific conferences or game types")
        elif roi > 15:
            print("   🎉 System is highly profitable!")
            print("   💰 Consider:")
            print("      - Upgrading to Gold tier for more advanced metrics")
            print("      - Increasing bet sizes gradually")
            print("      - Adding weather and injury data")
        else:
            print("   💡 System shows promise. Optimization ideas:")
            print("      - Tune confidence thresholds by conference")
            print("      - Add home/away splits")
            print("      - Consider weather integration")
            print("      - Analyze best performing game types")

        if total_bets < games_with_data * 0.10:
            print(f"   ⚠️  Low bet volume ({total_bets/games_with_data*100:.1f}% of games)")
            print("      Consider lowering min_edge or min_confidence thresholds")

        print("\n" + "="*70)


def main():
    """Run full backtest"""

    print("\n" + "="*70)
    print("🏈 NCAA FOOTBALL - COMPREHENSIVE BACKTEST")
    print("Using Real College Football Data API Data")
    print("="*70)

    # Initialize backtester
    backtester = FullNCAABacktester(
        bankroll=10000,
        unit_size=100,
        min_edge=0.03,      # 3% minimum edge
        min_confidence=0.60  # 60% minimum confidence
    )

    # Run backtest on 2023 and 2024 seasons
    bets = backtester.run_backtest(seasons=[2023, 2024])

    # Save results
    if bets:
        output_file = Path("ncaa_backtest_results.json")
        with open(output_file, 'w') as f:
            json.dump(bets, f, indent=2)
        print(f"\n💾 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
