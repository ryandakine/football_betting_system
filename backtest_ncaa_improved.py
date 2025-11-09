#!/usr/bin/env python3
"""
IMPROVED NCAA Backtest with Bug Fixes
Fixes edge calculation, Kelly criterion, adds conference analysis
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats


# Team name normalization (SP+ might use different names than games API)
TEAM_NAME_MAP = {
    'USC': 'Southern California',
    'Miami': 'Miami (FL)',
    'Miami FL': 'Miami (FL)',
    'UCF': 'Central Florida',
    'SMU': 'Southern Methodist',
    'BYU': 'Brigham Young',
    'UNLV': 'Nevada Las Vegas',
    'UMass': 'Massachusetts',
    'UConn': 'Connecticut',
    'FIU': 'Florida International',
    'FAU': 'Florida Atlantic',
    'UAB': 'Alabama Birmingham',
    'UTSA': 'Texas San Antonio',
    'UTEP': 'Texas El Paso',
    'USF': 'South Florida',
}


class ImprovedNCAABacktester:
    """NCAA backtester with bug fixes and enhancements"""

    def __init__(self, bankroll=10000, unit_size=100, min_edge=0.03, min_confidence=0.60):
        self.bankroll_start = bankroll
        self.bankroll = bankroll
        self.unit_size = unit_size
        self.min_edge = min_edge
        self.min_confidence = min_confidence

    def normalize_team_name(self, name):
        """Normalize team names for SP+ lookup"""
        if not name:
            return ''
        # Try exact match first
        if name in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[name]
        # Try reverse lookup
        reverse_map = {v: k for k, v in TEAM_NAME_MAP.items()}
        if name in reverse_map:
            return reverse_map[name]
        return name

    def load_sp_ratings(self, year):
        """Load SP+ ratings with fuzzy team name matching"""
        sp_file = Path(f"data/football/historical/ncaaf/ncaaf_{year}_sp_ratings.json")

        if not sp_file.exists():
            print(f"  ⚠️  No SP+ ratings found for {year}")
            return {}

        with open(sp_file) as f:
            sp_data = json.load(f)

        # Create lookup dict with normalized names
        ratings = {}
        for team in sp_data:
            team_name = team.get('team', '')
            rating = team.get('rating', 0)
            if team_name and rating:
                # Store under both original and normalized name
                ratings[team_name] = rating
                normalized = self.normalize_team_name(team_name)
                if normalized != team_name:
                    ratings[normalized] = rating

        print(f"  ✅ Loaded SP+ ratings for {len(set(ratings.values()))} teams")
        return ratings

    def calculate_game_edge(self, game, sp_ratings):
        """
        FIXED: Calculate edge correctly using proper probability math
        """
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')

        # Try to get SP+ ratings with normalization
        home_sp = sp_ratings.get(home_team, sp_ratings.get(self.normalize_team_name(home_team), 0))
        away_sp = sp_ratings.get(away_team, sp_ratings.get(self.normalize_team_name(away_team), 0))

        # Home field advantage (varies, but ~2.5-3.5 points average)
        home_advantage = 3.0

        # Predicted margin (positive = home favored)
        predicted_margin = (home_sp - away_sp) + home_advantage

        # Convert margin to win probability using logistic function
        # Research shows ~14 points = 90% win probability in college football
        win_prob = 1 / (1 + np.exp(-predicted_margin / 14))

        # FIXED: Edge calculation
        # Assuming -110 odds on both sides: break-even = 52.38%
        # If we predict 60% and market implies 52.4%, our edge is 7.6%
        market_implied_prob = 0.5238  # -110 odds break-even

        if win_prob > 0.5:
            # We're betting on home team
            edge = win_prob - market_implied_prob
        else:
            # We're betting on away team
            edge = (1 - win_prob) - market_implied_prob

        edge = max(0, edge)  # Never negative

        # Confidence based on SP+ differential magnitude
        sp_diff = abs(home_sp - away_sp)
        confidence = min(0.95, 0.5 + (sp_diff / 40))  # More differential = more confident

        return {
            'edge': edge,
            'confidence': confidence,
            'predicted_margin': predicted_margin,
            'win_prob': win_prob,
            'home_sp': home_sp,
            'away_sp': away_sp,
            'bet_on_home': win_prob > 0.5
        }

    def calculate_kelly_stake(self, edge, win_prob):
        """
        FIXED: Proper Kelly Criterion for -110 odds
        Kelly = (p * b - (1-p)) / b
        where p = win probability, b = net odds (100/110 = 0.909 for -110)
        """
        # For -110 odds: you risk 110 to win 100
        # Net odds: 100/110 = 0.909
        b = 0.909

        kelly_fraction = (win_prob * b - (1 - win_prob)) / b

        # Safety limits
        kelly_fraction = max(0, kelly_fraction)  # Never bet negative edge
        kelly_fraction = min(0.25, kelly_fraction)  # Never bet more than 25% (full Kelly is risky)

        return kelly_fraction

    def determine_winner(self, game):
        """Determine actual winner, trying multiple field names"""
        # Try home_points first
        home_score = game.get('home_points')
        away_score = game.get('away_points')

        # Fallback to home_score
        if home_score is None:
            home_score = game.get('home_score')
        if away_score is None:
            away_score = game.get('away_score')

        if home_score is not None and away_score is not None:
            return 1 if home_score > away_score else 0

        # Check winner field
        if 'home_id' in game and 'away_id' in game:
            winner_id = game.get('winner')
            if winner_id:
                return 1 if winner_id == game['home_id'] else 0

        return None

    def get_game_conference(self, game):
        """Extract conference from game data"""
        # Try multiple field names
        return (game.get('conference') or
                game.get('home_conference') or
                game.get('away_conference') or
                'Unknown')

    def run_backtest(self, seasons=[2023, 2024]):
        """Run improved backtest with detailed analytics"""

        print("\n" + "="*70)
        print("🏈 IMPROVED NCAA BACKTEST (Bug Fixes Applied)")
        print("="*70)
        print(f"Seasons: {', '.join(map(str, seasons))}")
        print(f"Starting Bankroll: ${self.bankroll_start:,.2f}")
        print(f"Min Edge: {self.min_edge:.1%}")
        print(f"Min Confidence: {self.min_confidence:.0%}")

        all_bets = []
        total_games = 0
        games_with_sp = 0
        skipped_no_sp = 0

        # Track by various categories
        by_conference = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit': 0})
        by_season = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit': 0})
        by_bet_type = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit': 0})

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

            if not sp_ratings:
                print(f"  ⚠️  Cannot backtest {year} without SP+ ratings")
                continue

            # Process games
            for game in games:
                # Skip incomplete games
                if game.get('completed') == False:
                    continue

                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')

                if not home_team or not away_team:
                    continue

                # Check SP+ availability (with normalization)
                home_sp_key = home_team if home_team in sp_ratings else self.normalize_team_name(home_team)
                away_sp_key = away_team if away_team in sp_ratings else self.normalize_team_name(away_team)

                if home_sp_key not in sp_ratings or away_sp_key not in sp_ratings:
                    skipped_no_sp += 1
                    continue

                games_with_sp += 1

                # Calculate prediction
                prediction = self.calculate_game_edge(game, sp_ratings)

                edge = prediction['edge']
                confidence = prediction['confidence']
                win_prob = prediction['win_prob']

                # Filter by thresholds
                if confidence < self.min_confidence or edge < self.min_edge:
                    continue

                # Determine actual result
                actual_winner = self.determine_winner(game)
                if actual_winner is None:
                    continue

                # Determine our bet
                bet_on_home = prediction['bet_on_home']
                predicted_winner = 1 if bet_on_home else 0
                won = (predicted_winner == actual_winner)

                # FIXED: Proper Kelly Criterion
                kelly_fraction = self.calculate_kelly_stake(edge, win_prob)

                # Calculate stake
                stake = self.bankroll * kelly_fraction
                stake = max(stake, 10)  # Minimum $10 bet
                stake = min(stake, self.unit_size * 3.0)  # Cap at 3x unit
                stake = min(stake, self.bankroll * 0.20)  # Never more than 20% of bankroll
                stake = min(stake, self.bankroll)  # Never more than bankroll

                # Calculate profit (assuming -110 odds)
                profit = stake * 0.909 if won else -stake

                # Update bankroll
                self.bankroll += profit

                # Get conference
                conference = self.get_game_conference(game)

                # Track bet
                bet_record = {
                    'season': year,
                    'game': f"{away_team} @ {home_team}",
                    'home_team': home_team,
                    'away_team': away_team,
                    'conference': conference,
                    'bet_on': 'home' if bet_on_home else 'away',
                    'predicted_margin': prediction['predicted_margin'],
                    'edge': edge,
                    'confidence': confidence,
                    'win_prob': win_prob,
                    'stake': stake,
                    'won': won,
                    'profit': profit,
                    'bankroll': self.bankroll,
                    'home_sp': prediction['home_sp'],
                    'away_sp': prediction['away_sp']
                }
                all_bets.append(bet_record)

                # Update category trackers
                by_season[year]['bets'] += 1
                by_season[year]['profit'] += profit
                if won:
                    by_season[year]['wins'] += 1

                by_conference[conference]['bets'] += 1
                by_conference[conference]['profit'] += profit
                if won:
                    by_conference[conference]['wins'] += 1

                bet_type = 'home' if bet_on_home else 'away'
                by_bet_type[bet_type]['bets'] += 1
                by_bet_type[bet_type]['profit'] += profit
                if won:
                    by_bet_type[bet_type]['wins'] += 1

            # Season summary
            season_bets = by_season[year]['bets']
            season_wins = by_season[year]['wins']
            season_wr = season_wins / season_bets if season_bets > 0 else 0

            print(f"\n  📈 {year} Results:")
            print(f"     Games with SP+ data: {games_with_sp}")
            print(f"     Bets Placed: {season_bets}")
            print(f"     Wins: {season_wins}")
            print(f"     Win Rate: {season_wr:.1%}")
            print(f"     Bankroll: ${self.bankroll:,.2f}")

        # Display comprehensive results
        self.display_results(all_bets, total_games, games_with_sp, skipped_no_sp,
                           by_conference, by_season, by_bet_type)

        return all_bets

    def display_results(self, bets, total_games, games_with_sp, skipped_no_sp,
                       by_conference, by_season, by_bet_type):
        """Display comprehensive results with statistical analysis"""

        if not bets:
            print("\n❌ No bets placed! Check your thresholds.")
            return

        # Basic metrics
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

        # Statistical significance
        t_stat, p_value = stats.ttest_1samp(profits, 0)

        # Max drawdown
        bankroll_curve = [self.bankroll_start] + [b['bankroll'] for b in bets]
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

        # Longest streak
        current_streak = 0
        max_losing_streak = 0
        for bet in bets:
            if not bet['won']:
                current_streak += 1
                max_losing_streak = max(max_losing_streak, current_streak)
            else:
                current_streak = 0

        # Display
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)

        print(f"\n📈 Data Coverage:")
        print(f"   Total Games Collected: {total_games:,}")
        print(f"   Games with SP+ Data: {games_with_sp:,}")
        print(f"   Games Missing SP+: {skipped_no_sp:,}")
        pct_bet = (total_bets / games_with_sp * 100) if games_with_sp > 0 else 0
        print(f"   Bets Placed: {total_bets:,} ({pct_bet:.1f}% of games with SP+)")

        print(f"\n💰 Performance:")
        print(f"   Starting Bankroll: ${self.bankroll_start:,.2f}")
        print(f"   Final Bankroll: ${self.bankroll:,.2f}")
        print(f"   Total Profit: ${total_profit:,.2f}")
        print(f"   ROI: {roi:.2f}%")

        print(f"\n🎯 Win Rate:")
        print(f"   Wins: {wins}")
        print(f"   Losses: {losses}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Break-even Rate: 52.4% (for -110 odds)")
        if win_rate > 0.524:
            print(f"   ✅ Beating the vig by {(win_rate - 0.524)*100:.1f}%!")

        print(f"\n📊 Risk Metrics:")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Max Drawdown: {max_dd:.1%} (${max_dd_dollars:,.2f})")
        print(f"   Max Losing Streak: {max_losing_streak} bets")
        print(f"   Avg Bet Size: ${np.mean([b['stake'] for b in bets]):,.2f}")

        print(f"\n📊 Statistical Significance:")
        print(f"   T-Statistic: {t_stat:.2f}")
        print(f"   P-Value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"   ✅ Results are statistically significant (p < 0.05)")
        else:
            print(f"   ⚠️  Results not statistically significant - could be luck")

        # Season breakdown
        print(f"\n📅 Results by Season:")
        for season in sorted(by_season.keys()):
            data = by_season[season]
            if data['bets'] > 0:
                wr = data['wins'] / data['bets']
                season_roi = (data['profit'] / self.bankroll_start) * 100
                print(f"   {season}: {data['bets']:3d} bets | WR: {wr:.1%} | "
                      f"Profit: ${data['profit']:>8,.2f} | ROI: {season_roi:>6.2f}%")

        # Conference breakdown
        print(f"\n🏟️  Results by Conference:")
        conf_sorted = sorted(by_conference.items(), key=lambda x: x[1]['profit'], reverse=True)
        for conf, data in conf_sorted[:10]:  # Top 10 conferences
            if data['bets'] >= 5:  # Only show if enough sample
                wr = data['wins'] / data['bets']
                print(f"   {conf:20s}: {data['bets']:3d} bets | WR: {wr:.1%} | "
                      f"Profit: ${data['profit']:>8,.2f}")

        # Bet type breakdown
        print(f"\n🎲 Results by Bet Type:")
        for bet_type in ['home', 'away']:
            data = by_bet_type[bet_type]
            if data['bets'] > 0:
                wr = data['wins'] / data['bets']
                print(f"   {bet_type.title():10s}: {data['bets']:3d} bets | WR: {wr:.1%} | "
                      f"Profit: ${data['profit']:>8,.2f}")

        # Performance grade
        if roi > 18:
            grade = "🏆 EXCELLENT"
            desc = "Outstanding! System is highly profitable."
        elif roi > 10:
            grade = "🥇 GOOD"
            desc = "Strong performance. Consistent edge detected."
        elif roi > 5:
            grade = "🥈 AVERAGE"
            desc = "Modest profitability. Room for optimization."
        elif roi > 0:
            grade = "🥉 MARGINAL"
            desc = "Barely profitable. Significant improvement needed."
        else:
            grade = "❌ POOR"
            desc = "Losing system. DO NOT use for live betting."

        print(f"\n📈 Performance Grade: {grade}")
        print(f"   {desc}")

        # Recommendations
        print(f"\n💡 Recommendations:")

        if p_value >= 0.05:
            print("   ⚠️  Results not statistically significant - need more data")
            print("      Consider collecting more seasons (2015-2022)")

        if win_rate < 0.524:
            print("   ⚠️  Win rate below break-even. System needs work.")
        elif win_rate > 0.58:
            print("   ✅ Excellent win rate! Strong predictive power.")

        if roi > 15:
            print("   🎉 System is highly profitable!")
            print("   💰 Next steps:")
            print("      - Start with small real-money units")
            print("      - Track live results for 20-30 bets")
            print("      - Scale up if live matches backtest")
        elif roi > 5:
            print("   💡 System shows promise. Consider:")
            print("      - Test on more seasons (2015-2022)")
            print("      - Analyze best-performing conferences")
            print("      - Add weather data for outdoor games")
        else:
            print("   🔧 System needs improvement:")
            print("      - Increase min_confidence threshold")
            print("      - Focus on conferences with positive ROI")
            print("      - Consider ensemble with other models")

        print("\n" + "="*70)


def main():
    """Run improved backtest"""

    print("\n" + "="*70)
    print("🏈 NCAA FOOTBALL - IMPROVED BACKTEST")
    print("Bug Fixes: Edge Calculation, Kelly Criterion, Team Names")
    print("="*70)

    backtester = ImprovedNCAABacktester(
        bankroll=10000,
        unit_size=100,
        min_edge=0.03,       # 3% edge minimum
        min_confidence=0.60  # 60% confidence minimum
    )

    bets = backtester.run_backtest(seasons=[2023, 2024])

    if bets:
        output_file = Path("ncaa_backtest_improved_results.json")
        with open(output_file, 'w') as f:
            json.dump(bets, f, indent=2)
        print(f"\n💾 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
