#!/usr/bin/env python3
"""
Live NFL Odds Monitor
=====================
Real-time odds tracking across multiple sportsbooks.

Features:
- Multi-sportsbook monitoring (DraftKings, FanDuel, BetMGM, etc.)
- Line movement alerts
- Best available odds finder
- Arbitrage opportunity detection
- Historical odds tracking

Usage:
    python live_odds_monitor.py --week 10
    python live_odds_monitor.py --week 10 --track-game "BUF @ KC"
    python live_odds_monitor.py --week 10 --alert-threshold 1.5
"""

import argparse
import asyncio
import json
import os
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import our odds fetcher
from nfl_odds_integration import fetch_and_integrate_nfl_odds, NFLGameOdds


@dataclass
class OddsSnapshot:
    """Single odds snapshot from a sportsbook at a point in time."""
    timestamp: str
    bookmaker: str
    game_id: str
    home_team: str
    away_team: str
    spread_home: Optional[float]
    spread_home_odds: Optional[int]
    total: Optional[float]
    over_odds: Optional[int]
    under_odds: Optional[int]
    moneyline_home: Optional[int]
    moneyline_away: Optional[int]


@dataclass
class LineMovement:
    """Represents a significant line movement."""
    game: str
    bet_type: str
    bookmaker: str
    old_line: float
    new_line: float
    movement: float
    timestamp: str


@dataclass
class BestOdds:
    """Best available odds across all sportsbooks."""
    game: str
    bet_type: str
    best_line: float
    best_odds: int
    bookmaker: str
    all_books: List[Dict]


class LiveOddsMonitor:
    """Monitors live NFL odds across multiple sportsbooks."""

    def __init__(self, data_dir: str = "data/live_odds"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.snapshots: List[OddsSnapshot] = []
        self.previous_odds: Dict[str, Dict] = {}
        self.line_movements: List[LineMovement] = []

        # Alert thresholds
        self.spread_movement_threshold = 1.0  # 1 point
        self.total_movement_threshold = 1.5  # 1.5 points
        self.odds_movement_threshold = 20  # 20 cents

    async def fetch_current_odds(self, target_date: Optional[date] = None) -> List[NFLGameOdds]:
        """Fetch current odds from The Odds API."""
        print(f"🔍 Fetching live odds from The Odds API...")

        try:
            result = await fetch_and_integrate_nfl_odds(target_date)

            if not result or not result.get('odds'):
                print("⚠️  No odds found")
                return []

            odds_list = result['odds']
            print(f"✅ Found {len(odds_list)} odds entries from {len(set(o.bookmaker for o in odds_list))} bookmakers")

            return odds_list

        except Exception as e:
            print(f"❌ Error fetching odds: {e}")
            return []

    def create_snapshot(self, odds_list: List[NFLGameOdds]) -> List[OddsSnapshot]:
        """Create snapshots from odds data."""
        snapshots = []
        timestamp = datetime.now().isoformat()

        for odds in odds_list:
            snapshot = OddsSnapshot(
                timestamp=timestamp,
                bookmaker=odds.bookmaker,
                game_id=odds.game_id,
                home_team=odds.home_team,
                away_team=odds.away_team,
                spread_home=odds.spread_home,
                spread_home_odds=odds.spread_home_odds,
                total=odds.total,
                over_odds=odds.over_odds,
                under_odds=odds.under_odds,
                moneyline_home=odds.moneyline_home,
                moneyline_away=odds.moneyline_away
            )
            snapshots.append(snapshot)

        return snapshots

    def detect_line_movements(self, new_snapshots: List[OddsSnapshot]) -> List[LineMovement]:
        """Detect significant line movements from previous snapshot."""
        movements = []

        for snapshot in new_snapshots:
            key = f"{snapshot.game_id}_{snapshot.bookmaker}"

            if key in self.previous_odds:
                old = self.previous_odds[key]

                # Check spread movement
                if snapshot.spread_home and old.get('spread_home'):
                    movement = abs(snapshot.spread_home - old['spread_home'])
                    if movement >= self.spread_movement_threshold:
                        movements.append(LineMovement(
                            game=f"{snapshot.away_team} @ {snapshot.home_team}",
                            bet_type='SPREAD',
                            bookmaker=snapshot.bookmaker,
                            old_line=old['spread_home'],
                            new_line=snapshot.spread_home,
                            movement=snapshot.spread_home - old['spread_home'],
                            timestamp=snapshot.timestamp
                        ))

                # Check total movement
                if snapshot.total and old.get('total'):
                    movement = abs(snapshot.total - old['total'])
                    if movement >= self.total_movement_threshold:
                        movements.append(LineMovement(
                            game=f"{snapshot.away_team} @ {snapshot.home_team}",
                            bet_type='TOTAL',
                            bookmaker=snapshot.bookmaker,
                            old_line=old['total'],
                            new_line=snapshot.total,
                            movement=snapshot.total - old['total'],
                            timestamp=snapshot.timestamp
                        ))

            # Update previous odds
            self.previous_odds[key] = {
                'spread_home': snapshot.spread_home,
                'total': snapshot.total,
                'moneyline_home': snapshot.moneyline_home
            }

        return movements

    def find_best_odds(self, snapshots: List[OddsSnapshot]) -> Dict[str, List[BestOdds]]:
        """Find best available odds for each game and bet type."""
        best_odds_by_game = defaultdict(list)

        # Group by game
        games = {}
        for snapshot in snapshots:
            game_key = f"{snapshot.away_team} @ {snapshot.home_team}"
            if game_key not in games:
                games[game_key] = []
            games[game_key].append(snapshot)

        # Find best odds for each game
        for game, game_snapshots in games.items():
            # Best spread (lowest number favors away, highest favors home)
            spreads = [(s.spread_home, s.spread_home_odds, s.bookmaker)
                      for s in game_snapshots if s.spread_home is not None]

            if spreads:
                # Best for home bettors (highest spread)
                best_home_spread = max(spreads, key=lambda x: x[0])
                best_odds_by_game[game].append(BestOdds(
                    game=game,
                    bet_type='SPREAD_HOME',
                    best_line=best_home_spread[0],
                    best_odds=best_home_spread[1],
                    bookmaker=best_home_spread[2],
                    all_books=[{'book': b, 'line': l, 'odds': o} for l, o, b in spreads]
                ))

                # Best for away bettors (lowest spread)
                best_away_spread = min(spreads, key=lambda x: x[0])
                best_odds_by_game[game].append(BestOdds(
                    game=game,
                    bet_type='SPREAD_AWAY',
                    best_line=best_away_spread[0],
                    best_odds=best_away_spread[1],
                    bookmaker=best_away_spread[2],
                    all_books=[{'book': b, 'line': l, 'odds': o} for l, o, b in spreads]
                ))

            # Best total (lowest for under bettors, highest for over)
            totals = [(s.total, s.over_odds, s.under_odds, s.bookmaker)
                     for s in game_snapshots if s.total is not None]

            if totals:
                # Best for over bettors (lowest total)
                best_over = min(totals, key=lambda x: x[0])
                best_odds_by_game[game].append(BestOdds(
                    game=game,
                    bet_type='OVER',
                    best_line=best_over[0],
                    best_odds=best_over[1],
                    bookmaker=best_over[3],
                    all_books=[{'book': b, 'line': l, 'odds': o} for l, o, u, b in totals]
                ))

                # Best for under bettors (highest total)
                best_under = max(totals, key=lambda x: x[0])
                best_odds_by_game[game].append(BestOdds(
                    game=game,
                    bet_type='UNDER',
                    best_line=best_under[0],
                    best_odds=best_under[2],
                    bookmaker=best_under[3],
                    all_books=[{'book': b, 'line': l, 'odds': u} for l, o, u, b in totals]
                ))

        return dict(best_odds_by_game)

    def detect_arbitrage(self, snapshots: List[OddsSnapshot]) -> List[Dict]:
        """Detect arbitrage opportunities across sportsbooks."""
        arb_opportunities = []

        # Group by game
        games = defaultdict(list)
        for snapshot in snapshots:
            game_key = f"{snapshot.away_team} @ {snapshot.home_team}"
            games[game_key].append(snapshot)

        # Check for spread arbitrage
        for game, game_snapshots in games.items():
            # Get all spreads
            spreads = {}
            for s in game_snapshots:
                if s.spread_home and s.spread_home_odds:
                    spreads[s.bookmaker] = {
                        'home': (s.spread_home, s.spread_home_odds),
                        'away': (-s.spread_home, s.spread_home_odds)
                    }

            # Check for arbitrage
            for book1, odds1 in spreads.items():
                for book2, odds2 in spreads.items():
                    if book1 == book2:
                        continue

                    # Check if betting home on book1 and away on book2 is arbitrage
                    # This is simplified - real arbitrage calc is more complex
                    if odds1['home'][0] + odds2['away'][0] > 0:
                        arb_opportunities.append({
                            'game': game,
                            'type': 'SPREAD_ARBITRAGE',
                            'book1': book1,
                            'bet1': f"Home {odds1['home'][0]}",
                            'book2': book2,
                            'bet2': f"Away {odds2['away'][0]}",
                            'details': 'Potential middle opportunity'
                        })

        return arb_opportunities

    def save_snapshot(self, snapshots: List[OddsSnapshot]):
        """Save odds snapshot to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.data_dir / f"odds_snapshot_{timestamp}.json"

        data = [asdict(s) for s in snapshots]

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Saved snapshot: {filename}")

    def generate_report(
        self,
        snapshots: List[OddsSnapshot],
        movements: List[LineMovement],
        best_odds: Dict[str, List[BestOdds]],
        arb_opportunities: List[Dict]
    ) -> str:
        """Generate live odds monitoring report."""
        report = []
        report.append("=" * 80)
        report.append("🔴 LIVE NFL ODDS MONITOR")
        report.append("=" * 80)
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Sportsbooks Tracked: {len(set(s.bookmaker for s in snapshots))}")
        report.append(f"Games Monitored: {len(best_odds)}")
        report.append("")

        # Line movements
        if movements:
            report.append("🚨 RECENT LINE MOVEMENTS:")
            report.append("")
            for movement in movements[-10:]:  # Last 10
                direction = "⬆️" if movement.movement > 0 else "⬇️"
                report.append(f"{direction} {movement.game}")
                report.append(f"   {movement.bet_type} moved {movement.movement:+.1f}")
                report.append(f"   {movement.old_line} → {movement.new_line} ({movement.bookmaker})")
                report.append("")

        # Best odds
        report.append("=" * 80)
        report.append("💰 BEST AVAILABLE ODDS (Line Shopping)")
        report.append("=" * 80)
        report.append("")

        for game, odds_list in best_odds.items():
            report.append(f"🏈 {game}")
            report.append("")

            for best in odds_list:
                report.append(f"   {best.bet_type}: {best.best_line} ({best.best_odds:+d}) @ {best.bookmaker}")

            report.append("")

        # Arbitrage opportunities
        if arb_opportunities:
            report.append("=" * 80)
            report.append("⚡ ARBITRAGE OPPORTUNITIES")
            report.append("=" * 80)
            report.append("")

            for arb in arb_opportunities:
                report.append(f"🎯 {arb['game']}")
                report.append(f"   {arb['book1']}: {arb['bet1']}")
                report.append(f"   {arb['book2']}: {arb['bet2']}")
                report.append(f"   Type: {arb['type']}")
                report.append("")

        report.append("=" * 80)

        return "\n".join(report)


async def monitor_loop(week: int, interval: int = 300):
    """Continuous monitoring loop."""
    monitor = LiveOddsMonitor()

    print(f"\n🔴 Starting live odds monitor for Week {week}")
    print(f"Update interval: {interval} seconds")
    print(f"Press Ctrl+C to stop\n")

    cycle = 0

    try:
        while True:
            cycle += 1
            print(f"\n{'='*80}")
            print(f"Cycle #{cycle} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*80}\n")

            # Fetch current odds
            odds_list = await monitor.fetch_current_odds()

            if not odds_list:
                print("⚠️  No odds available, retrying in {interval} seconds...")
                await asyncio.sleep(interval)
                continue

            # Create snapshot
            snapshots = monitor.create_snapshot(odds_list)

            # Detect movements
            movements = monitor.detect_line_movements(snapshots)
            if movements:
                print(f"\n🚨 {len(movements)} line movement(s) detected!")
                for m in movements:
                    print(f"   {m.game}: {m.bet_type} {m.old_line} → {m.new_line}")

            # Find best odds
            best_odds = monitor.find_best_odds(snapshots)

            # Detect arbitrage
            arb = monitor.detect_arbitrage(snapshots)
            if arb:
                print(f"\n⚡ {len(arb)} arbitrage opportunity(ies) found!")

            # Generate report
            report = monitor.generate_report(snapshots, movements, best_odds, arb)
            print("\n" + report)

            # Save snapshot
            monitor.save_snapshot(snapshots)

            # Wait for next cycle
            print(f"\n⏳ Next update in {interval} seconds...")
            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")


async def single_check(week: int):
    """Single odds check (no loop)."""
    monitor = LiveOddsMonitor()

    # Fetch current odds
    odds_list = await monitor.fetch_current_odds()

    if not odds_list:
        print("❌ No odds available")
        return

    # Create snapshot
    snapshots = monitor.create_snapshot(odds_list)

    # Find best odds
    best_odds = monitor.find_best_odds(snapshots)

    # Detect arbitrage
    arb = monitor.detect_arbitrage(snapshots)

    # Generate report
    report = monitor.generate_report(snapshots, [], best_odds, arb)
    print("\n" + report)

    # Save snapshot
    monitor.save_snapshot(snapshots)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor live NFL odds across multiple sportsbooks"
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="NFL week number"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Continuous monitoring mode (default: single check)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Update interval in seconds (default: 300 = 5 min)"
    )

    args = parser.parse_args()

    if args.monitor:
        asyncio.run(monitor_loop(args.week, args.interval))
    else:
        asyncio.run(single_check(args.week))


if __name__ == "__main__":
    main()
