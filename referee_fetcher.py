#!/usr/bin/env python3
"""
Referee Fetcher - Scrapes referee assignments for NFL games

WHY THIS EXISTS:
Certain referees correlate with game outcomes (penalties, pace, scoring).
This script automatically fetches referee assignments so the betting system
can adjust strategy based on referee intelligence.

DESIGN PHILOSOPHY: Investment → System
- Agent forgets referee data every session
- This script fetches current referee assignments on demand
- Integrates with auto_execute_bets.py for full automation

USAGE:
    python referee_fetcher.py --game "PHI @ GB"
    python referee_fetcher.py --week 10 --all
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install requests beautifulsoup4")
    sys.exit(1)


class RefereeFetcher:
    """Fetches NFL referee assignments from footballzebras.com"""

    # Known referees with betting intelligence
    REFEREE_INTEL = {
        "Shawn Hochuli": {
            "flags": "high",
            "pace": "slow",
            "betting_edge": "UNDER",
            "confidence_boost": 3,
            "notes": "GB averages 7.5 penalties with Hochuli"
        },
        "Adrian Hill": {
            "flags": "high",
            "pace": "slow",
            "betting_edge": "UNDER",
            "confidence_boost": 3,
            "notes": "PHI averages 7.9 penalties with Hill"
        },
        "Bill Vinovich": {
            "flags": "low",
            "pace": "fast",
            "betting_edge": "OVER",
            "confidence_boost": 2,
            "notes": "Lowest flag rate in NFL"
        },
        "Brad Allen": {
            "flags": "high",
            "pace": "slow",
            "betting_edge": "UNDER",
            "confidence_boost": 2,
            "notes": "Calls most holding penalties"
        }
    }

    def __init__(self):
        self.base_url = "https://www.footballzebras.com/category/assignments/"
        self.cache_dir = Path(__file__).parent / "data"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "referee_cache.json"

    def fetch_current_week_assignments(self) -> Dict:
        """
        Fetch referee assignments for current NFL week.

        Returns:
            Dict mapping "AWAY @ HOME" to referee name
        """
        try:
            # Fetch the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(self.base_url, headers=headers, timeout=10)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the latest assignment post
            assignments = {}

            # Look for article with referee assignments
            # footballzebras.com posts weekly referee assignments
            article = soup.find('article')
            if not article:
                print("⚠️  Could not find referee assignment article")
                return self._load_cache()

            # Extract text content
            text = article.get_text()

            # Parse format: "AWAY at HOME: Referee Name"
            # Common patterns:
            # "Philadelphia Eagles at Green Bay Packers: Shawn Hochuli"
            # "PHI @ GB: Shawn Hochuli"
            pattern = r'([A-Za-z\s]+)\s+(?:at|@)\s+([A-Za-z\s]+):\s*([A-Za-z\s]+)'
            matches = re.findall(pattern, text)

            for away, home, referee in matches:
                away = away.strip()
                home = home.strip()
                referee = referee.strip()

                # Normalize team names to abbreviations
                away_abbr = self._normalize_team_name(away)
                home_abbr = self._normalize_team_name(home)

                game_key = f"{away_abbr} @ {home_abbr}"
                assignments[game_key] = referee

            # Cache the results
            self._save_cache(assignments)

            return assignments

        except Exception as e:
            print(f"⚠️  Error fetching referee assignments: {e}")
            print("   Falling back to cache...")
            return self._load_cache()

    def get_referee_for_game(self, game: str) -> Optional[str]:
        """
        Get referee for specific game.

        Args:
            game: Game string like "PHI @ GB"

        Returns:
            Referee name or None
        """
        assignments = self.fetch_current_week_assignments()

        # Normalize game string
        game = game.upper().strip()

        # Try exact match first
        if game in assignments:
            return assignments[game]

        # Try fuzzy match
        for game_key, referee in assignments.items():
            if self._games_match(game, game_key):
                return referee

        return None

    def get_referee_intelligence(self, referee: str) -> Optional[Dict]:
        """
        Get betting intelligence for referee.

        Args:
            referee: Referee name

        Returns:
            Dict with betting edge info or None
        """
        return self.REFEREE_INTEL.get(referee)

    def _normalize_team_name(self, team: str) -> str:
        """Convert full team name to abbreviation."""
        team = team.upper().strip()

        # Map full names to abbreviations
        team_map = {
            "EAGLES": "PHI", "PHILADELPHIA": "PHI",
            "PACKERS": "GB", "GREEN BAY": "GB",
            "CHIEFS": "KC", "KANSAS CITY": "KC",
            "BILLS": "BUF", "BUFFALO": "BUF",
            "49ERS": "SF", "SAN FRANCISCO": "SF",
            "RAVENS": "BAL", "BALTIMORE": "BAL",
            "LIONS": "DET", "DETROIT": "DET",
            "COWBOYS": "DAL", "DALLAS": "DAL",
            # Add more as needed
        }

        for full_name, abbr in team_map.items():
            if full_name in team:
                return abbr

        # If already abbreviation, return as-is
        if len(team) <= 3:
            return team

        return team[:3]

    def _games_match(self, game1: str, game2: str) -> bool:
        """Check if two game strings refer to same game."""
        # Extract team abbreviations
        def extract_teams(game):
            parts = re.split(r'[@\s]+', game.upper().strip())
            return tuple(p for p in parts if p and len(p) <= 3)

        teams1 = extract_teams(game1)
        teams2 = extract_teams(game2)

        return teams1 == teams2

    def _save_cache(self, assignments: Dict):
        """Save assignments to cache file."""
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "assignments": assignments
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

    def _load_cache(self) -> Dict:
        """Load assignments from cache file."""
        if not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                return cache_data.get("assignments", {})
        except:
            return {}


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NFL referee assignments"
    )
    parser.add_argument(
        "--game",
        help="Game to check (e.g. 'PHI @ GB')"
    )
    parser.add_argument(
        "--week",
        type=int,
        help="NFL week number"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all assignments for current week"
    )
    parser.add_argument(
        "--intel",
        action="store_true",
        help="Show betting intelligence for referee"
    )

    args = parser.parse_args()

    fetcher = RefereeFetcher()

    if args.game:
        # Fetch referee for specific game
        referee = fetcher.get_referee_for_game(args.game)

        if referee:
            print(f"🏈 {args.game}")
            print(f"👨‍⚖️  Referee: {referee}")

            if args.intel:
                intel = fetcher.get_referee_intelligence(referee)
                if intel:
                    print(f"\n📊 BETTING INTELLIGENCE:")
                    print(f"   Flags: {intel['flags']}")
                    print(f"   Pace: {intel['pace']}")
                    print(f"   Edge: {intel['betting_edge']}")
                    print(f"   Confidence Boost: +{intel['confidence_boost']}%")
                    print(f"   Notes: {intel['notes']}")
                else:
                    print(f"\n📊 No betting intelligence for {referee}")
        else:
            print(f"❌ Could not find referee for {args.game}")
            print(f"   Try: python referee_fetcher.py --all")

    elif args.all:
        # Show all assignments
        assignments = fetcher.fetch_current_week_assignments()

        if assignments:
            print("=" * 60)
            print("🏈 NFL REFEREE ASSIGNMENTS")
            print("=" * 60)

            for game, referee in sorted(assignments.items()):
                intel = fetcher.get_referee_intelligence(referee)
                marker = "⭐⭐⭐" if intel else ""
                print(f"{game:20s} → {referee:20s} {marker}")

            print("=" * 60)
            print(f"Total games: {len(assignments)}")
            print("⭐⭐⭐ = Has betting intelligence")
        else:
            print("❌ Could not fetch referee assignments")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
