#!/usr/bin/env python3
"""
Contrarian Intelligence Fetcher - Tracks sharp money vs public betting

WHY THIS EXISTS:
DeepSeek-R1 has home favorite bias because favorites often win.
But betting favorites when public is too heavy = negative EV.
This tool fetches contrarian signals to feed into DeepSeek's analysis.

DESIGN PHILOSOPHY: Investment → System
- Contrarian data is fetched automatically
- Fed directly to DeepSeek-R1 as context
- DeepSeek adjusts picks based on sharp money signals

THE PROBLEM:
- Public bets home favorites heavily (70-90%)
- Books shade lines toward public money
- Home favorites are overpriced
- Sharp money fades public = contrarian edge

THE SOLUTION:
- Fetch line movement (opening → current)
- Get public betting percentages
- Identify reverse line movement (line moves AGAINST public money = sharp action)
- Feed to DeepSeek as "contrarian intelligence"

USAGE:
    python contrarian_intelligence.py --game "PHI @ GB"
    python contrarian_intelligence.py --week 11 --all
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    print("❌ Missing requests. Install with: pip install requests")
    sys.exit(1)


class ContrarianIntelligence:
    """Fetches contrarian betting signals (sharp money indicators)"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.cache_file = self.data_dir / "contrarian_cache.json"

    def fetch_line_movement(self, game: str) -> Dict:
        """
        Fetch line movement for a game to detect sharp money.

        Sharp money indicators:
        - Line moves OPPOSITE of public betting (reverse line movement)
        - Example: Public 75% on home, but line moves toward away = sharp on away

        Args:
            game: Game string like "PHI @ GB"

        Returns:
            Dict with line movement data
        """
        print(f"🔍 Fetching line movement for {game}...")

        if not self.api_key:
            print("⚠️  No ODDS_API_KEY - using cache/estimates")
            return self._estimate_line_movement(game)

        try:
            # Fetch from The Odds API
            url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
            params = {
                'apiKey': self.api_key,
                'regions': 'us',
                'markets': 'spreads,totals',
                'oddsFormat': 'american',
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            games = response.json()

            # Find our game
            for g in games:
                if self._match_game(g, game):
                    movement = self._analyze_line_movement(g)
                    self._cache_movement(game, movement)
                    return movement

            print(f"❌ Game not found: {game}")
            return self._estimate_line_movement(game)

        except Exception as e:
            print(f"⚠️  Error fetching odds: {e}")
            return self._estimate_line_movement(game)

    def _analyze_line_movement(self, game_data: Dict) -> Dict:
        """Analyze line movement from API data."""
        # Note: The Odds API doesn't give historical lines directly
        # This would require integration with a service like Action Network
        # For now, we'll use current consensus to detect sharp action

        spread_books = []
        total_books = []

        for book in game_data.get('bookmakers', []):
            for market in book.get('markets', []):
                if market['key'] == 'spreads':
                    for outcome in market.get('outcomes', []):
                        spread_books.append({
                            'book': book.get('title'),
                            'team': outcome.get('name'),
                            'spread': outcome.get('point'),
                            'odds': outcome.get('price')
                        })
                elif market['key'] == 'totals':
                    for outcome in market.get('outcomes', []):
                        total_books.append({
                            'book': book.get('title'),
                            'side': outcome.get('name'),
                            'total': outcome.get('point'),
                            'odds': outcome.get('price')
                        })

        # Detect sharp action by variance in lines
        # If one book is significantly different = sharp money moved that book
        spread_variance = self._calculate_variance([b['spread'] for b in spread_books if b.get('spread')])
        total_variance = self._calculate_variance([b['total'] for b in total_books if b.get('total')])

        return {
            'game': f"{game_data['away_team']} @ {game_data['home_team']}",
            'spread_data': spread_books,
            'total_data': total_books,
            'spread_variance': spread_variance,
            'total_variance': total_variance,
            'sharp_indicators': self._detect_sharp_action(spread_books, total_books),
            'timestamp': datetime.now().isoformat()
        }

    def _detect_sharp_action(self, spread_books: List[Dict], total_books: List[Dict]) -> Dict:
        """Detect sharp money based on book discrepancies."""
        indicators = {
            'sharp_side': None,
            'confidence': 0,
            'reasoning': []
        }

        if not spread_books:
            return indicators

        # Find outlier books (sharp money indicators)
        spreads = [b['spread'] for b in spread_books if b.get('spread')]
        if not spreads:
            return indicators

        avg_spread = sum(spreads) / len(spreads)
        max_spread = max(spreads)
        min_spread = min(spreads)

        # If spread variance > 0.5 points, sharp action detected
        if (max_spread - min_spread) > 0.5:
            # Books with outlier lines got sharp action
            if abs(max_spread - avg_spread) > abs(min_spread - avg_spread):
                indicators['sharp_side'] = 'AWAY'
                indicators['confidence'] = 2
                indicators['reasoning'].append(
                    f"Spread variance {max_spread - min_spread:.1f} points - sharp money on away"
                )
            else:
                indicators['sharp_side'] = 'HOME'
                indicators['confidence'] = 2
                indicators['reasoning'].append(
                    f"Spread variance {max_spread - min_spread:.1f} points - sharp money on home"
                )

        return indicators

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance in values."""
        if not values or len(values) < 2:
            return 0.0

        avg = sum(values) / len(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        return variance ** 0.5  # Standard deviation

    def _match_game(self, api_game: Dict, game_str: str) -> bool:
        """Check if API game matches our game string."""
        away = api_game.get('away_team', '').upper()
        home = api_game.get('home_team', '').upper()
        game_upper = game_str.upper()

        return any(team in game_upper for team in [away, home])

    def _estimate_line_movement(self, game: str) -> Dict:
        """
        Estimate line movement when API unavailable.

        Uses common patterns:
        - Home favorites with high public % = line often inflated
        - Divisional games = public overvalues home team
        - Prime time games = public overvalues favorites
        """
        # Check cache first
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
                if game in cache:
                    cached = cache[game]
                    print(f"   ℹ️  Using cached data from {cached.get('timestamp')}")
                    return cached

        # Default estimate
        return {
            'game': game,
            'status': 'estimated',
            'sharp_indicators': {
                'sharp_side': None,
                'confidence': 0,
                'reasoning': ['No API key - unable to detect sharp action']
            },
            'public_betting_estimate': {
                'likely_public_side': 'HOME',
                'estimated_percentage': 65,
                'reasoning': 'Home teams typically get 55-70% of public money'
            },
            'contrarian_recommendation': 'Consider fading home favorite if public% > 70%',
            'timestamp': datetime.now().isoformat()
        }

    def get_public_betting_percentages(self, game: str) -> Dict:
        """
        Get public betting percentages (requires premium data source).

        Sources:
        - Action Network (paid API)
        - Sports Insights (paid)
        - VegasInsider (scraping)

        For now, we'll estimate based on common patterns.
        """
        print(f"📊 Estimating public betting for {game}...")

        # Common public betting patterns:
        # - Home teams: 55-70% of public money
        # - Prime time games: 60-75% on favorites
        # - Divisional games: 60-70% on home team
        # - Popular teams: 65-80% public money

        # Parse game
        parts = game.split('@')
        if len(parts) != 2:
            return {'error': 'Invalid game format'}

        away = parts[0].strip()
        home = parts[1].strip()

        # Popular teams (get more public money)
        popular_teams = ['KC', 'BUF', 'SF', 'DAL', 'PHI', 'GB', 'NE', 'PIT']

        home_public_pct = 60  # Base assumption

        if home in popular_teams:
            home_public_pct += 10

        return {
            'game': game,
            'status': 'estimated',
            'home_team': home,
            'away_team': away,
            'public_percentage': {
                'home': home_public_pct,
                'away': 100 - home_public_pct
            },
            'contrarian_threshold': 70,  # Fade public when > 70%
            'contrarian_signal': 'FADE HOME' if home_public_pct > 70 else 'NO SIGNAL',
            'reasoning': f"Estimated {home_public_pct}% public on {home} (home team advantage + popularity)",
            'timestamp': datetime.now().isoformat()
        }

    def get_contrarian_intelligence(self, game: str) -> Dict:
        """
        Get complete contrarian intelligence for a game.

        Returns:
            Dict with line movement, public%, and contrarian recommendations
        """
        print(f"=" * 70)
        print(f"🎯 CONTRARIAN INTELLIGENCE: {game}")
        print(f"=" * 70)
        print()

        # Fetch line movement
        line_movement = self.fetch_line_movement(game)
        print()

        # Get public betting
        public_betting = self.get_public_betting_percentages(game)
        print()

        # Generate contrarian signals
        signals = self._generate_contrarian_signals(line_movement, public_betting)

        intelligence = {
            'game': game,
            'line_movement': line_movement,
            'public_betting': public_betting,
            'contrarian_signals': signals,
            'timestamp': datetime.now().isoformat()
        }

        return intelligence

    def _generate_contrarian_signals(self, line_movement: Dict, public_betting: Dict) -> Dict:
        """Generate contrarian betting signals."""
        signals = {
            'strength': 0,  # 0-5 scale
            'recommendation': None,
            'reasoning': []
        }

        # Check for reverse line movement
        sharp_indicators = line_movement.get('sharp_indicators', {})
        if sharp_indicators.get('sharp_side'):
            signals['strength'] += sharp_indicators.get('confidence', 0)
            signals['reasoning'].extend(sharp_indicators.get('reasoning', []))

        # Check public betting percentages
        public_pct = public_betting.get('public_percentage', {})
        home_pct = public_pct.get('home', 50)

        if home_pct > 70:
            signals['strength'] += 2
            signals['recommendation'] = 'FADE HOME - Take AWAY team'
            signals['reasoning'].append(
                f"Public heavily on home ({home_pct}%) - contrarian edge on away"
            )
        elif home_pct < 30:
            signals['strength'] += 2
            signals['recommendation'] = 'FADE AWAY - Take HOME team'
            signals['reasoning'].append(
                f"Public heavily on away ({100-home_pct}%) - contrarian edge on home"
            )

        # Sharp money + public betting alignment = strong signal
        if signals['strength'] >= 3:
            signals['reasoning'].append("STRONG CONTRARIAN SIGNAL - Sharp + public misalignment")

        return signals

    def _cache_movement(self, game: str, data: Dict):
        """Cache line movement data."""
        cache = {}
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)

        cache[game] = data

        with open(self.cache_file, 'w') as f:
            json.dump(cache, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch contrarian intelligence for NFL games"
    )
    parser.add_argument(
        "--game",
        help="Game to analyze (e.g. 'PHI @ GB')"
    )
    parser.add_argument(
        "--week",
        type=int,
        help="NFL week number"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all games for current week"
    )
    parser.add_argument(
        "--api-key",
        help="The Odds API key (or set ODDS_API_KEY env var)"
    )

    args = parser.parse_args()

    fetcher = ContrarianIntelligence(api_key=args.api_key)

    if args.game:
        intelligence = fetcher.get_contrarian_intelligence(args.game)

        # Print summary
        print("=" * 70)
        print("📋 CONTRARIAN SUMMARY")
        print("=" * 70)
        print()

        signals = intelligence['contrarian_signals']
        print(f"Strength: {'⭐' * signals['strength']} ({signals['strength']}/5)")
        print(f"Recommendation: {signals.get('recommendation', 'NO CLEAR SIGNAL')}")
        print()
        print("Reasoning:")
        for reason in signals['reasoning']:
            print(f"  • {reason}")
        print()

        # Save to file
        output_file = fetcher.data_dir / f"contrarian_{args.game.replace(' ', '_').replace('@', 'at')}.json"
        with open(output_file, 'w') as f:
            json.dump(intelligence, f, indent=2)
        print(f"💾 Saved to {output_file}")

    else:
        parser.print_help()
        print("\n💡 TIP: Use this to detect sharp money vs public betting!")
        print("   Example: python contrarian_intelligence.py --game 'PHI @ GB'")


if __name__ == "__main__":
    main()
