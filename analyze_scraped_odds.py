#!/usr/bin/env python3
"""
Analyze Scraped NFL Odds
=========================
Takes scraped odds data and runs it through the AI prediction models
to generate betting recommendations.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import re

from dotenv import load_dotenv
load_dotenv()

from trimodel_game_analyzer import TriModelAnalyzer
from tri_model_api_config import get_trimodel_api_keys, validate_api_configuration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScrapedOddsAnalyzer:
    """Analyzes scraped odds using AI models"""

    def __init__(self):
        self.api_keys = get_trimodel_api_keys()
        valid, errors = validate_api_configuration()

        if not valid:
            logger.warning(f"⚠️  Some API keys missing: {errors}")
            logger.info("Will use available models only")

    def load_scraped_data(self, filepath: str = 'data/scraped_nfl_odds.json') -> Dict:
        """Load scraped odds from JSON file"""
        logger.info(f"📂 Loading scraped data from: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return data

    def extract_games_from_scraped_data(self, scraped_data: Dict) -> List[Dict]:
        """Extract game information from scraped data"""
        logger.info("🔍 Extracting game information...")

        games = []

        # Parse through all sources
        for source, entries in scraped_data.items():
            if source == 'timestamp':
                continue

            if not isinstance(entries, list):
                continue

            for entry in entries:
                # Try to extract game data
                game = self._parse_entry(entry, source)
                if game:
                    games.append(game)

        # Deduplicate games
        unique_games = self._deduplicate_games(games)

        logger.info(f"✅ Found {len(unique_games)} unique games")
        return unique_games

    def _parse_entry(self, entry: Dict, source: str) -> Dict:
        """Parse a single scraped entry into game data"""
        game = {
            'source': source,
            'home_team': None,
            'away_team': None,
            'home_ml': None,
            'away_ml': None,
            'spread': None,
            'spread_odds': None,
            'total': None,
            'over_odds': None,
            'under_odds': None
        }

        # Convert entry to string for parsing
        entry_str = str(entry)

        # NFL team names (abbreviated)
        teams = {
            'chiefs': 'Kansas City Chiefs',
            'bills': 'Buffalo Bills',
            'bengals': 'Cincinnati Bengals',
            'ravens': 'Baltimore Ravens',
            'browns': 'Cleveland Browns',
            'steelers': 'Pittsburgh Steelers',
            'texans': 'Houston Texans',
            'colts': 'Indianapolis Colts',
            'jaguars': 'Jacksonville Jaguars',
            'titans': 'Tennessee Titans',
            'broncos': 'Denver Broncos',
            'raiders': 'Las Vegas Raiders',
            'chargers': 'Los Angeles Chargers',
            'patriots': 'New England Patriots',
            'dolphins': 'Miami Dolphins',
            'jets': 'New York Jets',
            'cowboys': 'Dallas Cowboys',
            'eagles': 'Philadelphia Eagles',
            'giants': 'New York Giants',
            'commanders': 'Washington Commanders',
            'bears': 'Chicago Bears',
            'packers': 'Green Bay Packers',
            'lions': 'Detroit Lions',
            'vikings': 'Minnesota Vikings',
            'falcons': 'Atlanta Falcons',
            'panthers': 'Carolina Panthers',
            'saints': 'New Orleans Saints',
            'buccaneers': 'Tampa Bay Buccaneers',
            'bucs': 'Tampa Bay Buccaneers',
            'cardinals': 'Arizona Cardinals',
            'rams': 'Los Angeles Rams',
            '49ers': 'San Francisco 49ers',
            'niners': 'San Francisco 49ers',
            'seahawks': 'Seattle Seahawks'
        }

        # Find teams in the entry
        found_teams = []
        for abbrev, full_name in teams.items():
            if abbrev in entry_str.lower():
                found_teams.append(full_name)

        if len(found_teams) >= 2:
            game['away_team'] = found_teams[0]
            game['home_team'] = found_teams[1]

            # Extract odds (American format: +150, -110, etc.)
            odds = re.findall(r'([+-]\d{2,})', entry_str)
            if len(odds) >= 2:
                game['away_ml'] = int(odds[0])
                game['home_ml'] = int(odds[1])

            return game

        return None

    def _deduplicate_games(self, games: List[Dict]) -> List[Dict]:
        """Remove duplicate games and merge odds from multiple sources"""
        unique = {}

        for game in games:
            if not game['home_team'] or not game['away_team']:
                continue

            key = f"{game['away_team']}_{game['home_team']}"

            if key not in unique:
                unique[key] = game
            else:
                # Merge odds from multiple sources
                for field in ['home_ml', 'away_ml', 'spread', 'total']:
                    if game[field] and not unique[key][field]:
                        unique[key][field] = game[field]

        return list(unique.values())

    async def analyze_games(self, games: List[Dict]) -> List[Dict]:
        """Run AI analysis on extracted games"""
        logger.info("🤖 Starting AI analysis...")

        analyzer = TriModelAnalyzer()
        results = []

        for game in games:
            logger.info(f"\n{'='*70}")
            logger.info(f"Analyzing: {game['away_team']} @ {game['home_team']}")
            logger.info(f"{'='*70}")

            # Format game info for AI models
            game_info = {
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_moneyline': game.get('home_ml'),
                'away_moneyline': game.get('away_ml'),
                'spread': game.get('spread'),
                'total': game.get('total'),
                'sport': 'NFL'
            }

            try:
                # Run through AI models
                analysis = await analyzer.analyze_game(game_info)

                result = {
                    'game': game,
                    'analysis': analysis,
                    'analyzed_at': datetime.now().isoformat()
                }
                results.append(result)

                # Display results
                self._display_analysis(game, analysis)

            except Exception as e:
                logger.error(f"❌ Analysis failed: {e}")
                continue

        return results

    def _display_analysis(self, game: Dict, analysis: Dict):
        """Display analysis results in a readable format"""
        print(f"\n📊 ANALYSIS RESULTS")
        print(f"{'='*70}")
        print(f"Game: {game['away_team']} @ {game['home_team']}")

        if game.get('home_ml') and game.get('away_ml'):
            print(f"Moneyline: {game['away_team']} {game['away_ml']:+d} | {game['home_team']} {game['home_ml']:+d}")

        print(f"\n🤖 AI Model Predictions:")

        # Display individual model predictions
        if 'models' in analysis:
            for model_name, prediction in analysis['models'].items():
                confidence = prediction.get('confidence_score', 0)
                pick = prediction.get('favored_team', 'N/A')
                reasoning = prediction.get('reasoning', 'No reasoning provided')

                print(f"\n  {model_name.upper()}:")
                print(f"    Pick: {pick}")
                print(f"    Confidence: {confidence:.1%}")
                print(f"    Reasoning: {reasoning}")

        # Display consensus
        if 'consensus' in analysis:
            consensus = analysis['consensus']
            print(f"\n🎯 CONSENSUS:")
            print(f"    Pick: {consensus.get('pick', 'N/A')}")
            print(f"    Confidence: {consensus.get('confidence', 0):.1%}")
            print(f"    Agreement: {consensus.get('agreement', 0):.1%}")

        print(f"{'='*70}")

    def save_analysis(self, results: List[Dict], filename: str = 'nfl_analysis_results.json'):
        """Save analysis results to file"""
        filepath = f"data/{filename}"

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 Analysis saved to: {filepath}")
        return filepath


async def main():
    """Main execution"""
    print("""
    ═══════════════════════════════════════════════════════════════
    🏈 NFL ODDS ANALYSIS SYSTEM
    ═══════════════════════════════════════════════════════════════
    Analyzing scraped odds with AI prediction models
    ═══════════════════════════════════════════════════════════════
    """)

    analyzer = ScrapedOddsAnalyzer()

    # Load scraped data
    try:
        scraped_data = analyzer.load_scraped_data()
    except FileNotFoundError:
        print("❌ No scraped data found!")
        print("Please run: python3 nfl_odds_scraper.py first")
        return

    # Extract games
    games = analyzer.extract_games_from_scraped_data(scraped_data)

    if not games:
        print("❌ No games found in scraped data!")
        print("The scraper may need adjustment for current website formats.")
        return

    print(f"\n✅ Found {len(games)} games to analyze\n")

    # Analyze games
    results = await analyzer.analyze_games(games)

    # Save results
    filepath = analyzer.save_analysis(results)

    print(f"""
    ═══════════════════════════════════════════════════════════════
    ✅ ANALYSIS COMPLETE!
    ═══════════════════════════════════════════════════════════════

    📊 Results: {len(results)} games analyzed
    💾 Saved to: {filepath}

    🎯 Next steps:
       • Review the analysis above
       • Check {filepath} for detailed results
       • Place bets based on high-confidence picks

    ═══════════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    asyncio.run(main())
