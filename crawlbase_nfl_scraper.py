#!/usr/bin/env python3
"""
NFL Data Scraper using Crawlbase
Fetches live odds, injuries, weather, and news for betting analysis
"""
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional

try:
    from crawlbase import CrawlingAPI
except ImportError:
    print("❌ Crawlbase not installed. Run: pip install crawlbase")
    sys.exit(1)


class NFLCrawlbaseCollector:
    """
    Automated NFL data collection using Crawlbase API

    Features:
    - Sunday game schedules
    - Live odds from multiple sportsbooks
    - Injury reports
    - Weather conditions
    - News and sentiment
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize Crawlbase API

        Args:
            token: Crawlbase API token (or reads from CRAWLBASE_TOKEN env var)
        """
        self.token = token or os.getenv('CRAWLBASE_TOKEN')

        if not self.token:
            raise ValueError(
                "No Crawlbase token found. Set CRAWLBASE_TOKEN env var "
                "or pass token to constructor"
            )

        self.api = CrawlingAPI({'token': self.token})
        self.data = {
            'timestamp': datetime.now().isoformat(),
            'collection_type': 'nfl_weekend',
            'games': [],
            'injuries': None,
            'weather': None,
            'odds': {},
            'news': []
        }

    def get_sunday_games(self) -> Dict:
        """
        Fetch Sunday's NFL schedule from ESPN

        Returns:
            HTML body of ESPN scoreboard
        """
        print("📅 Fetching Sunday NFL schedule...")

        response = self.api.get('https://www.espn.com/nfl/scoreboard')

        if response['statusCode'] == 200:
            print(f"   ✅ Got {len(response['body'])} bytes of schedule data")
            self.data['games'] = response['body']
            return response['body']
        else:
            print(f"   ❌ Error {response['statusCode']}")
            return None

    def get_injury_report(self) -> Dict:
        """
        Fetch latest NFL injury reports from ESPN

        Returns:
            HTML body of injury report
        """
        print("🏥 Fetching injury reports...")

        response = self.api.get('https://www.espn.com/nfl/injuries')

        if response['statusCode'] == 200:
            print(f"   ✅ Got {len(response['body'])} bytes of injury data")
            self.data['injuries'] = response['body']
            return response['body']
        else:
            print(f"   ❌ Error {response['statusCode']}")
            return None

    def get_weather(self) -> Dict:
        """
        Fetch weather for NFL stadiums

        Returns:
            HTML body of weather data
        """
        print("🌦️  Fetching stadium weather...")

        # Try multiple weather sources
        urls = [
            'https://www.nfl.com/weather',
            'https://www.espn.com/nfl/weather'
        ]

        for url in urls:
            response = self.api.get(url)
            if response['statusCode'] == 200:
                print(f"   ✅ Got weather from {url}")
                self.data['weather'] = response['body']
                return response['body']

        print("   ❌ No weather data available")
        return None

    def get_odds_draftkings(self) -> Dict:
        """
        Fetch odds from DraftKings

        Returns:
            HTML body of DraftKings odds
        """
        print("💰 Fetching DraftKings odds...")

        response = self.api.get('https://sportsbook.draftkings.com/leagues/football/nfl')

        if response['statusCode'] == 200:
            print(f"   ✅ Got {len(response['body'])} bytes of DraftKings data")
            self.data['odds']['draftkings'] = response['body']
            return response['body']
        else:
            print(f"   ❌ Error {response['statusCode']}")
            return None

    def get_odds_fanduel(self) -> Dict:
        """
        Fetch odds from FanDuel

        Returns:
            HTML body of FanDuel odds
        """
        print("💰 Fetching FanDuel odds...")

        response = self.api.get('https://sportsbook.fanduel.com/navigation/nfl')

        if response['statusCode'] == 200:
            print(f"   ✅ Got {len(response['body'])} bytes of FanDuel data")
            self.data['odds']['fanduel'] = response['body']
            return response['body']
        else:
            print(f"   ❌ Error {response['statusCode']}")
            return None

    def get_news_espn(self) -> Dict:
        """
        Fetch latest NFL news from ESPN

        Returns:
            HTML body of ESPN NFL news
        """
        print("📰 Fetching ESPN news...")

        response = self.api.get('https://www.espn.com/nfl/')

        if response['statusCode'] == 200:
            print(f"   ✅ Got {len(response['body'])} bytes of news data")
            self.data['news'].append({
                'source': 'espn',
                'data': response['body']
            })
            return response['body']
        else:
            print(f"   ❌ Error {response['statusCode']}")
            return None

    def get_reddit_sentiment(self) -> Dict:
        """
        Fetch Reddit r/NFL sentiment

        Returns:
            HTML body of Reddit r/NFL
        """
        print("💬 Fetching Reddit sentiment...")

        response = self.api.get('https://old.reddit.com/r/nfl/')

        if response['statusCode'] == 200:
            print(f"   ✅ Got {len(response['body'])} bytes of Reddit data")
            self.data['news'].append({
                'source': 'reddit',
                'data': response['body']
            })
            return response['body']
        else:
            print(f"   ❌ Error {response['statusCode']}")
            return None

    def run_full_collection(self, include_odds: bool = True,
                           include_news: bool = False) -> Dict:
        """
        Run complete data collection for NFL weekend

        Args:
            include_odds: Whether to fetch sportsbook odds (uses more requests)
            include_news: Whether to fetch news/sentiment (uses more requests)

        Returns:
            Complete data dictionary
        """
        print("\n" + "="*60)
        print("🏈 NFL DATA COLLECTION - Powered by Crawlbase")
        print("="*60 + "\n")

        # Core data (always collect)
        self.get_sunday_games()
        self.get_injury_report()
        self.get_weather()

        # Optional: Odds (uses more API requests)
        if include_odds:
            print("\n" + "-"*60)
            print("Collecting Sportsbook Odds...")
            print("-"*60)
            self.get_odds_draftkings()
            self.get_odds_fanduel()

        # Optional: News & Sentiment (uses more API requests)
        if include_news:
            print("\n" + "-"*60)
            print("Collecting News & Sentiment...")
            print("-"*60)
            self.get_news_espn()
            self.get_reddit_sentiment()

        # Save to file
        output_file = self._save_data()

        # Print summary
        self._print_summary(output_file)

        return self.data

    def _save_data(self) -> str:
        """Save collected data to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'nfl_crawlbase_data_{timestamp}.json'

        # Save raw data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        return output_file

    def _print_summary(self, output_file: str):
        """Print collection summary"""
        print("\n" + "="*60)
        print("✅ DATA COLLECTION COMPLETE")
        print("="*60)
        print(f"\n📁 Saved to: {output_file}")
        print(f"📊 Timestamp: {self.data['timestamp']}")
        print(f"\n📈 Data Collected:")
        print(f"   ✓ Games: {'Yes' if self.data['games'] else 'No'}")
        print(f"   ✓ Injuries: {'Yes' if self.data['injuries'] else 'No'}")
        print(f"   ✓ Weather: {'Yes' if self.data['weather'] else 'No'}")
        print(f"   ✓ Odds sources: {len(self.data['odds'])}")
        print(f"   ✓ News sources: {len(self.data['news'])}")
        print("\n" + "="*60 + "\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='NFL Data Scraper using Crawlbase'
    )
    parser.add_argument(
        '--token',
        help='Crawlbase API token (or set CRAWLBASE_TOKEN env var)'
    )
    parser.add_argument(
        '--no-odds',
        action='store_true',
        help='Skip sportsbook odds collection'
    )
    parser.add_argument(
        '--include-news',
        action='store_true',
        help='Include news and sentiment collection'
    )

    args = parser.parse_args()

    try:
        collector = NFLCrawlbaseCollector(token=args.token)
        collector.run_full_collection(
            include_odds=not args.no_odds,
            include_news=args.include_news
        )

        print("🎉 Ready to run your NFL predictions!")
        print("\nNext step:")
        print("   python3 unified_nfl_intelligence_system.py")

    except ValueError as e:
        print(f"\n❌ Error: {e}\n")
        print("Setup instructions:")
        print("1. Sign up at https://crawlbase.com/signup")
        print("2. Get your free API token")
        print("3. Set environment variable:")
        print("   export CRAWLBASE_TOKEN='your_token_here'")
        print("   OR add to .env file")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
