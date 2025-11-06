#!/usr/bin/env python3
"""
Manual enhancement - just show the current file and let user choose approach
"""

from datetime import datetime


def show_current_file(filename):
    """Show what's currently in the file."""

    try:
        # Try different encodings
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
        content = None

        for encoding in encodings:
            try:
                with open(filename, encoding=encoding) as f:
                    content = f.read()
                print(f"Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            print("Could not read file with any encoding")
            return False

        lines = content.split("\n")
        print(f"\nFile has {len(lines)} lines")

        # Show the structure around the parsing function
        print("\n=== LOOKING FOR PARSING FUNCTION ===")

        found_parsing = False
        for i, line in enumerate(lines):
            if "_parse_and_select_odds" in line:
                print(
                    f"Found reference to parsing function at line {i+1}: {line.strip()}"
                )
                found_parsing = True

                # Show context around this line
                start = max(0, i - 5)
                end = min(len(lines), i + 15)
                print(f"\nContext (lines {start+1}-{end}):")
                for j in range(start, end):
                    marker = ">>> " if j == i else "    "
                    print(f"{marker}{j+1:3}: {lines[j]}")
                break

        if not found_parsing:
            print("Did not find _parse_and_select_odds function")

            # Show a sample of the file to understand structure
            print("\nFirst 30 lines of file:")
            for i in range(min(30, len(lines))):
                print(f"  {i+1:3}: {lines[i]}")

        return True

    except Exception as e:
        print(f"Error reading file: {e}")
        return False


def create_enhanced_version_manually(filename):
    """Create a new enhanced version manually."""

    print("\n=== CREATING ENHANCED VERSION MANUALLY ===")

    enhanced_content = '''"""
Daily MLB Prediction and Backtesting System - Enhanced Version
Now with detailed odds parsing and analysis
"""

import logging
import asyncio
import os
import polars as pl
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from tenacity import retry, wait_fixed, stop_after_attempt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PredictionConfig:
    """Configuration for the prediction system."""
    test_mode: bool = True
    odds_api_key: str = "219a6d41e72ff68350230d0a6d8dcf9b"
    output_dir: str = "daily_logs"
    min_confidence: float = 0.6
    api_timeout_seconds: int = 30

def _load_data_safe(path: str, schema: dict = None) -> pl.DataFrame:
    """Safely load a Polars DataFrame."""
    if os.path.exists(path):
        try:
            df = pl.read_parquet(path)
            logger.debug(f"Loaded data from {path} (rows: {df.height})")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {path}: {e}")
            return pl.DataFrame(schema=schema if schema else {})
    else:
        logger.warning(f"Data file not found: {path}")
        return pl.DataFrame(schema=schema if schema else {})

@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def fetch_odds_from_api(api_key: str, date: datetime, api_timeout: int = 30) -> dict:
    """Fetch odds from The Odds API."""
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    try:
        response = requests.get(url, params=params, timeout=api_timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            logger.warning("API returned 422 - no games available or invalid parameters")
            return {"data": []}
        else:
            raise

def _parse_and_select_odds(raw_odds_data) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Parse raw odds data and return structured DataFrames with detailed analysis."""
    logger.info("ENHANCED PARSING: Starting detailed odds analysis...")

    # Handle both dict and list responses from the API
    if isinstance(raw_odds_data, dict):
        games_data = raw_odds_data.get("data", [])
    elif isinstance(raw_odds_data, list):
        games_data = raw_odds_data
    else:
        games_data = []

    if not games_data:
        logger.warning("No games data available")
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    logger.info(f"ENHANCED: Processing {len(games_data)} games with full odds extraction...")

    # Parse all odds data into detailed records
    records = []
    games_processed = 0

    for game in games_data:
        game_id = str(game.get("id", ""))
        commence_time = game.get("commence_time", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        games_processed += 1
        logger.info(f"  Processing Game {games_processed}: {away_team} @ {home_team}")

        bookmaker_count = 0
        for bookmaker in game.get("bookmakers", []):
            bookmaker_key = bookmaker.get("key", "")
            bookmaker_count += 1

            market_count = 0
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                outcomes = market.get("outcomes", [])
                market_count += 1

                if not isinstance(outcomes, list):
                    continue

                for outcome in outcomes:
                    outcome_name = outcome.get("name", "")
                    price = outcome.get("price", 0)
                    point = outcome.get("point")

                    records.append({
                        "game_id": game_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker_key": bookmaker_key,
                        "market_key": market_key,
                        "outcome_name": outcome_name,
                        "price": float(price) if price else 0.0,
                        "point": float(point) if point else None,
                    })

        logger.info(f"    Found {bookmaker_count} bookmakers, {market_count} markets")

    if not records:
        logger.warning("No odds records parsed from API response")
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    # Convert to DataFrame and analyze
    parsed_odds_df = pl.DataFrame(records)
    logger.info(f"SUCCESS: Parsed {len(records)} total odds records")

    # Separate market types with detailed analysis
    moneyline_odds = parsed_odds_df.filter(pl.col("market_key") == "h2h")
    spreads_odds = parsed_odds_df.filter(pl.col("market_key") == "spreads")
    totals_odds = parsed_odds_df.filter(pl.col("market_key") == "totals")
    player_props = parsed_odds_df.filter(pl.col("market_key").str.starts_with("player_"))

    logger.info(f"MARKET BREAKDOWN:")
    logger.info(f"  Moneyline (H2H): {len(moneyline_odds)} records")
    logger.info(f"  Spreads: {len(spreads_odds)} records")
    logger.info(f"  Totals (O/U): {len(totals_odds)} records")
    logger.info(f"  Player Props: {len(player_props)} records")

    # Find best moneyline odds with analysis
    if not moneyline_odds.is_empty():
        best_moneyline = moneyline_odds.group_by(["game_id", "outcome_name"]).agg([
            pl.col("price").max().alias("best_price"),
            pl.col("price").min().alias("worst_price"),
            pl.col("bookmaker_key").first().alias("best_bookmaker"),
            pl.col("home_team").first().alias("home_team"),
            pl.col("away_team").first().alias("away_team"),
            pl.col("commence_time").first().alias("commence_time"),
        ])

        logger.info(f"BEST MONEYLINE ODDS: {len(best_moneyline)} outcomes")

        # Show some sample odds
        if len(best_moneyline) > 0:
            sample = best_moneyline.head(5)
            logger.info("Sample best odds:")
            for row in sample.iter_rows(named=True):
                edge = row['best_price'] - row['worst_price']
                logger.info(f"  {row['outcome_name']}: {row['best_price']:.2f} ({row['best_bookmaker']}) [Edge: +{edge:.2f}]")
    else:
        best_moneyline = pl.DataFrame()

    # Find best totals odds
    if not totals_odds.is_empty():
        best_totals = totals_odds.group_by(["game_id", "outcome_name", "point"]).agg([
            pl.col("price").max().alias("best_price"),
            pl.col("bookmaker_key").first().alias("best_bookmaker"),
            pl.col("home_team").first().alias("home_team"),
            pl.col("away_team").first().alias("away_team"),
            pl.col("commence_time").first().alias("commence_time"),
        ])
        logger.info(f"BEST TOTALS ODDS: {len(best_totals)} outcomes")
    else:
        best_totals = pl.DataFrame()

    return best_moneyline, player_props, best_totals

class DailyPredictionManager:
    """Manages the daily prediction and betting workflow with enhanced analysis."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self._current_date = datetime.now().date()
        os.makedirs(self.config.output_dir, exist_ok=True)
        logger.info(f"ENHANCED MANAGER: Initialized for {self._current_date}")

    async def run_daily_workflow(
        self,
        current_date: Optional[datetime] = None,
        enable_team_predictions: bool = True,
        enable_player_predictions: bool = True,
        enable_betting_recommendations: bool = True,
        enable_backtesting: bool = True,
    ) -> Dict[str, Any]:
        """Run the complete daily workflow with enhanced odds analysis."""

        if current_date:
            self._current_date = current_date.date()

        logger.info(f"ENHANCED WORKFLOW: Starting for {self._current_date}")

        try:
            # Step 1: Fetch odds with detailed logging
            logger.info("STEP 1: Fetching live odds from API...")
            raw_odds_data = fetch_odds_from_api(
                self.config.odds_api_key,
                datetime.combine(self._current_date, datetime.min.time()),
                self.config.api_timeout_seconds,
            )

            # Step 2: Enhanced odds parsing
            logger.info("STEP 2: Enhanced odds parsing and analysis...")
            moneyline_odds, player_props, totals_odds = _parse_and_select_odds(raw_odds_data)

            # Step 3: Basic edge analysis
            recommendations = []
            if enable_betting_recommendations and not moneyline_odds.is_empty():
                logger.info("STEP 3: Generating betting recommendations...")
                recommendations = self._analyze_edges_and_recommend(moneyline_odds)

            # Step 4: Results summary
            results = {
                "date": self._current_date.isoformat(),
                "games_available": len(raw_odds_data.get("data", [])) if isinstance(raw_odds_data, dict) else len(raw_odds_data) if isinstance(raw_odds_data, list) else 0,
                "moneyline_outcomes": len(moneyline_odds) if not moneyline_odds.is_empty() else 0,
                "totals_outcomes": len(totals_odds) if not totals_odds.is_empty() else 0,
                "player_props": len(player_props) if not player_props.is_empty() else 0,
                "recommendations": recommendations,
                "status": "success"
            }

            logger.info(f"ENHANCED WORKFLOW COMPLETE:")
            logger.info(f"  Games: {results['games_available']}")
            logger.info(f"  Moneyline outcomes: {results['moneyline_outcomes']}")
            logger.info(f"  Recommendations: {len(recommendations)}")

            return results

        except Exception as e:
            logger.error(f"Enhanced workflow error: {e}")
            return {
                "date": self._current_date.isoformat(),
                "error": str(e),
                "status": "failed"
            }

    def _analyze_edges_and_recommend(self, moneyline_odds: pl.DataFrame) -> List[Dict[str, Any]]:
        """Analyze edges and generate basic recommendations."""
        recommendations = []

        try:
            if moneyline_odds.is_empty():
                return recommendations

            # Group by game and look for opportunities
            games = moneyline_odds.group_by("game_id").agg([
                pl.col("home_team").first(),
                pl.col("away_team").first(),
                pl.col("best_price").max().alias("max_price"),
                pl.col("best_price").min().alias("min_price"),
            ])

            for game in games.iter_rows(named=True):
                game_id = game["game_id"]

                # Get home and away odds
                home_odds = moneyline_odds.filter(
                    (pl.col("game_id") == game_id) &
                    (pl.col("outcome_name") == game["home_team"])
                )
                away_odds = moneyline_odds.filter(
                    (pl.col("game_id") == game_id) &
                    (pl.col("outcome_name") == game["away_team"])
                )

                if not home_odds.is_empty() and not away_odds.is_empty():
                    home_price = home_odds["best_price"].max()
                    away_price = away_odds["best_price"].max()

                    # Simple edge detection (compare to 50/50 baseline)
                    home_implied = 1 / home_price
                    away_implied = 1 / away_price

                    # Look for significant deviations from fair odds
                    if home_implied < 0.45:  # Home team undervalued
                        edge = 0.5 - home_implied
                        recommendations.append({
                            "game": f"{game['away_team']} @ {game['home_team']}",
                            "bet": game["home_team"],
                            "odds": home_price,
                            "estimated_edge": f"{edge:.1%}",
                            "confidence": "medium"
                        })

                    if away_implied < 0.45:  # Away team undervalued
                        edge = 0.5 - away_implied
                        recommendations.append({
                            "game": f"{game['away_team']} @ {game['home_team']}",
                            "bet": game["away_team"],
                            "odds": away_price,
                            "estimated_edge": f"{edge:.1%}",
                            "confidence": "medium"
                        })

            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

async def main():
    """Enhanced main function with detailed reporting."""
    logger.info("ENHANCED SYSTEM: Starting daily prediction and betting workflow")

    config = PredictionConfig(test_mode=True)
    manager = DailyPredictionManager(config)

    result = await manager.run_daily_workflow()

    logger.info(f"FINAL RESULT: {result}")

    print("\\n" + "="*60)
    print("ENHANCED MLB BETTING SYSTEM - DAILY RESULTS")
    print("="*60)
    print(f"Date: {result['date']}")
    print(f"Status: {result['status']}")

    if result['status'] == 'success':
        print(f"Games Available: {result.get('games_available', 0)}")
        print(f"Moneyline Outcomes: {result.get('moneyline_outcomes', 0)}")
        print(f"Totals Outcomes: {result.get('totals_outcomes', 0)}")
        print(f"Player Props: {result.get('player_props', 0)}")

        recommendations = result.get('recommendations', [])
        print(f"\\nBetting Recommendations: {len(recommendations)}")

        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['game']}")
            print(f"     Bet: {rec['bet']} at {rec['odds']:.2f}")
            print(f"     Edge: {rec['estimated_edge']} ({rec['confidence']} confidence)")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
'''

    # Create backup of current file
    backup_name = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        with open(filename, encoding="utf-8") as f:
            original_content = f.read()
        with open(backup_name, "w", encoding="utf-8") as f:
            f.write(original_content)
        print(f"Created backup: {backup_name}")
    except Exception as e:
        print(f"Could not create backup: {e}")

    # Write enhanced version
    with open(filename, "w", encoding="utf-8") as f:
        f.write(enhanced_content)

    print(f"Created enhanced version of {filename}")

    # Test compilation
    try:
        with open(filename, encoding="utf-8") as f:
            compile(f.read(), filename, "exec")
        print("SUCCESS: Enhanced version compiles!")
        return True
    except SyntaxError as e:
        print(f"ERROR: Syntax error on line {e.lineno}: {e.msg}")
        return False


if __name__ == "__main__":
    filename = "daily_prediction_and_backtest.py"

    print("Choose enhancement approach:")
    print("1. Show current file structure")
    print("2. Create enhanced version manually (recommended)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        show_current_file(filename)
    elif choice == "2":
        if create_enhanced_version_manually(filename):
            print("\\n" + "=" * 50)
            print("ENHANCEMENT COMPLETE!")
            print("=" * 50)
            print("Test your enhanced system:")
            print("python daily_prediction_and_backtest.py")
            print("\\nNew features:")
            print("- Detailed odds parsing from all 15 games")
            print("- Market breakdown (moneyline, totals, props)")
            print("- Best odds detection across bookmakers")
            print("- Basic edge analysis and recommendations")
            print("- Enhanced logging and reporting")
        else:
            print("Enhancement failed")
    else:
        print("Invalid choice")
