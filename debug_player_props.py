# Enhanced fetch_odds_from_api function with player props debugging

import asyncio
import json
import logging
from typing import Any, Dict, List

import aiohttp

logger = logging.getLogger(__name__)


async def fetch_odds_from_api(
    api_key: str,
    sport: str = "baseball_mlb",
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
) -> list[dict[str, Any]]:
    """
    Fetch odds data from The Odds API with enhanced player props support
    """
    # Add MLB player props to markets
    enhanced_markets = f"{markets},batter_home_runs,batter_hits,batter_total_bases,batter_rbis,pitcher_strikeouts,pitcher_hits_allowed"

    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": enhanced_markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    try:
        async with aiohttp.ClientSession() as session:
            logger.info(f"🔄 Fetching odds from API with markets: {enhanced_markets}")
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    response_data = await response.json()

                    # Debug logging for player props
                    logger.info(f"📊 API Response Summary:")
                    logger.info(f"   Total games returned: {len(response_data)}")

                    prop_count = 0
                    for game in response_data[:3]:  # Check first 3 games
                        game_id = game.get("id", "unknown")
                        bookmakers = game.get("bookmakers", [])
                        logger.info(f"   Game {game_id}: {len(bookmakers)} bookmakers")

                        for bookmaker in bookmakers:
                            markets = bookmaker.get("markets", [])
                            logger.info(
                                f"     {bookmaker.get('title', 'Unknown')}: {len(markets)} markets"
                            )

                            for market in markets:
                                market_key = market.get("key", "")
                                if (
                                    "player" in market_key
                                    or "batter" in market_key
                                    or "pitcher" in market_key
                                ):
                                    prop_count += len(market.get("outcomes", []))
                                    logger.info(
                                        f"       🎯 PROP MARKET: {market_key} with {len(market.get('outcomes', []))} outcomes"
                                    )

                    logger.info(f"   Total prop outcomes found in sample: {prop_count}")

                    return response_data
                else:
                    logger.error(f"❌ API request failed with status {response.status}")
                    error_text = await response.text()
                    logger.error(f"   Error details: {error_text}")
                    return []
    except Exception as e:
        logger.error(f"❌ Exception in fetch_odds_from_api: {str(e)}")
        return []


def parse_and_structure_odds(raw_odds_data: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Enhanced parsing function with proper player props handling
    """
    structured_data = {
        "h2h_bets": [],
        "spread_bets": [],
        "total_bets": [],
        "prop_bets": [],
        "games_metadata": [],
    }

    logger.info(f"🔄 Parsing {len(raw_odds_data)} games for all bet types...")

    for game in raw_odds_data:
        game_id = game.get("id", "unknown")
        game_info = {
            "game_id": game_id,
            "sport_title": game.get("sport_title", "MLB"),
            "commence_time": game.get("commence_time", ""),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
        }

        structured_data["games_metadata"].append(game_info)

        for bookmaker in game.get("bookmakers", []):
            bookmaker_title = bookmaker.get("title", "Unknown")

            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")

                if market_key == "h2h":
                    # Handle H2H bets (existing logic)
                    for outcome in market.get("outcomes", []):
                        h2h_data = {
                            "game_id": game_id,
                            "game_info": game_info,
                            "bookmaker": bookmaker_title,
                            "team": outcome.get("name", ""),
                            "odds": outcome.get("price", 0.0),
                        }
                        structured_data["h2h_bets"].append(h2h_data)

                elif market_key in ["spreads", "point_spreads"]:
                    # Handle spread bets (existing logic)
                    for outcome in market.get("outcomes", []):
                        spread_data = {
                            "game_id": game_id,
                            "game_info": game_info,
                            "bookmaker": bookmaker_title,
                            "team": outcome.get("name", ""),
                            "point": outcome.get("point", 0.0),
                            "odds": outcome.get("price", 0.0),
                        }
                        structured_data["spread_bets"].append(spread_data)

                elif market_key in ["totals", "over_under"]:
                    # Handle totals bets (existing logic)
                    for outcome in market.get("outcomes", []):
                        total_data = {
                            "game_id": game_id,
                            "game_info": game_info,
                            "bookmaker": bookmaker_title,
                            "over_under": outcome.get("name", ""),
                            "point": outcome.get("point", 0.0),
                            "odds": outcome.get("price", 0.0),
                        }
                        structured_data["total_bets"].append(total_data)

                elif any(keyword in market_key for keyword in ["batter_", "pitcher_"]):
                    # Handle player props - ENHANCED LOGIC
                    logger.debug(f"🎯 Processing prop market: {market_key}")

                    for outcome in market.get("outcomes", []):
                        # Extract player name and prop details
                        description = outcome.get("description", "")
                        name = outcome.get("name", "")

                        # Try to parse player name from description or name
                        player_name = "Unknown Player"
                        prop_type = market_key
                        over_under = ""

                        if description:
                            # Format: "Player Name Over/Under X.5"
                            parts = description.split()
                            if len(parts) >= 3:
                                if "Over" in parts or "Under" in parts:
                                    over_under_idx = next(
                                        (
                                            i
                                            for i, part in enumerate(parts)
                                            if part in ["Over", "Under"]
                                        ),
                                        -1,
                                    )
                                    if over_under_idx > 0:
                                        player_name = " ".join(parts[:over_under_idx])
                                        over_under = parts[over_under_idx]

                        if not player_name or player_name == "Unknown Player":
                            # Fallback to name field
                            if name and any(word in name for word in ["Over", "Under"]):
                                parts = name.split()
                                over_under = (
                                    parts[-1] if parts[-1] in ["Over", "Under"] else ""
                                )
                                player_name = (
                                    " ".join(parts[:-1]) if over_under else name
                                )

                        prop_data = {
                            "game_id": game_id,
                            "game_info": game_info,
                            "bookmaker": bookmaker_title,
                            "prop_type": prop_type,
                            "player_name": player_name,
                            "over_under": over_under,
                            "point": outcome.get("point", 0.0),
                            "odds": outcome.get("price", 0.0),
                            "description": description,
                            "name": name,
                        }
                        structured_data["prop_bets"].append(prop_data)
                        logger.debug(
                            f"   ✅ Parsed prop: {player_name} {over_under} {prop_type}"
                        )

    # Summary logging
    logger.info(f"📊 Parsing Results:")
    logger.info(f"   H2H bets: {len(structured_data['h2h_bets'])}")
    logger.info(f"   Spread bets: {len(structured_data['spread_bets'])}")
    logger.info(f"   Total bets: {len(structured_data['total_bets'])}")
    logger.info(f"   Prop bets: {len(structured_data['prop_bets'])}")
    logger.info(f"   Games: {len(structured_data['games_metadata'])}")

    return structured_data


# Test function to verify player props
async def test_player_props_api():
    """Test function to check if player props are available"""
    from api_config import get_api_keys

    api_keys = get_api_keys()
    odds_api_key = api_keys.get("odds_api")

    if not odds_api_key:
        logger.error("❌ No odds API key found")
        return

    logger.info("🧪 Testing player props API...")
    raw_data = await fetch_odds_from_api(odds_api_key)
    structured_data = parse_and_structure_odds(raw_data)

    logger.info(f"🎯 Player Props Test Results:")
    logger.info(f"   Found {len(structured_data['prop_bets'])} prop bets")

    if structured_data["prop_bets"]:
        logger.info("   Sample prop bets:")
        for prop in structured_data["prop_bets"][:5]:
            logger.info(
                f"     - {prop['player_name']}: {prop['prop_type']} {prop['over_under']}"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_player_props_api())
