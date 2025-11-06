#!/usr/bin/env python3
"""
Correct MLB Player Props Fetching Script
Uses the proper two-step process required by The Odds API
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

import aiohttp

logger = logging.getLogger(__name__)


async def fetch_games_list(
    api_key: str, sport: str = "baseball_mlb", regions: str = "us"
) -> list[dict[str, Any]]:
    """
    Step 1: Fetch basic game list with standard markets (h2h, spreads, totals)
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h,spreads,totals",  # Only basic markets work here
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    try:
        async with aiohttp.ClientSession() as session:
            logger.info(f"🔄 Step 1: Fetching games list for {sport}")
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    games_data = await response.json()
                    logger.info(f"✅ Found {len(games_data)} games")
                    return games_data
                else:
                    logger.error(
                        f"❌ Games list API failed with status {response.status}"
                    )
                    error_text = await response.text()
                    logger.error(f"   Error: {error_text}")
                    return []
    except Exception as e:
        logger.error(f"❌ Exception fetching games list: {str(e)}")
        return []


async def fetch_event_odds_with_props(
    api_key: str, event_id: str, regions: str = "us"
) -> dict[str, Any]:
    """
    Step 2: Fetch specific event odds including player props
    """
    url = f"https://api.the-odds-api.com/v4/events/{event_id}/odds"

    # MLB Player prop markets that should work
    prop_markets = [
        "batter_home_runs",
        "batter_hits",
        "batter_total_bases",
        "batter_rbis",
        "pitcher_strikeouts",
        "pitcher_hits_allowed",
        "pitcher_earned_runs",
    ]

    # Combine basic and prop markets
    all_markets = "h2h,spreads,totals," + ",".join(prop_markets)

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": all_markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    try:
        async with aiohttp.ClientSession() as session:
            logger.info(f"🎯 Step 2: Fetching props for event {event_id}")
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    event_data = await response.json()

                    # Count prop markets found
                    prop_count = 0
                    if "bookmakers" in event_data:
                        for bookmaker in event_data["bookmakers"]:
                            for market in bookmaker.get("markets", []):
                                market_key = market.get("key", "")
                                if any(
                                    prop in market_key
                                    for prop in ["batter_", "pitcher_"]
                                ):
                                    prop_count += len(market.get("outcomes", []))
                                    logger.info(
                                        f"   🎯 Found prop market: {market_key} with {len(market.get('outcomes', []))} outcomes"
                                    )

                    logger.info(f"   📊 Total prop outcomes: {prop_count}")
                    return event_data

                elif response.status == 422:
                    # Some markets might not be available for this event
                    error_text = await response.text()
                    logger.warning(
                        f"⚠️ Some markets unavailable for {event_id}: {error_text}"
                    )

                    # Try with fewer markets
                    basic_props = "batter_home_runs,pitcher_strikeouts"
                    reduced_markets = f"h2h,spreads,totals,{basic_props}"

                    params["markets"] = reduced_markets
                    async with session.get(url, params=params) as retry_response:
                        if retry_response.status == 200:
                            logger.info(f"✅ Retry successful with basic props")
                            return await retry_response.json()
                        else:
                            logger.error(
                                f"❌ Retry failed: {await retry_response.text()}"
                            )
                            return {}
                else:
                    logger.error(f"❌ Event API failed with status {response.status}")
                    error_text = await response.text()
                    logger.error(f"   Error: {error_text}")
                    return {}

    except Exception as e:
        logger.error(f"❌ Exception fetching event {event_id}: {str(e)}")
        return {}


def parse_combined_odds_data(
    games_list: list[dict[str, Any]], events_data: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Parse combined data from both endpoints
    """
    structured_data = {
        "h2h_bets": [],
        "spread_bets": [],
        "total_bets": [],
        "prop_bets": [],
        "games_metadata": [],
    }

    logger.info(
        f"🔄 Parsing {len(games_list)} games with {len(events_data)} event details"
    )

    # Create a mapping of event IDs to event data
    events_map = {event.get("id", ""): event for event in events_data if event}

    for game in games_list:
        game_id = game.get("id", "unknown")
        game_info = {
            "game_id": game_id,
            "sport_title": game.get("sport_title", "MLB"),
            "commence_time": game.get("commence_time", ""),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
        }

        structured_data["games_metadata"].append(game_info)

        # Get event-specific data if available
        event_data = events_map.get(game_id, game)

        for bookmaker in event_data.get("bookmakers", []):
            bookmaker_title = bookmaker.get("title", "Unknown")

            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")

                if market_key == "h2h":
                    for outcome in market.get("outcomes", []):
                        structured_data["h2h_bets"].append(
                            {
                                "game_id": game_id,
                                "game_info": game_info,
                                "bookmaker": bookmaker_title,
                                "team": outcome.get("name", ""),
                                "odds": outcome.get("price", 0.0),
                            }
                        )

                elif market_key in ["spreads", "point_spreads"]:
                    for outcome in market.get("outcomes", []):
                        structured_data["spread_bets"].append(
                            {
                                "game_id": game_id,
                                "game_info": game_info,
                                "bookmaker": bookmaker_title,
                                "team": outcome.get("name", ""),
                                "point": outcome.get("point", 0.0),
                                "odds": outcome.get("price", 0.0),
                            }
                        )

                elif market_key in ["totals", "over_under"]:
                    for outcome in market.get("outcomes", []):
                        structured_data["total_bets"].append(
                            {
                                "game_id": game_id,
                                "game_info": game_info,
                                "bookmaker": bookmaker_title,
                                "over_under": outcome.get("name", ""),
                                "point": outcome.get("point", 0.0),
                                "odds": outcome.get("price", 0.0),
                            }
                        )

                elif any(keyword in market_key for keyword in ["batter_", "pitcher_"]):
                    # Handle MLB player props
                    logger.debug(f"🎯 Processing prop market: {market_key}")

                    for outcome in market.get("outcomes", []):
                        description = outcome.get("description", "")
                        name = outcome.get("name", "")

                        # Parse player name from description
                        player_name = "Unknown Player"
                        over_under = ""

                        if description:
                            parts = description.split()
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
                            "prop_type": market_key,
                            "player_name": player_name,
                            "over_under": over_under,
                            "point": outcome.get("point", 0.0),
                            "odds": outcome.get("price", 0.0),
                            "description": description,
                            "name": name,
                        }
                        structured_data["prop_bets"].append(prop_data)
                        logger.debug(
                            f"   ✅ Parsed prop: {player_name} {over_under} {market_key}"
                        )

    # Summary logging
    logger.info(f"📊 Final Parsing Results:")
    logger.info(f"   H2H bets: {len(structured_data['h2h_bets'])}")
    logger.info(f"   Spread bets: {len(structured_data['spread_bets'])}")
    logger.info(f"   Total bets: {len(structured_data['total_bets'])}")
    logger.info(f"   🎯 Prop bets: {len(structured_data['prop_bets'])}")
    logger.info(f"   Games: {len(structured_data['games_metadata'])}")

    return structured_data


async def test_complete_player_props_pipeline():
    """
    Test the complete two-step player props pipeline
    """
    from api_config import get_api_keys

    api_keys = get_api_keys()
    odds_api_key = api_keys.get("odds_api")

    if not odds_api_key:
        logger.error("❌ No odds API key found")
        return

    logger.info("🧪 Testing Complete Player Props Pipeline")
    logger.info("=" * 50)

    # Step 1: Get games list
    games_list = await fetch_games_list(odds_api_key)

    if not games_list:
        logger.error("❌ No games found")
        return

    logger.info(f"✅ Step 1 Complete: {len(games_list)} games found")

    # Step 2: Get props for first 3 games (to test)
    events_data = []
    test_games = games_list[:3]  # Limit to first 3 for testing

    for game in test_games:
        event_id = game.get("id")
        if event_id:
            logger.info(
                f"🎯 Testing props for: {game.get('away_team', 'TBA')} @ {game.get('home_team', 'TBA')}"
            )
            event_data = await fetch_event_odds_with_props(odds_api_key, event_id)
            if event_data:
                events_data.append(event_data)

            # Small delay to be respectful to API
            await asyncio.sleep(1)

    logger.info(f"✅ Step 2 Complete: Props fetched for {len(events_data)} games")

    # Step 3: Parse combined data
    structured_data = parse_combined_odds_data(test_games, events_data)

    # Display results
    logger.info("🎯 Player Props Test Results:")
    logger.info(f"   Total prop bets found: {len(structured_data['prop_bets'])}")

    if structured_data["prop_bets"]:
        logger.info("   Sample prop bets:")
        for i, prop in enumerate(structured_data["prop_bets"][:10]):
            logger.info(
                f"     {i+1}. {prop['player_name']}: {prop['prop_type']} {prop['over_under']} @ {prop['odds']}"
            )

    # Save results for inspection
    with open("testing_logs/player_props_test_results.json", "w") as f:
        json.dump(structured_data, f, indent=2, default=str)

    logger.info("✅ Results saved to testing_logs/player_props_test_results.json")

    return structured_data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Ensure testing_logs directory exists
    import os

    os.makedirs("testing_logs", exist_ok=True)

    asyncio.run(test_complete_player_props_pipeline())
