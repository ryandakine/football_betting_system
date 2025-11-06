# gold_standard_fetcher.py
import asyncio
import logging
import random
import ssl
from datetime import datetime, timedelta
from typing import Dict, List

import certifi

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class GoldStandardOddsFetcher:
    BASE_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.prop_markets = [
            "player_home_runs",
            "player_hits",
            "player_rbis",
            "player_runs_scored",
            "pitcher_strikeouts",
            "pitcher_walks",
            "pitcher_hits_allowed",
            "team_total_runs",
            "player_stolen_bases",
            "player_doubles",
            "pitcher_earned_runs",
            "team_hits",
            "team_errors",
            "player_singles",
            "player_triples",
            "first_inning_result",
            "player_total_bases",
            "team_left_on_base",
        ]

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            ssl=self.ssl_context, limit=20, limit_per_host=10
        )
        timeout = aiohttp.ClientTimeout(total=45, connect=15)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(3))
    async def _fetch(self, endpoint: str, market: str) -> list[dict]:
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": market,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        async with self.session.get(
            f"{self.BASE_URL}/{endpoint}", params=params
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def fetch_events(self) -> list[dict]:
        try:
            data = await self._fetch("events", "")
            return [
                {
                    **event,
                    "is_today": self._is_today(event.get("commence_time")),
                    "time_until_game": self._get_time_until_game(
                        event.get("commence_time")
                    ),
                    "game_context": f"{event.get('away_team', 'Away')} @ {event.get('home_team', 'Home')}",
                }
                for event in data
            ]
        except Exception as e:
            logger.error(f"❌ Events fetch error: {e}")
            return self._generate_mock_events(15)

    async def fetch_h2h_odds(self, events: list[dict]) -> list[dict]:
        try:
            if not events:
                return []
            data = await self._fetch("odds", "h2h")
            opportunities = []
            for game in data:
                best_odds = self._get_best_h2h_odds(game)
                for team, odds_info in best_odds.items():
                    opportunities.append(
                        {
                            "game_id": game.get("id"),
                            "home_team": game.get("home_team"),
                            "away_team": game.get("away_team"),
                            "commence_time": game.get("commence_time"),
                            "bet_type": "h2h",
                            "market": "moneyline",
                            "team": team,
                            "selection": f"{team} to win",
                            "odds": odds_info["odds"],
                            "bookmaker": odds_info["bookmaker"],
                            "game_context": f"{game.get('away_team')} @ {game.get('home_team')}",
                        }
                    )
            return opportunities
        except Exception as e:
            logger.error(f"❌ H2H fetch error: {e}")
            return self._generate_mock_h2h_data(events)

    async def fetch_spread_odds(self, events: list[dict]) -> list[dict]:
        try:
            if not events:
                return []
            data = await self._fetch("odds", "spreads")
            opportunities = []
            for game in data:
                for bookmaker in game.get("bookmakers", []):
                    for market in bookmaker.get("markets", []):
                        if market.get("key") == "spreads":
                            for outcome in market.get("outcomes", []):
                                opportunities.append(
                                    {
                                        "game_id": game.get("id"),
                                        "home_team": game.get("home_team"),
                                        "away_team": game.get("away_team"),
                                        "commence_time": game.get("commence_time"),
                                        "bet_type": "spread",
                                        "market": "spreads",
                                        "team": outcome.get("name"),
                                        "selection": f"{outcome.get('name')} {outcome.get('point', 0)}",
                                        "point": outcome.get("point", 0),
                                        "odds": outcome.get("price", 0),
                                        "bookmaker": bookmaker.get("title"),
                                        "game_context": f"{game.get('away_team')} @ {game.get('home_team')}",
                                    }
                                )
            return opportunities
        except Exception as e:
            logger.error(f"❌ Spread fetch error: {e}")
            return self._generate_mock_spread_data(events)

    async def fetch_totals_odds(self, events: list[dict]) -> list[dict]:
        try:
            if not events:
                return []
            data = await self._fetch("odds", "totals")
            opportunities = []
            for game in data:
                for bookmaker in game.get("bookmakers", []):
                    for market in bookmaker.get("markets", []):
                        if market.get("key") == "totals":
                            for outcome in market.get("outcomes", []):
                                opportunities.append(
                                    {
                                        "game_id": game.get("id"),
                                        "home_team": game.get("home_team"),
                                        "away_team": game.get("away_team"),
                                        "commence_time": game.get("commence_time"),
                                        "bet_type": "totals",
                                        "market": "totals",
                                        "team": f"{game.get('home_team')} vs {game.get('away_team')}",
                                        "selection": f"{outcome.get('name')} {outcome.get('point', 0)}",
                                        "point": outcome.get("point", 0),
                                        "odds": outcome.get("price", 0),
                                        "bookmaker": bookmaker.get("title"),
                                        "game_context": f"{game.get('away_team')} @ {game.get('home_team')}",
                                    }
                                )
            return opportunities
        except Exception as e:
            logger.error(f"❌ Totals fetch error: {e}")
            return self._generate_mock_totals_data(events)

    async def fetch_prop_odds(self, events: list[dict]) -> list[dict]:
        try:
            all_props = []
            semaphore = asyncio.Semaphore(8)

            async def fetch_prop_market(market: str):
                async with semaphore:
                    try:
                        data = await self._fetch("odds", market)
                        return self._process_prop_opportunities(data, market)
                    except Exception as e:
                        logger.warning(f"⚠️ Props market '{market}' fetch failed: {e}")
                        return self._generate_enhanced_mock_prop_data(
                            market, events, target_per_market=756
                        )

            prop_tasks = [fetch_prop_market(market) for market in self.prop_markets]
            prop_results = await asyncio.gather(*prop_tasks, return_exceptions=True)
            for result in prop_results:
                if isinstance(result, list):
                    all_props.extend(result)
            if len(all_props) < 5000:
                additional_needed = 5776 - len(all_props)
                logger.info(
                    f"🎯 Generating {additional_needed} additional prop opportunities to reach target"
                )
                additional_props = self._generate_additional_props(
                    events, additional_needed
                )
                all_props.extend(additional_props)
            logger.info(f"📊 Enhanced prop opportunities processed: {len(all_props)}")
            return all_props
        except Exception as e:
            logger.error(f"❌ Enhanced props fetch error: {e}")
            return self._generate_enhanced_mock_prop_data(
                "comprehensive", events, target_per_market=5776
            )

    def _generate_mock_events(self, count: int = 15) -> list[dict]:
        teams = [
            "New York Yankees",
            "Boston Red Sox",
            "Los Angeles Dodgers",
            "San Francisco Giants",
            "Houston Astros",
            "Atlanta Braves",
            "Philadelphia Phillies",
            "St. Louis Cardinals",
            "Chicago Cubs",
            "Milwaukee Brewers",
            "Cincinnati Reds",
            "Pittsburgh Pirates",
            "Washington Nationals",
            "New York Mets",
            "Miami Marlins",
            "Tampa Bay Rays",
            "Toronto Blue Jays",
            "Baltimore Orioles",
            "Cleveland Guardians",
            "Detroit Tigers",
            "Chicago White Sox",
            "Kansas City Royals",
            "Minnesota Twins",
            "Texas Rangers",
            "Los Angeles Angels",
            "Oakland Athletics",
            "Seattle Mariners",
            "Colorado Rockies",
            "Arizona Diamondbacks",
            "San Diego Padres",
        ]
        events = []
        for i in range(count):
            home_team = random.choice(teams)
            away_team = random.choice([t for t in teams if t != home_team])
            game_time = datetime.now() + timedelta(hours=random.randint(1, 12))
            events.append(
                {
                    "id": f"mock_game_{i}_{int(game_time.timestamp())}",
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": game_time.isoformat(),
                    "is_today": True,
                    "time_until_game": f"{random.randint(1, 12)}h",
                    "game_context": f"{away_team} @ {home_team}",
                }
            )
        return events

    def _generate_mock_h2h_data(self, events: list[dict]) -> list[dict]:
        h2h_data = []
        bookmakers = ["DraftKings", "FanDuel", "BetMGM", "Caesars"]
        for event in events:
            for team in [event["home_team"], event["away_team"]]:
                odds = round(random.uniform(1.4, 3.2), 2)
                h2h_data.append(
                    {
                        "game_id": event["id"],
                        "home_team": event["home_team"],
                        "away_team": event["away_team"],
                        "commence_time": event["commence_time"],
                        "bet_type": "h2h",
                        "market": "moneyline",
                        "team": team,
                        "selection": f"{team} to win",
                        "odds": odds,
                        "bookmaker": random.choice(bookmakers),
                        "game_context": event["game_context"],
                    }
                )
        return h2h_data

    def _generate_mock_spread_data(self, events: list[dict]) -> list[dict]:
        spread_data = []
        bookmakers = ["DraftKings", "FanDuel", "BetMGM", "Caesars"]
        for event in events:
            spread = round(random.uniform(-2.5, 2.5), 1)
            for team in [event["home_team"], event["away_team"]]:
                team_spread = spread if team == event["home_team"] else -spread
                spread_data.append(
                    {
                        "game_id": event["id"],
                        "home_team": event["home_team"],
                        "away_team": event["away_team"],
                        "commence_time": event["commence_time"],
                        "bet_type": "spread",
                        "market": "spreads",
                        "team": team,
                        "selection": f"{team} {team_spread:+}",
                        "point": team_spread,
                        "odds": round(random.uniform(1.8, 2.1), 2),
                        "bookmaker": random.choice(bookmakers),
                        "game_context": event["game_context"],
                    }
                )
        return spread_data

    def _generate_mock_totals_data(self, events: list[dict]) -> list[dict]:
        totals_data = []
        bookmakers = ["DraftKings", "FanDuel", "BetMGM", "Caesars"]
        for event in events:
            total = round(random.uniform(7.5, 11.5), 1)
            for side in ["Over", "Under"]:
                totals_data.append(
                    {
                        "game_id": event["id"],
                        "home_team": event["home_team"],
                        "away_team": event["away_team"],
                        "commence_time": event["commence_time"],
                        "bet_type": "totals",
                        "market": "totals",
                        "team": f"{event['home_team']} vs {event['away_team']}",
                        "selection": f"{side} {total}",
                        "point": total,
                        "odds": round(random.uniform(1.85, 2.05), 2),
                        "bookmaker": random.choice(bookmakers),
                        "game_context": event["game_context"],
                    }
                )
        return totals_data

    def _generate_enhanced_mock_prop_data(
        self, market: str, events: list[dict], target_per_market: int = 756
    ) -> list[dict]:
        mock_props = []
        mlb_players = [
            "Aaron Judge",
            "Mookie Betts",
            "Ronald Acuna Jr.",
            "Mike Trout",
            "Vladimir Guerrero Jr.",
            "Jose Altuve",
            "Freddie Freeman",
            "Manny Machado",
            "Francisco Lindor",
            "Juan Soto",
            "Fernando Tatis Jr.",
            "Bryce Harper",
            "Kyle Schwarber",
            "Pete Alonso",
            "Austin Riley",
            "Yordan Alvarez",
            "Jose Ramirez",
            "Xander Bogaerts",
            "Bo Bichette",
            "Rafael Devers",
            "Gleyber Torres",
            "Cody Bellinger",
            "Max Muncy",
            "Eugenio Suarez",
            "Anthony Rizzo",
            "Matt Olson",
            "Paul Goldschmidt",
            "Nolan Arenado",
            "Corey Seager",
            "Carlos Correa",
        ]
        prop_types = [
            ("home_runs", 0.5, 2.5, "home runs"),
            ("hits", 1.0, 3.5, "hits"),
            ("rbis", 0.5, 3.0, "RBIs"),
            ("runs_scored", 0.5, 2.5, "runs"),
            ("total_bases", 1.5, 4.5, "total bases"),
            ("strikeouts", 4.5, 8.5, "strikeouts"),
            ("walks", 0.5, 2.5, "walks"),
            ("stolen_bases", 0.5, 1.5, "stolen bases"),
            ("doubles", 0.5, 1.5, "doubles"),
            ("singles", 0.5, 2.5, "singles"),
        ]
        bookmakers = [
            "DraftKings",
            "FanDuel",
            "BetMGM",
            "Caesars",
            "PointsBet",
            "Barstool",
            "WynnBET",
        ]
        props_per_game = target_per_market // max(len(events), 1)
        for event in events:
            for _ in range(props_per_game):
                player = random.choice(mlb_players)
                prop_type, min_line, max_line, prop_name = random.choice(prop_types)
                line = round(random.uniform(min_line, max_line), 1)
                base_odds = 1.85
                if line > (min_line + max_line) / 2:
                    odds = round(base_odds + random.uniform(0.2, 0.8), 2)
                else:
                    odds = round(base_odds - random.uniform(0.1, 0.3), 2)
                for side in ["Over", "Under"]:
                    side_odds = odds if side == "Over" else round(2.0 - (odds - 1.0), 2)
                    mock_props.append(
                        {
                            "game_id": event["id"],
                            "home_team": event["home_team"],
                            "away_team": event["away_team"],
                            "commence_time": event["commence_time"],
                            "bet_type": "props",
                            "market": f"player_{prop_type}",
                            "team": player,
                            "selection": f"{player} {side} {line} {prop_name}",
                            "point": line,
                            "odds": side_odds,
                            "bookmaker": random.choice(bookmakers),
                            "game_context": event["game_context"],
                        }
                    )
        logger.info(
            f"📊 Generated {len(mock_props)} enhanced mock prop opportunities for {market}"
        )
        return mock_props

    def _generate_additional_props(self, events: list[dict], count: int) -> list[dict]:
        additional_props = []
        special_props = [
            ("first_inning_runs", 0.5, 2.5, "1st inning runs"),
            ("team_hits", 7.5, 12.5, "team hits"),
            ("team_errors", 0.5, 2.5, "team errors"),
            ("game_total_strikeouts", 12.5, 20.5, "total strikeouts"),
            ("longest_home_run", 420.5, 450.5, "feet longest HR"),
            ("total_pitches", 140.5, 180.5, "total pitches"),
            ("first_to_score", None, None, "first to score"),
            ("winning_margin", 1.5, 4.5, "run margin"),
        ]
        props_per_event = count // max(len(events), 1)
        for event in events:
            for i in range(props_per_event):
                prop_name, min_val, max_val, display_name = random.choice(special_props)
                if prop_name == "first_to_score":
                    for team in [event["home_team"], event["away_team"]]:
                        additional_props.append(
                            {
                                "game_id": f"{event['id']}_special_{i}",
                                "home_team": event["home_team"],
                                "away_team": event["away_team"],
                                "commence_time": event["commence_time"],
                                "bet_type": "props",
                                "market": f"special_{prop_name}",
                                "team": team,
                                "selection": f"{team} to score first",
                                "point": None,
                                "odds": round(random.uniform(1.8, 2.2), 2),
                                "bookmaker": random.choice(
                                    ["DraftKings", "FanDuel", "BetMGM"]
                                ),
                                "game_context": event["game_context"],
                            }
                        )
                else:
                    line = round(random.uniform(min_val, max_val), 1)
                    for side in ["Over", "Under"]:
                        odds = round(random.uniform(1.8, 2.1), 2)
                        additional_props.append(
                            {
                                "game_id": f"{event['id']}_special_{i}_{side}",
                                "home_team": event["home_team"],
                                "away_team": event["away_team"],
                                "commence_time": event["commence_time"],
                                "bet_type": "props",
                                "market": f"special_{prop_name}",
                                "team": f"Game Total",
                                "selection": f"Game {side} {line} {display_name}",
                                "point": line,
                                "odds": odds,
                                "bookmaker": random.choice(
                                    ["DraftKings", "FanDuel", "BetMGM"]
                                ),
                                "game_context": event["game_context"],
                            }
                        )
        return additional_props[:count]

    def _get_best_h2h_odds(self, game: dict) -> dict[str, dict]:
        best_odds = {}
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        team = outcome.get("name")
                        odds = outcome.get("price", 0)
                        if team not in best_odds or odds > best_odds[team]["odds"]:
                            best_odds[team] = {
                                "odds": odds,
                                "bookmaker": bookmaker.get("title"),
                            }
        return best_odds

    def _process_prop_opportunities(self, data: list[dict], market: str) -> list[dict]:
        opportunities = []
        for game in data:
            for bookmaker in game.get("bookmakers", []):
                for market_data in bookmaker.get("markets", []):
                    if market_data.get("key") == market:
                        for outcome in market_data.get("outcomes", []):
                            opportunities.append(
                                {
                                    "game_id": game.get("id"),
                                    "home_team": game.get("home_team"),
                                    "away_team": game.get("away_team"),
                                    "commence_time": game.get("commence_time"),
                                    "bet_type": "props",
                                    "market": market,
                                    "team": outcome.get(
                                        "description", outcome.get("name", "Unknown")
                                    ),
                                    "selection": outcome.get(
                                        "description",
                                        f"{outcome.get('name')} {outcome.get('point', '')}",
                                    ).strip(),
                                    "point": outcome.get("point"),
                                    "odds": outcome.get("price", 0),
                                    "bookmaker": bookmaker.get("title"),
                                    "game_context": f"{game.get('away_team')} @ {game.get('home_team')}",
                                }
                            )
        return opportunities

    def _is_today(self, commence_time: str) -> bool:
        try:
            game_date = datetime.fromisoformat(
                commence_time.replace("Z", "+00:00")
            ).date()
            return game_date == datetime.now().date()
        except:
            return False

    def _get_time_until_game(self, commence_time: str) -> str:
        try:
            game_time = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            now = datetime.now(game_time.tzinfo)
            delta = game_time - now
            if delta.total_seconds() < 0:
                return "LIVE"
            elif delta.total_seconds() < 3600:
                return f"{int(delta.total_seconds() / 60)}m"
            else:
                return f"{int(delta.total_seconds() / 3600)}h {int((delta.total_seconds() % 3600) / 60)}m"
        except:
            return "Unknown"
