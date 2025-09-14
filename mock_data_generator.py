#!/usr/bin/env python3
"""
Mock Data Generator for Football Betting System Testing
Creates realistic sample data when API keys aren't available.
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any


class FootballMockDataGenerator:
    """Generates realistic mock data for football betting system testing."""

    def __init__(self, sport_type: str = "nfl"):
        self.sport_type = sport_type
        self.nfl_teams = [
            "BUF", "MIA", "NE", "NYJ", "BAL", "CIN", "CLE", "PIT",
            "CHI", "DET", "GB", "MIN", "HOU", "IND", "JAX", "TEN",
            "ATL", "CAR", "NO", "TB", "DAL", "NYG", "PHI", "WAS",
            "ARI", "LAR", "SF", "SEA", "DEN", "KC", "LV", "LAC"
        ]

        self.college_teams = [
            "Alabama", "Georgia", "Ohio State", "USC", "Texas", "Oregon",
            "Florida State", "LSU", "Clemson", "Penn State", "Michigan", "Oklahoma",
            "Notre Dame", "Florida", "Miami", "Auburn", "Texas A&M", "Wisconsin",
            "UCLA", "Stanford", "South Carolina", "North Carolina", "Duke", "Wake Forest"
        ]

    def generate_mock_games(self, num_games: int = 10) -> List[Dict[str, Any]]:
        """Generate mock football games with realistic data."""
        games = []
        teams = self.nfl_teams if self.sport_type == "nfl" else self.college_teams

        # Generate games starting from tomorrow
        base_date = datetime.now() + timedelta(days=1)

        for i in range(num_games):
            # Select random teams
            home_team, away_team = random.sample(teams, 2)

            # Generate realistic odds (American format)
            home_moneyline = random.randint(-300, 300)
            if home_moneyline > 0:
                away_moneyline = -int(home_moneyline * random.uniform(0.8, 1.2))
            else:
                away_moneyline = -int(-home_moneyline * random.uniform(1.2, 1.5))

            # Generate spread
            spread = random.uniform(0, 14)
            home_spread = round(spread, 1)
            away_spread = -home_spread

            # Generate totals
            total = random.uniform(37, 55)
            total_line = round(total, 1)

            # Generate game time (NFL: Thursday/Sunday/Monday, College: Thursday/Saturday)
            if self.sport_type == "nfl":
                days_ahead = random.choice([1, 4, 6])  # Thu, Sun, Mon
            else:
                days_ahead = random.choice([1, 5])  # Thu, Sat

            game_date = base_date + timedelta(days=days_ahead)
            game_time = f"{random.randint(13, 20):02d}:00"  # 1 PM to 8 PM

            game = {
                "game_id": f"{away_team}_{home_team}_{game_date.strftime('%Y%m%d')}",
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": game_date.strftime("%Y-%m-%d") + "T" + game_time + ":00Z",
                "sport_key": f"americanfootball_{self.sport_type}",
                "odds": {
                    "moneyline": {
                        "home": home_moneyline,
                        "away": away_moneyline
                    },
                    "spreads": {
                        "home": home_spread,
                        "away": away_spread,
                        "home_odds": random.randint(-110, -105),
                        "away_odds": random.randint(-110, -105)
                    },
                    "totals": {
                        "line": total_line,
                        "over_odds": random.randint(-110, -105),
                        "under_odds": random.randint(-110, -105)
                    }
                },
                "metadata": {
                    "venue": f"{home_team} Stadium",
                    "weather": random.choice(["Clear", "Cloudy", "Rain", "Snow"]),
                    "temperature": random.randint(40, 80),
                    "is_rivalry": random.random() < 0.3,
                    "home_field_advantage": random.uniform(1.5, 4.0)
                }
            }

            games.append(game)

        return games

    def generate_mock_player_props(self, game: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mock player prop bets for a game."""
        props = []

        # Quarterback passing props
        qb_props = [
            {
                "player": f"{game['home_team']} QB",
                "market": "passing_yards",
                "line": random.randint(220, 320),
                "odds": random.randint(-115, 105),
                "game_id": game["game_id"]
            },
            {
                "player": f"{game['away_team']} QB",
                "market": "passing_yards",
                "line": random.randint(220, 320),
                "odds": random.randint(-115, 105),
                "game_id": game["game_id"]
            }
        ]

        # Running back rushing props
        rb_props = [
            {
                "player": f"{game['home_team']} RB",
                "market": "rushing_yards",
                "line": random.randint(60, 120),
                "odds": random.randint(-115, 105),
                "game_id": game["game_id"]
            },
            {
                "player": f"{game['away_team']} RB",
                "market": "rushing_yards",
                "line": random.randint(60, 120),
                "odds": random.randint(-115, 105),
                "game_id": game["game_id"]
            }
        ]

        # Touchdown scorer props
        td_props = [
            {
                "player": f"{game['home_team']} WR",
                "market": "touchdowns",
                "line": 0.5,
                "odds": random.randint(150, 400),
                "game_id": game["game_id"]
            },
            {
                "player": f"{game['away_team']} WR",
                "market": "touchdowns",
                "line": 0.5,
                "odds": random.randint(150, 400),
                "game_id": game["game_id"]
            }
        ]

        props.extend(qb_props)
        props.extend(rb_props)
        props.extend(td_props)

        return props

    def generate_complete_mock_data(self, num_games: int = 10) -> Dict[str, Any]:
        """Generate complete mock dataset for testing."""
        games = self.generate_mock_games(num_games)

        # Generate player props for a few games
        player_props = []
        for game in random.sample(games, min(3, len(games))):
            player_props.extend(self.generate_mock_player_props(game))

        mock_data = {
            "games": games,
            "player_props": player_props,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "sport_type": self.sport_type,
                "total_games": len(games),
                "total_props": len(player_props),
                "data_source": "mock_generator"
            }
        }

        return mock_data

    async def generate_mock_odds_data(self) -> 'StructuredOdds':
        """Generate StructuredOdds object with mock data for testing."""
        from football_odds_fetcher import StructuredOdds, GameInfo, H2HBet, SpreadBet, TotalBet, PlayerPropBet

        # Generate mock games
        mock_games_data = self.generate_mock_games(8)
        games = []

        for game_data in mock_games_data:
            game = GameInfo(
                game_id=game_data["game_id"],
                home_team=game_data["home_team"],
                away_team=game_data["away_team"],
                commence_time=game_data["commence_time"],
                sport_key=self.sport_type,
                sport_title="NFL" if self.sport_type == "nfl" else "College Football",
                bookmakers=[{"key": "draftkings", "title": "DraftKings"}]
            )
            games.append(game)

        # Generate mock bets
        h2h_bets = []
        spread_bets = []
        total_bets = []
        prop_bets = []

        for game in games:
            # H2H bets
            h2h_bets.append(H2HBet(
                game_id=game.game_id,
                home_team=game.home_team,
                away_team=game.away_team,
                home_odds=random.randint(-150, 200),
                away_odds=random.randint(-150, 200),
                bookmaker="draftkings",
                last_update=datetime.now().isoformat()
            ))

            # Spread bets
            spread_val = random.uniform(-10, 10)
            spread_bets.append(SpreadBet(
                game_id=game.game_id,
                home_team=game.home_team,
                away_team=game.away_team,
                home_spread=spread_val,
                away_spread=-spread_val,
                home_odds=random.randint(-110, -105),
                away_odds=random.randint(-110, -105),
                bookmaker="draftkings",
                last_update=datetime.now().isoformat()
            ))

            # Total bets
            total_bets.append(TotalBet(
                game_id=game.game_id,
                home_team=game.home_team,
                away_team=game.away_team,
                total_points=random.uniform(40, 55),
                over_odds=random.randint(-110, -105),
                under_odds=random.randint(-110, -105),
                bookmaker="draftkings",
                last_update=datetime.now().isoformat()
            ))

        # Generate some player props
        for game in random.sample(games, min(3, len(games))):
            for team in [game.home_team, game.away_team]:
                prop_bets.append(PlayerPropBet(
                    game_id=game.game_id,
                    player_name=f"{team} QB",
                    prop_type="passing_yards",
                    line_value=random.randint(220, 320),
                    over_odds=random.randint(-115, 105),
                    under_odds=random.randint(-115, 105),
                    bookmaker="draftkings",
                    last_update=datetime.now().isoformat()
                ))

        return StructuredOdds(
            games=games,
            h2h_bets=h2h_bets,
            spread_bets=spread_bets,
            total_bets=total_bets,
            prop_bets=prop_bets,
            api_requests_used=0,
            fetch_timestamp=datetime.now().isoformat()
        )


def create_sample_data_for_testing():
    """Create sample data files for testing."""
    print("🏈 Generating mock football data for testing...")

    # Generate NFL data
    nfl_generator = FootballMockDataGenerator("nfl")
    nfl_data = nfl_generator.generate_complete_mock_data(8)

    # Generate College data
    college_generator = FootballMockDataGenerator("ncaaf")
    college_data = college_generator.generate_complete_mock_data(12)

    print(f"✅ Generated {nfl_data['metadata']['total_games']} NFL games")
    print(f"✅ Generated {college_data['metadata']['total_games']} College games")
    print(f"✅ Generated {nfl_data['metadata']['total_props']} NFL player props")
    print(f"✅ Generated {college_data['metadata']['total_props']} College player props")

    return {
        "nfl": nfl_data,
        "college": college_data
    }


if __name__ == "__main__":
    create_sample_data_for_testing()
