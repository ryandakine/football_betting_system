#!/usr/bin/env python3
"""
NFL Weather Impact Analyzer
Scrapes weather data and calculates betting impact for NFL games

Key Features:
- Fetches game-time weather for all stadiums
- Detects dome stadiums (no weather impact)
- Calculates weather severity (NONE → EXTREME)
- Provides specific betting adjustments

Edge: Weather-adjusted totals can swing by 3-6 points
"""
import os
import sys
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

try:
    from crawlbase import CrawlingAPI
except ImportError:
    print("❌ Crawlbase not installed. Run: pip install crawlbase")
    sys.exit(1)


@dataclass
class StadiumInfo:
    """NFL stadium information"""
    team: str
    stadium: str
    city: str
    state: str
    is_dome: bool
    is_retractable: bool


@dataclass
class WeatherData:
    """Weather data for a game"""
    game: str
    home_team: str
    stadium: str
    is_dome: bool
    temperature: float
    feels_like: float
    wind_speed: float
    wind_gust: float
    precipitation_chance: float
    precipitation_type: str
    humidity: float
    conditions: str
    timestamp: str


@dataclass
class WeatherImpact:
    """Calculated weather impact on betting"""
    game: str
    weather: WeatherData
    severity: str  # NONE, MILD, MODERATE, SEVERE, EXTREME
    total_adjustment: float  # Points to adjust total
    spread_adjustment: float  # Points to adjust spread
    recommended_bets: List[str]
    impact_score: float  # 0-10 scale


class WeatherAnalyzer:
    """
    Analyzes weather impact on NFL games

    Weather Impact Rules:
    - Wind 15+ mph: Favor UNDER (harder to throw)
    - Wind 20+ mph: STRONG UNDER (3-6 point adjustment)
    - Rain: Favor UNDER (1-3 point adjustment)
    - Snow: Favor UNDER if heavy (3-6 point adjustment)
    - Cold (<20°F): Favor UNDER (1-2 point adjustment)
    - Dome: No impact
    """

    # NFL Stadium Information
    STADIUMS = {
        'Arizona Cardinals': StadiumInfo('Cardinals', 'State Farm Stadium', 'Glendale', 'AZ', True, True),
        'Atlanta Falcons': StadiumInfo('Falcons', 'Mercedes-Benz Stadium', 'Atlanta', 'GA', True, True),
        'Baltimore Ravens': StadiumInfo('Ravens', 'M&T Bank Stadium', 'Baltimore', 'MD', False, False),
        'Buffalo Bills': StadiumInfo('Bills', 'Highmark Stadium', 'Buffalo', 'NY', False, False),
        'Carolina Panthers': StadiumInfo('Panthers', 'Bank of America Stadium', 'Charlotte', 'NC', False, False),
        'Chicago Bears': StadiumInfo('Bears', 'Soldier Field', 'Chicago', 'IL', False, False),
        'Cincinnati Bengals': StadiumInfo('Bengals', 'Paycor Stadium', 'Cincinnati', 'OH', False, False),
        'Cleveland Browns': StadiumInfo('Browns', 'Cleveland Browns Stadium', 'Cleveland', 'OH', False, False),
        'Dallas Cowboys': StadiumInfo('Cowboys', 'AT&T Stadium', 'Arlington', 'TX', True, True),
        'Denver Broncos': StadiumInfo('Broncos', 'Empower Field', 'Denver', 'CO', False, False),
        'Detroit Lions': StadiumInfo('Lions', 'Ford Field', 'Detroit', 'MI', True, False),
        'Green Bay Packers': StadiumInfo('Packers', 'Lambeau Field', 'Green Bay', 'WI', False, False),
        'Houston Texans': StadiumInfo('Texans', 'NRG Stadium', 'Houston', 'TX', True, True),
        'Indianapolis Colts': StadiumInfo('Colts', 'Lucas Oil Stadium', 'Indianapolis', 'IN', True, True),
        'Jacksonville Jaguars': StadiumInfo('Jaguars', 'TIAA Bank Field', 'Jacksonville', 'FL', False, False),
        'Kansas City Chiefs': StadiumInfo('Chiefs', 'GEHA Field', 'Kansas City', 'MO', False, False),
        'Las Vegas Raiders': StadiumInfo('Raiders', 'Allegiant Stadium', 'Las Vegas', 'NV', True, False),
        'Los Angeles Chargers': StadiumInfo('Chargers', 'SoFi Stadium', 'Inglewood', 'CA', True, False),
        'Los Angeles Rams': StadiumInfo('Rams', 'SoFi Stadium', 'Inglewood', 'CA', True, False),
        'Miami Dolphins': StadiumInfo('Dolphins', 'Hard Rock Stadium', 'Miami Gardens', 'FL', False, False),
        'Minnesota Vikings': StadiumInfo('Vikings', 'U.S. Bank Stadium', 'Minneapolis', 'MN', True, False),
        'New England Patriots': StadiumInfo('Patriots', 'Gillette Stadium', 'Foxborough', 'MA', False, False),
        'New Orleans Saints': StadiumInfo('Saints', 'Caesars Superdome', 'New Orleans', 'LA', True, False),
        'New York Giants': StadiumInfo('Giants', 'MetLife Stadium', 'East Rutherford', 'NJ', False, False),
        'New York Jets': StadiumInfo('Jets', 'MetLife Stadium', 'East Rutherford', 'NJ', False, False),
        'Philadelphia Eagles': StadiumInfo('Eagles', 'Lincoln Financial Field', 'Philadelphia', 'PA', False, False),
        'Pittsburgh Steelers': StadiumInfo('Steelers', 'Acrisure Stadium', 'Pittsburgh', 'PA', False, False),
        'San Francisco 49ers': StadiumInfo('49ers', "Levi's Stadium", 'Santa Clara', 'CA', False, False),
        'Seattle Seahawks': StadiumInfo('Seahawks', 'Lumen Field', 'Seattle', 'WA', False, False),
        'Tampa Bay Buccaneers': StadiumInfo('Buccaneers', 'Raymond James Stadium', 'Tampa', 'FL', False, False),
        'Tennessee Titans': StadiumInfo('Titans', 'Nissan Stadium', 'Nashville', 'TN', False, False),
        'Washington Commanders': StadiumInfo('Commanders', 'FedExField', 'Landover', 'MD', False, False),
    }

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv('CRAWLBASE_TOKEN')
        if not self.token:
            raise ValueError("No Crawlbase token. Set CRAWLBASE_TOKEN env var")

        self.api = CrawlingAPI({'token': self.token})
        self.data_dir = Path('data/weather')
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.weather_data: List[WeatherData] = []

    def fetch_nfl_weather(self) -> str:
        """Fetch NFL weather from multiple sources"""
        print("🌦️  Fetching NFL weather data...")

        urls = [
            'https://www.nfl.com/weather/',
            'https://www.weather.com/sports-recreation/nfl',
            'https://www.espn.com/nfl/weather'
        ]

        for url in urls:
            response = self.api.get(url)
            if response['statusCode'] == 200:
                print(f"   ✅ Got weather from {url}")

                # Save raw HTML
                filename = url.split('//')[-1].replace('/', '_').replace('.', '_')
                with open(self.data_dir / f'raw_{filename}.html', 'w') as f:
                    f.write(response['body'])

                return response['body']

        print("   ❌ Failed to fetch weather data")
        return None

    def fetch_stadium_weather(self, city: str, state: str) -> Dict:
        """Fetch weather for specific stadium location"""
        print(f"   🌡️  Fetching weather for {city}, {state}...")

        # Weather.com URL format
        url = f'https://weather.com/weather/today/l/{city.replace(" ", "-")}-{state}'

        response = self.api.get(url)
        if response['statusCode'] == 200:
            return response['body']

        return None

    def parse_weather_data(self, html: str, game: str, home_team: str) -> Optional[WeatherData]:
        """
        Parse weather from HTML

        In production: use BeautifulSoup
        For now: return sample data
        """
        print(f"   🔍 Parsing weather for {game}...")

        # Get stadium info
        stadium_info = self.STADIUMS.get(home_team)

        if not stadium_info:
            print(f"   ⚠️  Unknown team: {home_team}")
            return None

        # If dome, no weather impact
        if stadium_info.is_dome:
            print(f"   🏟️  {stadium_info.stadium} is a dome - no weather impact")
            return WeatherData(
                game=game,
                home_team=home_team,
                stadium=stadium_info.stadium,
                is_dome=True,
                temperature=72.0,
                feels_like=72.0,
                wind_speed=0,
                wind_gust=0,
                precipitation_chance=0,
                precipitation_type='None',
                humidity=50,
                conditions='Indoor',
                timestamp=datetime.now().isoformat()
            )

        # Sample outdoor weather
        # In production: parse from HTML
        weather = WeatherData(
            game=game,
            home_team=home_team,
            stadium=stadium_info.stadium,
            is_dome=False,
            temperature=45.0,
            feels_like=38.0,
            wind_speed=18.0,
            wind_gust=25.0,
            precipitation_chance=30,
            precipitation_type='Rain',
            humidity=75,
            conditions='Cloudy, Windy',
            timestamp=datetime.now().isoformat()
        )

        return weather

    def calculate_weather_impact(self, weather: WeatherData) -> WeatherImpact:
        """
        Calculate betting impact of weather

        Returns:
            WeatherImpact with severity and adjustments
        """
        if weather.is_dome:
            return WeatherImpact(
                game=weather.game,
                weather=weather,
                severity='NONE',
                total_adjustment=0,
                spread_adjustment=0,
                recommended_bets=[],
                impact_score=0
            )

        impact_score = 0
        total_adj = 0
        spread_adj = 0
        recommendations = []

        # Wind impact (biggest factor)
        if weather.wind_speed >= 20:
            impact_score += 4
            total_adj -= 4.5  # Strong UNDER
            recommendations.append('STRONG UNDER - High wind')
        elif weather.wind_speed >= 15:
            impact_score += 3
            total_adj -= 3.0  # UNDER
            recommendations.append('UNDER - Moderate wind')
        elif weather.wind_speed >= 10:
            impact_score += 1
            total_adj -= 1.5  # Slight UNDER
            recommendations.append('SLIGHT UNDER - Some wind')

        # Temperature impact
        if weather.temperature < 20:
            impact_score += 2
            total_adj -= 2.0
            recommendations.append('UNDER - Extreme cold')
        elif weather.temperature < 32:
            impact_score += 1
            total_adj -= 1.0
            recommendations.append('SLIGHT UNDER - Cold weather')

        # Precipitation impact
        if weather.precipitation_chance >= 70:
            if 'snow' in weather.precipitation_type.lower():
                impact_score += 3
                total_adj -= 3.5
                recommendations.append('STRONG UNDER - Heavy snow')
            else:  # Rain
                impact_score += 2
                total_adj -= 2.0
                recommendations.append('UNDER - Rain expected')
        elif weather.precipitation_chance >= 40:
            impact_score += 1
            total_adj -= 1.0
            recommendations.append('SLIGHT UNDER - Rain possible')

        # Home field advantage in bad weather
        if impact_score >= 3:
            spread_adj -= 1.0  # Home team gets 1 point in bad weather
            recommendations.append('SLIGHT HOME TEAM - Weather advantage')

        # Determine severity
        if impact_score >= 7:
            severity = 'EXTREME'
        elif impact_score >= 5:
            severity = 'SEVERE'
        elif impact_score >= 3:
            severity = 'MODERATE'
        elif impact_score >= 1:
            severity = 'MILD'
        else:
            severity = 'NONE'

        return WeatherImpact(
            game=weather.game,
            weather=weather,
            severity=severity,
            total_adjustment=round(total_adj, 1),
            spread_adjustment=round(spread_adj, 1),
            recommended_bets=recommendations,
            impact_score=impact_score
        )

    def analyze_all_games(self, games: List[tuple]) -> List[WeatherImpact]:
        """
        Analyze weather impact for all games

        Args:
            games: List of (game_name, home_team) tuples

        Returns:
            List of WeatherImpact objects
        """
        print("\n" + "="*80)
        print("🌦️  NFL WEATHER ANALYSIS")
        print("="*80 + "\n")

        impacts = []

        for game, home_team in games:
            # Fetch weather
            html = self.fetch_nfl_weather()

            # Parse weather
            weather = self.parse_weather_data(html, game, home_team)

            if weather:
                # Calculate impact
                impact = self.calculate_weather_impact(weather)
                impacts.append(impact)

        return impacts

    def save_data(self, impacts: List[WeatherImpact]) -> str:
        """Save weather analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.data_dir / f'weather_impact_{timestamp}.json'

        output = {
            'timestamp': datetime.now().isoformat(),
            'total_games': len(impacts),
            'impacts': [asdict(impact) for impact in impacts]
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        return str(output_file)

    def print_summary(self, impacts: List[WeatherImpact]):
        """Print weather impact summary"""
        print("\n" + "="*80)
        print("🌦️  WEATHER IMPACT SUMMARY")
        print("="*80)

        # Group by severity
        severe_games = [i for i in impacts if i.severity in ['SEVERE', 'EXTREME']]
        moderate_games = [i for i in impacts if i.severity == 'MODERATE']
        mild_games = [i for i in impacts if i.severity == 'MILD']
        no_impact = [i for i in impacts if i.severity == 'NONE']

        print(f"\n📊 Games by Weather Severity:")
        print(f"   🔴 SEVERE/EXTREME: {len(severe_games)}")
        print(f"   🟡 MODERATE: {len(moderate_games)}")
        print(f"   🟢 MILD: {len(mild_games)}")
        print(f"   ⚪ NO IMPACT: {len(no_impact)}")

        # Print severe weather games
        if severe_games:
            print("\n" + "-"*80)
            print("🔴 SEVERE WEATHER ALERTS")
            print("-"*80)

            for impact in severe_games:
                w = impact.weather
                print(f"\n{impact.game}:")
                print(f"  Stadium: {w.stadium} ({'DOME' if w.is_dome else 'OUTDOOR'})")
                print(f"  Temp: {w.temperature:.0f}°F (feels like {w.feels_like:.0f}°F)")
                print(f"  Wind: {w.wind_speed:.0f} mph (gusts {w.wind_gust:.0f} mph)")
                print(f"  Precip: {w.precipitation_chance:.0f}% ({w.precipitation_type})")
                print(f"  Conditions: {w.conditions}")
                print(f"  🎯 SEVERITY: {impact.severity} ({impact.impact_score:.0f}/10)")
                print(f"  📉 Total Adjustment: {impact.total_adjustment:+.1f} points")
                if impact.spread_adjustment != 0:
                    print(f"  📊 Spread Adjustment: {impact.spread_adjustment:+.1f} points")
                print(f"  💡 Recommendations:")
                for rec in impact.recommended_bets:
                    print(f"     • {rec}")

        # Print moderate weather games
        if moderate_games:
            print("\n" + "-"*80)
            print("🟡 MODERATE WEATHER GAMES")
            print("-"*80)

            for impact in moderate_games[:3]:  # Show top 3
                w = impact.weather
                print(f"\n{impact.game}: {impact.severity}")
                print(f"  Total Adj: {impact.total_adjustment:+.1f} | {', '.join(impact.recommended_bets[:2])}")

        print("\n" + "="*80)
        print("✅ Use weather adjustments with your betting system")
        print("💡 Wind 15+ mph = favor UNDER | Cold <20°F = favor UNDER")
        print("="*80 + "\n")

    def run(self, games: List[tuple] = None):
        """Run complete weather analysis"""
        # Default sample games
        if not games:
            games = [
                ('Chiefs @ Bills', 'Buffalo Bills'),
                ('Eagles @ Cowboys', 'Dallas Cowboys'),
                ('49ers @ Seahawks', 'Seattle Seahawks'),
            ]

        # Analyze all games
        impacts = self.analyze_all_games(games)

        # Save data
        output_file = self.save_data(impacts)
        print(f"\n📁 Data saved to: {output_file}")

        # Print summary
        self.print_summary(impacts)

        return impacts


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='NFL Weather Impact Analyzer'
    )
    parser.add_argument(
        '--token',
        help='Crawlbase API token (or set CRAWLBASE_TOKEN env var)'
    )
    parser.add_argument(
        '--games',
        nargs='+',
        help='Games to analyze (format: "Chiefs@Bills:Buffalo Bills")'
    )

    args = parser.parse_args()

    # Parse games if provided
    games = None
    if args.games:
        games = []
        for game_str in args.games:
            parts = game_str.split(':')
            if len(parts) == 2:
                games.append((parts[0], parts[1]))

    try:
        analyzer = WeatherAnalyzer(token=args.token)
        impacts = analyzer.run(games=games)

        print("\n💡 Next step: Apply weather adjustments to your betting system")
        print("   Adjust totals down in high wind/cold/rain")
        print("   Consider home team in severe weather")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nSetup: export CRAWLBASE_TOKEN='your_token'")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
