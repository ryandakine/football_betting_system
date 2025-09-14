"""
Game Data Enricher - Adds weather, injury, and other contextual data to football games
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    """Weather conditions for a game"""
    temperature_f: float
    temperature_c: float
    conditions: str
    humidity: int
    wind_speed_mph: float
    wind_speed_kmh: float
    wind_direction: str
    precipitation_chance: int
    visibility_miles: float
    uv_index: int
    feels_like_f: float
    feels_like_c: float
    pressure_mb: float
    dew_point_f: float
    cloud_cover: int

@dataclass
class InjuryData:
    """Injury information for a player"""
    player_name: str
    team: str
    position: str
    injury_type: str
    injury_status: str  # Questionable, Doubtful, Out, etc.
    expected_return: Optional[str]
    last_updated: str

@dataclass
class TeamInjuryReport:
    """Injury report for a team"""
    team_name: str
    injuries: List[InjuryData]
    total_out: int
    total_questionable: int
    total_doubtful: int
    last_updated: str

@dataclass
class GameFactors:
    """External factors affecting game outcome"""
    weather_impact: str  # High, Medium, Low impact on game
    key_injuries: List[str]  # List of key player injuries
    travel_distance: Optional[float]  # Miles traveled for away team
    rest_days: int  # Days since last game
    altitude: Optional[int]  # Feet above sea level
    grass_type: str  # Natural, Turf, Hybrid
    time_of_day: str  # Early, Afternoon, Night, Late Night

@dataclass
class FootballGameData:
    """Enhanced game data with weather, injuries, and factors"""
    game_id: str
    home_team: str
    away_team: str
    game_time: str
    venue: str
    location: str
    weather: Optional[WeatherData]
    home_injuries: Optional[TeamInjuryReport]
    away_injuries: Optional[TeamInjuryReport]
    game_factors: GameFactors
    last_updated: str

class GameDataEnricher:
    """Enriches football game data with weather, injury reports, and other factors"""

    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys

        # API endpoints and keys
        self.weather_api_key = api_keys.get('openweather')
        self.sports_data_api_key = api_keys.get('sportsdata')

        # Cache for API responses
        self.weather_cache = {}
        self.injury_cache = {}
        self.cache_expiry = 3600  # 1 hour

        # Team venue mapping for weather lookups
        self.team_venues = {
            # NFL Teams
            'Kansas City Chiefs': {'venue': 'Arrowhead Stadium', 'city': 'Kansas City', 'state': 'MO'},
            'Buffalo Bills': {'venue': 'Highmark Stadium', 'city': 'Orchard Park', 'state': 'NY'},
            'Philadelphia Eagles': {'venue': 'Lincoln Financial Field', 'city': 'Philadelphia', 'state': 'PA'},
            'San Francisco 49ers': {'venue': 'Levis Stadium', 'city': 'Santa Clara', 'state': 'CA'},
            'Detroit Lions': {'venue': 'Ford Field', 'city': 'Detroit', 'state': 'MI'},
            'Cleveland Browns': {'venue': 'FirstEnergy Stadium', 'city': 'Cleveland', 'state': 'OH'},
            'Dallas Cowboys': {'venue': 'AT&T Stadium', 'city': 'Arlington', 'state': 'TX'},
            'Miami Dolphins': {'venue': 'Hard Rock Stadium', 'city': 'Miami Gardens', 'state': 'FL'},
            'Jacksonville Jaguars': {'venue': 'TIAA Bank Field', 'city': 'Jacksonville', 'state': 'FL'},
            'New England Patriots': {'venue': 'Gillette Stadium', 'city': 'Foxborough', 'state': 'MA'},
            'Pittsburgh Steelers': {'venue': 'Heinz Field', 'city': 'Pittsburgh', 'state': 'PA'},
            'Washington Commanders': {'venue': 'FedEx Field', 'city': 'Landover', 'state': 'MD'},
            'Tennessee Titans': {'venue': 'Nissan Stadium', 'city': 'Nashville', 'state': 'TN'},
            'Indianapolis Colts': {'venue': 'Lucas Oil Stadium', 'city': 'Indianapolis', 'state': 'IN'},
            'Cincinnati Bengals': {'venue': 'Paycor Stadium', 'city': 'Cincinnati', 'state': 'OH'},
            'Seattle Seahawks': {'venue': 'Lumen Field', 'city': 'Seattle', 'state': 'WA'},
            'Arizona Cardinals': {'venue': 'State Farm Stadium', 'city': 'Glendale', 'state': 'AZ'},
            'Atlanta Falcons': {'venue': 'Mercedes-Benz Stadium', 'city': 'Atlanta', 'state': 'GA'},
            'Carolina Panthers': {'venue': 'Bank of America Stadium', 'city': 'Charlotte', 'state': 'NC'},
            'New Orleans Saints': {'venue': 'Caesars Superdome', 'city': 'New Orleans', 'state': 'LA'},
            'Tampa Bay Buccaneers': {'venue': 'Raymond James Stadium', 'city': 'Tampa', 'state': 'FL'},
            'Green Bay Packers': {'venue': 'Lambeau Field', 'city': 'Green Bay', 'state': 'WI'},
            'Chicago Bears': {'venue': 'Soldier Field', 'city': 'Chicago', 'state': 'IL'},
            'Minnesota Vikings': {'venue': 'U.S. Bank Stadium', 'city': 'Minneapolis', 'state': 'MN'},
            'New York Giants': {'venue': 'MetLife Stadium', 'city': 'East Rutherford', 'state': 'NJ'},
            'New York Jets': {'venue': 'MetLife Stadium', 'city': 'East Rutherford', 'state': 'NJ'},
            'Las Vegas Raiders': {'venue': 'Allegiant Stadium', 'city': 'Las Vegas', 'state': 'NV'},
            'Los Angeles Chargers': {'venue': 'SoFi Stadium', 'city': 'Inglewood', 'state': 'CA'},
            'Los Angeles Rams': {'venue': 'SoFi Stadium', 'city': 'Inglewood', 'state': 'CA'},
            'Denver Broncos': {'venue': 'Empower Field at Mile High', 'city': 'Denver', 'state': 'CO'},
            'Baltimore Ravens': {'venue': 'M&T Bank Stadium', 'city': 'Baltimore', 'state': 'MD'},

            # College Football (Major Programs)
            'Alabama Crimson Tide': {'venue': 'Bryant-Denny Stadium', 'city': 'Tuscaloosa', 'state': 'AL'},
            'Clemson Tigers': {'venue': 'Memorial Stadium', 'city': 'Clemson', 'state': 'SC'},
            'Ohio State Buckeyes': {'venue': 'Ohio Stadium', 'city': 'Columbus', 'state': 'OH'},
            'Oklahoma Sooners': {'venue': 'Memorial Stadium', 'city': 'Norman', 'state': 'OK'},
            'Texas Longhorns': {'venue': 'DKR-Texas Memorial Stadium', 'city': 'Austin', 'state': 'TX'},
            'USC Trojans': {'venue': 'Los Angeles Memorial Coliseum', 'city': 'Los Angeles', 'state': 'CA'},
            'Oregon Ducks': {'venue': 'Autzen Stadium', 'city': 'Eugene', 'state': 'OR'},
            'Florida Gators': {'venue': 'Ben Hill Griffin Stadium', 'city': 'Gainesville', 'state': 'FL'},
            'Georgia Bulldogs': {'venue': 'Sanford Stadium', 'city': 'Athens', 'state': 'GA'},
            'LSU Tigers': {'venue': 'Tiger Stadium', 'city': 'Baton Rouge', 'state': 'LA'},
            'Michigan Wolverines': {'venue': 'The Big House', 'city': 'Ann Arbor', 'state': 'MI'},
            'Penn State Nittany Lions': {'venue': 'Beaver Stadium', 'city': 'University Park', 'state': 'PA'},
            'Notre Dame Fighting Irish': {'venue': 'Notre Dame Stadium', 'city': 'Notre Dame', 'state': 'IN'},
            'Miami Hurricanes': {'venue': 'Hard Rock Stadium', 'city': 'Miami Gardens', 'state': 'FL'},
            'Nebraska Cornhuskers': {'venue': 'Memorial Stadium', 'city': 'Lincoln', 'state': 'NE'},
            'Auburn Tigers': {'venue': 'Jordan-Hare Stadium', 'city': 'Auburn', 'state': 'AL'}
        }

    def enrich_game_data(self, game_data: Dict[str, Any]) -> FootballGameData:
        """Enrich basic game data with weather, injuries, and other factors"""

        game_id = game_data.get('id', game_data.get('game_id', 'unknown'))
        home_team = game_data.get('home_team', 'Unknown')
        away_team = game_data.get('away_team', 'Unknown')

        # Get weather data
        weather = self._get_weather_data(home_team, game_data.get('commence_time'))

        # Get injury reports
        home_injuries = self._get_injury_report(home_team)
        away_injuries = self._get_injury_report(away_team)

        # Calculate game factors
        game_factors = self._calculate_game_factors(game_data, weather, home_injuries, away_injuries)

        return FootballGameData(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            game_time=game_data.get('commence_time', ''),
            venue=self.team_venues.get(home_team, {}).get('venue', 'Unknown'),
            location=f"{self.team_venues.get(home_team, {}).get('city', 'Unknown')}, {self.team_venues.get(home_team, {}).get('state', 'Unknown')}",
            weather=weather,
            home_injuries=home_injuries,
            away_injuries=away_injuries,
            game_factors=game_factors,
            last_updated=datetime.now().isoformat()
        )

    def _get_weather_data(self, home_team: str, game_time: Optional[str]) -> Optional[WeatherData]:
        """Fetch weather data for game location and time"""
        if not self.weather_api_key:
            logger.warning("No OpenWeather API key provided")
            return None

        venue_info = self.team_venues.get(home_team)
        if not venue_info:
            logger.warning(f"No venue info for {home_team}")
            return None

        city = venue_info['city']

        # Parse game time
        try:
            if game_time:
                game_datetime = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
            else:
                game_datetime = datetime.now() + timedelta(days=1)  # Default to tomorrow
        except:
            game_datetime = datetime.now() + timedelta(days=1)

        cache_key = f"{city}_{game_datetime.date().isoformat()}"

        # Check cache
        if cache_key in self.weather_cache:
            cached_data, timestamp = self.weather_cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                return cached_data

        try:
            # OpenWeatherMap API call
            url = f"http://api.openweathermap.org/data/2.5/forecast"
            params = {
                'q': f"{city},US",
                'appid': self.weather_api_key,
                'units': 'imperial'
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Weather API error: {response.status_code}")
                return None

            data = response.json()

            # Find forecast closest to game time
            target_hour = game_datetime.hour
            closest_forecast = min(data['list'],
                                 key=lambda x: abs(datetime.fromtimestamp(x['dt']).hour - target_hour))

            weather_data = WeatherData(
                temperature_f=round(closest_forecast['main']['temp'], 1),
                temperature_c=round((closest_forecast['main']['temp'] - 32) * 5/9, 1),
                conditions=closest_forecast['weather'][0]['description'].title(),
                humidity=closest_forecast['main']['humidity'],
                wind_speed_mph=round(closest_forecast['wind']['speed'], 1),
                wind_speed_kmh=round(closest_forecast['wind']['speed'] * 1.60934, 1),
                wind_direction=self._get_wind_direction(closest_forecast['wind'].get('deg', 0)),
                precipitation_chance=round(closest_forecast.get('pop', 0) * 100),
                visibility_miles=5.0,  # Default visibility
                uv_index=0,  # Not available in free API
                feels_like_f=round(closest_forecast['main']['feels_like'], 1),
                feels_like_c=round((closest_forecast['main']['feels_like'] - 32) * 5/9, 1),
                pressure_mb=closest_forecast['main']['pressure'],
                dew_point_f=round(closest_forecast['main'].get('dew_point', closest_forecast['main']['temp'] - 10), 1),
                cloud_cover=closest_forecast.get('clouds', {}).get('all', 0)
            )

            # Cache the result
            self.weather_cache[cache_key] = (weather_data, time.time())

            return weather_data

        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None

    def _get_wind_direction(self, degrees: float) -> str:
        """Convert wind degrees to cardinal direction"""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        index = round(degrees / 22.5) % 16
        return directions[index]

    def _get_injury_report(self, team_name: str) -> Optional[TeamInjuryReport]:
        """Fetch injury report for a team"""
        if not self.sports_data_api_key:
            # Return mock data if no API key
            return self._get_mock_injury_report(team_name)

        cache_key = f"injuries_{team_name}"

        # Check cache
        if cache_key in self.injury_cache:
            cached_data, timestamp = self.injury_cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                return cached_data

        try:
            # Use ESPN API or similar for injury data
            # This is a simplified version - in production you'd use proper sports APIs
            injuries = self._get_mock_injury_report(team_name)

            # Cache the result
            self.injury_cache[cache_key] = (injuries, time.time())

            return injuries

        except Exception as e:
            logger.error(f"Error fetching injury data for {team_name}: {e}")
            return self._get_mock_injury_report(team_name)

    def _get_mock_injury_report(self, team_name: str) -> TeamInjuryReport:
        """Generate realistic mock injury data"""
        # This creates plausible injury scenarios for demonstration
        mock_injuries = []

        # Generate some realistic injury scenarios based on team
        injury_scenarios = {
            'Kansas City Chiefs': [
                InjuryData("Patrick Mahomes", "Kansas City Chiefs", "QB", "Ankle", "Questionable", "Week 15", datetime.now().isoformat()),
                InjuryData("Travis Kelce", "Kansas City Chiefs", "TE", "Knee", "Doubtful", "Week 16", datetime.now().isoformat())
            ],
            'Buffalo Bills': [
                InjuryData("Josh Allen", "Buffalo Bills", "QB", "Shoulder", "Questionable", "This Week", datetime.now().isoformat())
            ],
            'Detroit Lions': [
                InjuryData("Jared Goff", "Detroit Lions", "QB", "Back", "Out", "Week 17", datetime.now().isoformat()),
                InjuryData("Aidan Hutchinson", "Detroit Lions", "DE", "Hamstring", "Questionable", "This Week", datetime.now().isoformat())
            ]
        }

        injuries = injury_scenarios.get(team_name, [])

        total_out = sum(1 for inj in injuries if inj.injury_status == "Out")
        total_questionable = sum(1 for inj in injuries if inj.injury_status == "Questionable")
        total_doubtful = sum(1 for inj in injuries if inj.injury_status == "Doubtful")

        return TeamInjuryReport(
            team_name=team_name,
            injuries=injuries,
            total_out=total_out,
            total_questionable=total_questionable,
            total_doubtful=total_doubtful,
            last_updated=datetime.now().isoformat()
        )

    def _calculate_game_factors(self, game_data: Dict, weather: Optional[WeatherData],
                              home_injuries: Optional[TeamInjuryReport],
                              away_injuries: Optional[TeamInjuryReport]) -> GameFactors:
        """Calculate various external factors affecting the game"""

        # Weather impact assessment
        weather_impact = self._assess_weather_impact(weather)

        # Key injuries assessment
        key_injuries = self._identify_key_injuries(home_injuries, away_injuries)

        # Travel distance (simplified - would need actual location data)
        travel_distance = None  # Would calculate based on team locations

        # Rest days (simplified)
        rest_days = 7  # Default assumption

        # Altitude
        altitude = self._get_venue_altitude(game_data.get('home_team'))

        # Grass type
        grass_type = self._get_venue_grass_type(game_data.get('home_team'))

        # Time of day
        time_of_day = self._get_time_of_day(game_data.get('commence_time'))

        return GameFactors(
            weather_impact=weather_impact,
            key_injuries=key_injuries,
            travel_distance=travel_distance,
            rest_days=rest_days,
            altitude=altitude,
            grass_type=grass_type,
            time_of_day=time_of_day
        )

    def _assess_weather_impact(self, weather: Optional[WeatherData]) -> str:
        """Assess weather impact on game outcome"""
        if not weather:
            return "Unknown"

        # High impact conditions
        if weather.temperature_f < 32:  # Freezing
            return "High"
        if weather.wind_speed_mph > 20:  # High winds
            return "High"
        if weather.precipitation_chance > 70:  # Heavy rain/snow expected
            return "High"
        if weather.temperature_f > 90:  # Extreme heat
            return "Medium"

        # Medium impact conditions
        if weather.wind_speed_mph > 15:
            return "Medium"
        if weather.precipitation_chance > 40:
            return "Medium"
        if weather.temperature_f < 40 or weather.temperature_f > 80:
            return "Medium"

        return "Low"

    def _identify_key_injuries(self, home_injuries: Optional[TeamInjuryReport],
                             away_injuries: Optional[TeamInjuryReport]) -> List[str]:
        """Identify key injuries that could affect game outcome"""
        key_injuries = []

        # Check quarterback injuries (highest impact)
        for team, injuries in [("Home", home_injuries), ("Away", away_injuries)]:
            if injuries:
                for injury in injuries.injuries:
                    if injury.position == "QB" and injury.injury_status in ["Out", "Doubtful"]:
                        key_injuries.append(f"{team} QB {injury.player_name} ({injury.injury_status})")

        # Check other key positions
        key_positions = ["RB", "WR", "TE", "OL", "DL", "LB", "DB"]
        for team, injuries in [("Home", home_injuries), ("Away", away_injuries)]:
            if injuries:
                for injury in injuries.injuries:
                    if injury.position in key_positions and injury.injury_status == "Out":
                        key_injuries.append(f"{team} {injury.position} {injury.player_name}")

        return key_injuries

    def _get_venue_altitude(self, home_team: str) -> Optional[int]:
        """Get venue altitude in feet"""
        altitude_map = {
            'Denver Broncos': 5280,  # Mile High Stadium
            'Salt Lake City': 4226,  # General area
            'Albuquerque': 5312,    # General area
        }

        venue_info = self.team_venues.get(home_team, {})
        city = venue_info.get('city')

        return altitude_map.get(city)

    def _get_venue_grass_type(self, home_team: str) -> str:
        """Get venue grass type"""
        # Most NFL stadiums are turf now
        turf_stadiums = [
            'Arrowhead Stadium', 'Highmark Stadium', 'Lincoln Financial Field',
            'Levis Stadium', 'Ford Field', 'FirstEnergy Stadium', 'AT&T Stadium',
            'Hard Rock Stadium', 'TIAA Bank Field', 'Gillette Stadium', 'Heinz Field',
            'FedEx Field', 'Nissan Stadium', 'Lucas Oil Stadium', 'Paycor Stadium',
            'Lumen Field', 'State Farm Stadium', 'Mercedes-Benz Stadium',
            'Bank of America Stadium', 'Caesars Superdome', 'Raymond James Stadium',
            'Soldier Field', 'U.S. Bank Stadium', 'MetLife Stadium',
            'Allegiant Stadium', 'SoFi Stadium', 'Empower Field at Mile High',
            'M&T Bank Stadium'
        ]

        venue = self.team_venues.get(home_team, {}).get('venue')
        return "Turf" if venue in turf_stadiums else "Natural"

    def _get_time_of_day(self, game_time: Optional[str]) -> str:
        """Determine time of day for game"""
        if not game_time:
            return "Afternoon"

        try:
            dt = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
            hour = dt.hour

            if 6 <= hour < 12:
                return "Morning"
            elif 12 <= hour < 17:
                return "Afternoon"
            elif 17 <= hour < 21:
                return "Evening"
            else:
                return "Night"
        except:
            return "Afternoon"

    def get_weather_summary(self, weather: Optional[WeatherData]) -> str:
        """Get human-readable weather summary"""
        if not weather:
            return "Weather data unavailable"
        
        return f"{weather.temperature_f:.1f}°F/{weather.temperature_c:.1f}°C, {weather.conditions}, "  \
f"{weather.wind_speed_mph:.1f} mph {weather.wind_direction}"

    def get_injury_summary(self, injury_report: Optional[TeamInjuryReport]) -> str:
        """Get human-readable injury summary"""
        if not injury_report or not injury_report.injuries:
            return "No reported injuries"

        summary = f"{len(injury_report.injuries)} total injuries"
        if injury_report.total_out > 0:
            summary += f", {injury_report.total_out} out"
        if injury_report.total_doubtful > 0:
            summary += f", {injury_report.total_doubtful} doubtful"
        if injury_report.total_questionable > 0:
            summary += f", {injury_report.total_questionable} questionable"

        return summary
