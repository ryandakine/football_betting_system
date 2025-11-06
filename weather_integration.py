#!/usr/bin/env python3
"""
Weather Integration for NFL Betting
====================================
Fetches game-day weather conditions and calculates impact on totals.

Weather Impact Rules:
- Wind >15 mph: Favor UNDER (passing game affected)
- Heavy rain/snow: Favor UNDER (slippery ball, conservative play)
- Dome games: No weather impact
- Temperature <20°F or >95°F: Slight UNDER lean
"""

import asyncio
import json
import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import aiohttp

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/referee_conspiracy")
WEATHER_CACHE_DIR = DATA_DIR / "weather_cache"
WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# NFL Stadium locations (lat, lon)
STADIUM_LOCATIONS = {
    "ARI": {"lat": 33.5276, "lon": -112.2626, "name": "State Farm Stadium", "dome": True},
    "ATL": {"lat": 33.7554, "lon": -84.4008, "name": "Mercedes-Benz Stadium", "dome": True},
    "BAL": {"lat": 39.2780, "lon": -76.6227, "name": "M&T Bank Stadium", "dome": False},
    "BUF": {"lat": 42.7738, "lon": -78.7870, "name": "Highmark Stadium", "dome": False},
    "CAR": {"lat": 35.2258, "lon": -80.8529, "name": "Bank of America Stadium", "dome": False},
    "CHI": {"lat": 41.8623, "lon": -87.6167, "name": "Soldier Field", "dome": False},
    "CIN": {"lat": 39.0954, "lon": -84.5160, "name": "Paycor Stadium", "dome": False},
    "CLE": {"lat": 41.5061, "lon": -81.6995, "name": "Cleveland Browns Stadium", "dome": False},
    "DAL": {"lat": 32.7473, "lon": -97.0945, "name": "AT&T Stadium", "dome": True},
    "DEN": {"lat": 39.7439, "lon": -105.0201, "name": "Empower Field", "dome": False},
    "DET": {"lat": 42.3400, "lon": -83.0456, "name": "Ford Field", "dome": True},
    "GB": {"lat": 44.5013, "lon": -88.0622, "name": "Lambeau Field", "dome": False},
    "HOU": {"lat": 29.6847, "lon": -95.4107, "name": "NRG Stadium", "dome": True},
    "IND": {"lat": 39.7601, "lon": -86.1639, "name": "Lucas Oil Stadium", "dome": True},
    "JAX": {"lat": 30.3240, "lon": -81.6373, "name": "TIAA Bank Field", "dome": False},
    "KC": {"lat": 39.0489, "lon": -94.4839, "name": "GEHA Field at Arrowhead", "dome": False},
    "LV": {"lat": 36.0908, "lon": -115.1833, "name": "Allegiant Stadium", "dome": True},
    "LAC": {"lat": 33.9534, "lon": -118.3390, "name": "SoFi Stadium", "dome": True},
    "LA": {"lat": 33.9534, "lon": -118.3390, "name": "SoFi Stadium", "dome": True},
    "MIA": {"lat": 25.9580, "lon": -80.2389, "name": "Hard Rock Stadium", "dome": False},
    "MIN": {"lat": 44.9738, "lon": -93.2577, "name": "U.S. Bank Stadium", "dome": True},
    "NE": {"lat": 42.0909, "lon": -71.2643, "name": "Gillette Stadium", "dome": False},
    "NO": {"lat": 29.9511, "lon": -90.0812, "name": "Caesars Superdome", "dome": True},
    "NYG": {"lat": 40.8128, "lon": -74.0742, "name": "MetLife Stadium", "dome": False},
    "NYJ": {"lat": 40.8128, "lon": -74.0742, "name": "MetLife Stadium", "dome": False},
    "PHI": {"lat": 39.9008, "lon": -75.1675, "name": "Lincoln Financial Field", "dome": False},
    "PIT": {"lat": 40.4468, "lon": -80.0158, "name": "Acrisure Stadium", "dome": False},
    "SF": {"lat": 37.4032, "lon": -121.9698, "name": "Levi's Stadium", "dome": False},
    "SEA": {"lat": 47.5952, "lon": -122.3316, "name": "Lumen Field", "dome": False},
    "TB": {"lat": 27.9759, "lon": -82.5033, "name": "Raymond James Stadium", "dome": False},
    "TEN": {"lat": 36.1665, "lon": -86.7713, "name": "Nissan Stadium", "dome": False},
    "WAS": {"lat": 38.9076, "lon": -76.8645, "name": "Northwest Stadium", "dome": False},
}


@dataclass
class WeatherConditions:
    """Game weather conditions"""
    team: str
    stadium: str
    game_time: str
    
    # Weather data
    temperature: Optional[float] = None  # Fahrenheit
    feels_like: Optional[float] = None
    wind_speed: Optional[float] = None  # mph
    wind_gust: Optional[float] = None
    precipitation: Optional[float] = None  # inches
    humidity: Optional[int] = None  # %
    description: str = ""
    
    # Dome/Indoor
    is_dome: bool = False
    
    # Impact analysis
    weather_severity: float = 0.0  # 0-1 scale
    total_adjustment: float = 0.0  # Points to adjust total
    recommendation: str = ""  # 'STRONG_UNDER', 'LEAN_UNDER', 'NEUTRAL', etc.
    
    def to_dict(self) -> dict:
        return asdict(self)


class WeatherFetcher:
    """Fetch weather for NFL games"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENWEATHER_API_KEY
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self
    
    async def __aexit__(self, *exc):
        if self.session:
            await self.session.close()
    
    async def fetch_game_weather(
        self,
        home_team: str,
        game_datetime: datetime
    ) -> WeatherConditions:
        """Fetch weather for a specific game"""
        
        stadium = STADIUM_LOCATIONS.get(home_team)
        if not stadium:
            logger.warning(f"Unknown stadium for {home_team}")
            return self._neutral_weather(home_team, game_datetime)
        
        # Check if dome
        if stadium["dome"]:
            logger.info(f"{home_team} plays in dome - no weather impact")
            return WeatherConditions(
                team=home_team,
                stadium=stadium["name"],
                game_time=game_datetime.isoformat(),
                is_dome=True,
                weather_severity=0.0,
                total_adjustment=0.0,
                recommendation="NEUTRAL",
                description="Indoor/Dome",
            )
        
        # Check cache
        cache_key = f"{home_team}_{game_datetime.date()}"
        cache_file = WEATHER_CACHE_DIR / f"weather_{cache_key}.json"
        
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            logger.info(f"Using cached weather for {cache_key}")
            return WeatherConditions(**cached)
        
        if not self.api_key:
            logger.warning("No OpenWeather API key - using neutral weather")
            return self._neutral_weather(home_team, game_datetime)
        
        try:
            weather = await self._fetch_from_openweather(
                stadium["lat"],
                stadium["lon"],
                game_datetime
            )
            
            conditions = WeatherConditions(
                team=home_team,
                stadium=stadium["name"],
                game_time=game_datetime.isoformat(),
                is_dome=False,
                **weather
            )
            
            # Analyze impact
            conditions = self._analyze_weather_impact(conditions)
            
            # Cache it
            cache_file.write_text(json.dumps(conditions.to_dict(), indent=2))
            
            return conditions
        
        except Exception as e:
            logger.error(f"Weather fetch failed for {home_team}: {e}")
            return self._neutral_weather(home_team, game_datetime)
    
    async def _fetch_from_openweather(
        self,
        lat: float,
        lon: float,
        game_datetime: datetime
    ) -> Dict[str, Any]:
        """Fetch from OpenWeather API"""
        
        # Use forecast API (3-hour intervals, 5 days)
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "imperial",
        }
        
        async with self.session.get(url, params=params) as resp:
            if resp.status != 200:
                raise RuntimeError(f"OpenWeather API returned {resp.status}")
            
            data = await resp.json()
        
        # Find closest forecast to game time
        forecasts = data.get("list", [])
        if not forecasts:
            raise RuntimeError("No forecast data returned")
        
        # Find closest time slot
        game_timestamp = game_datetime.timestamp()
        closest = min(
            forecasts,
            key=lambda f: abs(f["dt"] - game_timestamp)
        )
        
        # Extract weather data
        main = closest.get("main", {})
        wind = closest.get("wind", {})
        rain = closest.get("rain", {})
        snow = closest.get("snow", {})
        weather_desc = closest.get("weather", [{}])[0].get("description", "")
        
        # Convert precipitation
        precip = rain.get("3h", 0) + snow.get("3h", 0)  # mm in 3 hours
        precip_inches = precip / 25.4  # Convert to inches
        
        return {
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "wind_speed": wind.get("speed"),
            "wind_gust": wind.get("gust"),
            "precipitation": precip_inches,
            "humidity": main.get("humidity"),
            "description": weather_desc,
        }
    
    def _analyze_weather_impact(self, conditions: WeatherConditions) -> WeatherConditions:
        """Analyze weather impact on totals"""
        
        severity = 0.0
        adjustment = 0.0
        factors = []
        
        # Wind impact (biggest factor for passing)
        if conditions.wind_speed:
            if conditions.wind_speed > 20:
                severity += 0.5
                adjustment -= 5.0
                factors.append(f"EXTREME WIND ({conditions.wind_speed:.0f} mph)")
            elif conditions.wind_speed > 15:
                severity += 0.3
                adjustment -= 3.0
                factors.append(f"High wind ({conditions.wind_speed:.0f} mph)")
            elif conditions.wind_speed > 10:
                severity += 0.15
                adjustment -= 1.5
                factors.append(f"Moderate wind ({conditions.wind_speed:.0f} mph)")
        
        # Precipitation (slippery ball, conservative play)
        if conditions.precipitation:
            if conditions.precipitation > 0.3:
                severity += 0.4
                adjustment -= 4.0
                factors.append(f"HEAVY RAIN ({conditions.precipitation:.1f} in)")
            elif conditions.precipitation > 0.1:
                severity += 0.2
                adjustment -= 2.0
                factors.append(f"Rain ({conditions.precipitation:.1f} in)")
        
        # Temperature extremes
        if conditions.temperature:
            if conditions.temperature < 20:
                severity += 0.25
                adjustment -= 2.5
                factors.append(f"EXTREME COLD ({conditions.temperature:.0f}°F)")
            elif conditions.temperature < 32:
                severity += 0.15
                adjustment -= 1.5
                factors.append(f"Freezing ({conditions.temperature:.0f}°F)")
            elif conditions.temperature > 95:
                severity += 0.1
                adjustment -= 1.0
                factors.append(f"Extreme heat ({conditions.temperature:.0f}°F)")
        
        # Snow
        if "snow" in conditions.description.lower():
            severity += 0.3
            adjustment -= 3.0
            factors.append("SNOW")
        
        # Determine recommendation
        if severity >= 0.6:
            recommendation = "STRONG_UNDER"
        elif severity >= 0.4:
            recommendation = "LEAN_UNDER"
        elif severity >= 0.2:
            recommendation = "SLIGHT_UNDER"
        else:
            recommendation = "NEUTRAL"
        
        conditions.weather_severity = min(1.0, severity)
        conditions.total_adjustment = adjustment
        conditions.recommendation = recommendation
        
        if factors:
            conditions.description = f"{conditions.description} | " + ", ".join(factors)
        
        return conditions
    
    def _neutral_weather(self, team: str, game_datetime: datetime) -> WeatherConditions:
        """Return neutral weather (no impact)"""
        stadium = STADIUM_LOCATIONS.get(team, {})
        return WeatherConditions(
            team=team,
            stadium=stadium.get("name", "Unknown"),
            game_time=game_datetime.isoformat(),
            is_dome=stadium.get("dome", False),
            weather_severity=0.0,
            total_adjustment=0.0,
            recommendation="NEUTRAL",
            description="No data available",
        )


async def fetch_weather_for_games(odds_file: Optional[Path] = None) -> List[WeatherConditions]:
    """Fetch weather for all games in odds file"""
    
    # Find most recent odds file if not specified
    if not odds_file:
        odds_files = sorted(DATA_DIR.glob("nfl_odds_*.json"))
        if not odds_files:
            logger.error("No odds files found")
            return []
        odds_file = odds_files[-1]
    
    logger.info(f"Loading odds from {odds_file}")
    odds_data = json.loads(odds_file.read_text())
    
    # Fetch weather for each game
    weather_conditions = []
    
    async with WeatherFetcher() as fetcher:
        for game_odds in odds_data:
            home_team = game_odds.get("home_team")
            commence_time = game_odds.get("commence_time")
            
            if not home_team or not commence_time:
                continue
            
            try:
                game_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                weather = await fetcher.fetch_game_weather(home_team, game_dt)
                weather_conditions.append(weather)
                
                logger.info(
                    f"🌤️  {game_odds['away_team']} @ {home_team}: "
                    f"{weather.recommendation} ({weather.total_adjustment:+.1f} pts)"
                )
            
            except Exception as e:
                logger.error(f"Failed to fetch weather for {home_team}: {e}")
    
    # Save weather analysis
    output_file = DATA_DIR / f"weather_analysis_{date.today()}.json"
    output_file.write_text(
        json.dumps([w.to_dict() for w in weather_conditions], indent=2)
    )
    logger.info(f"💾 Saved weather analysis to {output_file}")
    
    return weather_conditions


async def main():
    """Main weather integration runner"""
    weather_list = await fetch_weather_for_games()
    
    # Print summary
    print("\n" + "=" * 80)
    print("🌤️  WEATHER IMPACT ANALYSIS")
    print("=" * 80)
    
    significant = [w for w in weather_list if w.weather_severity >= 0.3]
    
    if significant:
        print(f"\n⚠️  {len(significant)} games with significant weather impact:\n")
        
        for weather in sorted(significant, key=lambda w: w.weather_severity, reverse=True):
            print(f"🔴 {weather.team} - {weather.stadium}")
            print(f"   Severity: {weather.weather_severity:.0%}")
            print(f"   Adjustment: {weather.total_adjustment:+.1f} points")
            print(f"   Recommendation: {weather.recommendation}")
            print(f"   Conditions: {weather.description}")
            print()
    else:
        print("\n✅ No significant weather impacts detected")
    
    return weather_list


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
