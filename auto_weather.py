#!/usr/bin/env python3
"""
Auto Weather Fetcher - Scrapes weather data for game-time conditions
Uses Crawlbase MCP for reliable scraping
"""

import os
import sys
import json
import re
import subprocess
from datetime import datetime

# NFL Stadium locations (city, state for weather lookup)
STADIUM_LOCATIONS = {
    'Cardinals': 'Phoenix, AZ',
    'Falcons': 'Atlanta, GA',
    'Ravens': 'Baltimore, MD',
    'Bills': 'Buffalo, NY',
    'Panthers': 'Charlotte, NC',
    'Bears': 'Chicago, IL',
    'Bengals': 'Cincinnati, OH',
    'Browns': 'Cleveland, OH',
    'Cowboys': 'Dallas, TX',
    'Broncos': 'Denver, CO',
    'Lions': 'Detroit, MI',
    'Packers': 'Green Bay, WI',
    'Texans': 'Houston, TX',
    'Colts': 'Indianapolis, IN',
    'Jaguars': 'Jacksonville, FL',
    'Chiefs': 'Kansas City, MO',
    'Raiders': 'Las Vegas, NV',
    'Chargers': 'Los Angeles, CA',
    'Rams': 'Los Angeles, CA',
    'Dolphins': 'Miami, FL',
    'Vikings': 'Minneapolis, MN',
    'Patriots': 'Foxborough, MA',
    'Saints': 'New Orleans, LA',
    'Giants': 'East Rutherford, NJ',
    'Jets': 'East Rutherford, NJ',
    'Eagles': 'Philadelphia, PA',
    'Steelers': 'Pittsburgh, PA',
    '49ers': 'Santa Clara, CA',
    'Seahawks': 'Seattle, WA',
    'Buccaneers': 'Tampa, FL',
    'Titans': 'Nashville, TN',
    'Commanders': 'Landover, MD'
}

# Dome stadiums (no weather impact)
DOME_STADIUMS = [
    'Cardinals', 'Falcons', 'Cowboys', 'Texans', 'Colts', 
    'Lions', 'Saints', 'Vikings', 'Raiders', 'Rams', 'Chargers'
]


def fetch_game_weather(home_team, game_time=None):
    """
    Fetch weather conditions for game location.
    
    Args:
        home_team: "Packers", "Eagles", etc.
        game_time: Optional datetime for game (default: now)
    
    Returns:
        dict: {
            'temp': 42,
            'wind_speed': 15,
            'wind_direction': 'NW',
            'precipitation': 0.0,
            'conditions': 'Cloudy',
            'feels_like': 35,
            'humidity': 65,
            'is_dome': False,
            'weather_impact': 'MODERATE',
            'timestamp': '2025-01-14T14:05:00Z'
        }
    """
    
    # Check if dome stadium
    if home_team in DOME_STADIUMS:
        print(f"🏟️  {home_team} play in a dome - no weather impact")
        return {
            'temp': 72,
            'wind_speed': 0,
            'wind_direction': 'N/A',
            'precipitation': 0.0,
            'conditions': 'Dome',
            'feels_like': 72,
            'humidity': 50,
            'is_dome': True,
            'weather_impact': 'NONE',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    
    # Get stadium location
    if home_team not in STADIUM_LOCATIONS:
        print(f"❌ Unknown team: {home_team}")
        return None
    
    location = STADIUM_LOCATIONS[home_team]
    
    print(f"🌤️  Fetching weather for {home_team} ({location})...")
    
    # Scrape weather data
    weather_data = scrape_weather(location)
    
    if weather_data:
        # Calculate weather impact
        weather_data['weather_impact'] = calculate_weather_impact(weather_data)
        weather_data['is_dome'] = False
        
        print(f"✅ Weather: {weather_data['temp']}°F, {weather_data['wind_speed']} mph wind")
        print(f"   Impact: {weather_data['weather_impact']}")
        
        return weather_data
    else:
        print(f"❌ Failed to fetch weather for {location}")
        return None


def scrape_weather(location):
    """
    Scrape weather data from Weather.com using Crawlbase MCP.
    
    Returns:
        dict: Weather conditions
    """
    
    # Format location for URL
    location_slug = location.replace(' ', '-').replace(',', '').lower()
    url = f"https://weather.com/weather/today/l/{location_slug}"
    
    print(f"📍 URL: {url}")
    
    try:
        result = subprocess.run(
            ['npx', '@crawlbase/mcp@latest', 'crawl_markdown', url],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return None
        
        markdown = result.stdout
        
        # Parse markdown for weather data
        weather = parse_weather_from_markdown(markdown)
        
        return weather
        
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def parse_weather_from_markdown(markdown):
    """
    Parse weather markdown for conditions.
    
    Weather.com shows:
    - Temperature: "42°F"
    - Wind: "15 mph NW"
    - Conditions: "Cloudy", "Rain", etc.
    - Precipitation: "0%"
    """
    
    weather = {
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    # Temperature
    temp_match = re.search(r'(\d+)°F', markdown)
    if temp_match:
        weather['temp'] = int(temp_match.group(1))
    
    # Wind speed and direction
    wind_match = re.search(r'(\d+)\s*mph\s*([NSEW]{1,2})', markdown, re.IGNORECASE)
    if wind_match:
        weather['wind_speed'] = int(wind_match.group(1))
        weather['wind_direction'] = wind_match.group(2).upper()
    else:
        weather['wind_speed'] = 0
        weather['wind_direction'] = 'N/A'
    
    # Conditions
    conditions = ['Sunny', 'Cloudy', 'Partly Cloudy', 'Rain', 'Snow', 'Clear']
    for condition in conditions:
        if condition.lower() in markdown.lower():
            weather['conditions'] = condition
            break
    if 'conditions' not in weather:
        weather['conditions'] = 'Unknown'
    
    # Precipitation
    precip_match = re.search(r'precipitation.*?(\d+)%', markdown, re.IGNORECASE)
    if precip_match:
        weather['precipitation'] = float(precip_match.group(1)) / 100
    else:
        weather['precipitation'] = 0.0
    
    # Feels like
    feels_match = re.search(r'feels like.*?(\d+)°', markdown, re.IGNORECASE)
    if feels_match:
        weather['feels_like'] = int(feels_match.group(1))
    else:
        weather['feels_like'] = weather.get('temp', 50)
    
    # Humidity
    humidity_match = re.search(r'humidity.*?(\d+)%', markdown, re.IGNORECASE)
    if humidity_match:
        weather['humidity'] = int(humidity_match.group(1))
    else:
        weather['humidity'] = 50
    
    return weather if 'temp' in weather else None


def calculate_weather_impact(weather):
    """
    Calculate weather impact on game (NONE, LOW, MODERATE, HIGH, EXTREME).
    
    Factors:
    - Temperature: <20°F or >90°F = HIGH
    - Wind: >15 mph = MODERATE, >25 mph = HIGH
    - Precipitation: >30% = MODERATE, >60% = HIGH
    """
    
    impact_score = 0
    
    # Temperature impact
    temp = weather.get('temp', 70)
    if temp < 20 or temp > 90:
        impact_score += 3
    elif temp < 32 or temp > 85:
        impact_score += 2
    elif temp < 40 or temp > 80:
        impact_score += 1
    
    # Wind impact (most important for passing)
    wind = weather.get('wind_speed', 0)
    if wind > 25:
        impact_score += 3
    elif wind > 15:
        impact_score += 2
    elif wind > 10:
        impact_score += 1
    
    # Precipitation impact
    precip = weather.get('precipitation', 0)
    if precip > 0.6:
        impact_score += 3
    elif precip > 0.3:
        impact_score += 2
    elif precip > 0.1:
        impact_score += 1
    
    # Map score to impact level
    if impact_score == 0:
        return 'NONE'
    elif impact_score <= 2:
        return 'LOW'
    elif impact_score <= 4:
        return 'MODERATE'
    elif impact_score <= 6:
        return 'HIGH'
    else:
        return 'EXTREME'


def save_weather_data(game_id, weather_data):
    """Save weather data to file."""
    
    data_dir = 'data/weather'
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = f"{data_dir}/{game_id}.json"
    
    with open(filepath, 'w') as f:
        json.dump(weather_data, f, indent=2)
    
    print(f"💾 Saved weather data to {filepath}")


def get_weather_betting_adjustment(weather):
    """
    Get betting adjustment based on weather.
    
    Returns:
        dict: {
            'favor_under': True/False,
            'favor_rushing': True/False,
            'confidence_adjustment': -0.1 to 0.1
        }
    """
    
    impact = weather['weather_impact']
    wind = weather.get('wind_speed', 0)
    
    adjustment = {
        'favor_under': False,
        'favor_rushing': False,
        'confidence_adjustment': 0.0
    }
    
    # High wind = favor UNDER and rushing
    if wind > 15:
        adjustment['favor_under'] = True
        adjustment['favor_rushing'] = True
        adjustment['confidence_adjustment'] = 0.05  # +5% for under/rushing bets
    
    # Extreme weather = lower confidence
    if impact in ['HIGH', 'EXTREME']:
        adjustment['confidence_adjustment'] = -0.1  # -10% confidence
    
    return adjustment


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 auto_weather.py <home_team>")
        print("Example: python3 auto_weather.py Packers")
        sys.exit(1)
    
    home_team = sys.argv[1]
    
    # Fetch weather
    weather = fetch_game_weather(home_team)
    
    if weather:
        game_id = f"{home_team}_{datetime.now().strftime('%Y%m%d')}"
        save_weather_data(game_id, weather)
        
        # Get betting adjustment
        adjustment = get_weather_betting_adjustment(weather)
        
        print("\n🌤️  WEATHER DATA:")
        print(json.dumps(weather, indent=2))
        
        print("\n📊 BETTING ADJUSTMENT:")
        print(json.dumps(adjustment, indent=2))
    else:
        print("\n❌ Failed to fetch weather")
        sys.exit(1)
