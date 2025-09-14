"""
Advanced Data Ingestion Layer for Football Betting System
YOLO Mode Implementation - Production-Ready Data Pipeline

Handles ingestion from multiple professional football data sources:
- NFL API (stats.nfl.com)
- ESPN API
- Pro Football Reference
- SportsData.io
- Custom data processing pipeline
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import sqlite3
from pathlib import Path
import requests
from dataclasses import dataclass, asdict
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlayerStats:
    """Professional player statistics data structure"""
    player_id: str
    player_name: str
    team: str
    position: str
    season: int
    week: int
    
    # Traditional stats
    passing_yds: Optional[float] = None
    passing_tds: Optional[float] = None
    rushing_yds: Optional[float] = None
    rushing_tds: Optional[float] = None
    receiving_yds: Optional[float] = None
    receiving_tds: Optional[float] = None
    
    # Advanced metrics (EPA, DVOA, etc.)
    epa_per_play: Optional[float] = None
    success_rate: Optional[float] = None
    dvoa: Optional[float] = None
    dyar: Optional[float] = None
    pfr_grade: Optional[float] = None
    
    # Game situation stats
    red_zone_opportunities: Optional[int] = None
    third_down_conversions: Optional[int] = None
    fourth_down_conversions: Optional[int] = None
    
    # Physical metrics
    age: Optional[float] = None
    experience_years: Optional[int] = None
    height_inches: Optional[int] = None
    weight_lbs: Optional[int] = None

@dataclass
class TeamStats:
    """Professional team statistics data structure"""
    team_id: str
    team_name: str
    season: int
    week: int
    
    # Offensive stats
    total_offensive_yds: Optional[float] = None
    passing_yds: Optional[float] = None
    rushing_yds: Optional[float] = None
    points_scored: Optional[int] = None
    
    # Defensive stats
    points_allowed: Optional[int] = None
    total_defensive_yds: Optional[float] = None
    sacks: Optional[float] = None
    interceptions: Optional[int] = None
    
    # Special teams
    kick_return_yds: Optional[float] = None
    punt_return_yds: Optional[float] = None
    field_goals_made: Optional[int] = None
    
    # Advanced team metrics
    offensive_epa: Optional[float] = None
    defensive_epa: Optional[float] = None
    team_dvoa: Optional[float] = None
    
    # Situational stats
    red_zone_efficiency: Optional[float] = None
    third_down_conversion_pct: Optional[float] = None
    turnover_differential: Optional[int] = None

@dataclass
class GameData:
    """Complete game data structure"""
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    game_date: Optional[datetime] = None
    venue: Optional[str] = None
    weather_temp: Optional[float] = None
    weather_conditions: Optional[str] = None
    spread_line: Optional[float] = None
    total_line: Optional[float] = None

class AdvancedDataIngestion:
    """
    Professional-grade data ingestion system for football analytics.
    YOLO Mode: Comprehensive, scalable, production-ready.
    """
    
    def __init__(self, cache_dir: str = "data_cache", db_path: str = "football_stats.db"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = db_path
        self.session = None
        self._init_database()
        
        # API configurations
        self.api_configs = {
            "nfl_api": {
                "base_url": "https://api.nfl.com/v1",
                "headers": {"User-Agent": "Mozilla/5.0"}
            },
            "espn_api": {
                "base_url": "https://site.api.espn.com/apis/site/v2/sports/football/nfl",
                "headers": {"User-Agent": "Mozilla/5.0"}
            },
            "sportsdata": {
                "base_url": "https://api.sportsdata.io/v2/json",
                "api_key": None  # Set via environment
            },
            "pro_football_ref": {
                "base_url": "https://www.pro-football-reference.com",
                "headers": {"User-Agent": "Mozilla/5.0"}
            }
        }
        
        # Rate limiting
        self.request_delays = {
            "nfl_api": 1.0,
            "espn_api": 0.5,
            "sportsdata": 1.5,
            "pro_football_ref": 2.0
        }
        
        self.last_requests = {}
    
    def _init_database(self):
        """Initialize SQLite database with proper schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_stats (
                player_id TEXT,
                player_name TEXT,
                team TEXT,
                position TEXT,
                season INTEGER,
                week INTEGER,
                passing_yds REAL,
                passing_tds REAL,
                rushing_yds REAL,
                rushing_tds REAL,
                receiving_yds REAL,
                receiving_tds REAL,
                epa_per_play REAL,
                success_rate REAL,
                dvoa REAL,
                dyar REAL,
                pfr_grade REAL,
                red_zone_opportunities INTEGER,
                third_down_conversions INTEGER,
                fourth_down_conversions INTEGER,
                age REAL,
                experience_years INTEGER,
                height_inches INTEGER,
                weight_lbs INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, season, week)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_stats (
                team_id TEXT,
                team_name TEXT,
                season INTEGER,
                week INTEGER,
                total_offensive_yds REAL,
                passing_yds REAL,
                rushing_yds REAL,
                points_scored INTEGER,
                points_allowed INTEGER,
                total_defensive_yds REAL,
                sacks REAL,
                interceptions INTEGER,
                kick_return_yds REAL,
                punt_return_yds REAL,
                field_goals_made INTEGER,
                offensive_epa REAL,
                defensive_epa REAL,
                team_dvoa REAL,
                red_zone_efficiency REAL,
                third_down_conversion_pct REAL,
                turnover_differential INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, season, week)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                season INTEGER,
                week INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                game_date TIMESTAMP,
                venue TEXT,
                weather_temp REAL,
                weather_conditions TEXT,
                spread_line REAL,
                total_line REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_season_week ON player_stats(season, week)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_team_season_week ON team_stats(season, week)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week)")
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    async def _make_request(self, url: str, api_name: str, headers: Optional[Dict] = None) -> Optional[Dict]:
        """Make rate-limited API request with caching"""
        # Rate limiting
        now = time.time()
        if api_name in self.last_requests:
            elapsed = now - self.last_requests[api_name]
            delay = self.request_delays.get(api_name, 1.0)
            if elapsed < delay:
                await asyncio.sleep(delay - elapsed)
        
        self.last_requests[api_name] = time.time()
        
        # Check cache first
        cache_key = f"{api_name}_{hash(url)}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 3600:  # 1 hour cache
                try:
                    with open(cache_file, "r") as f:
                        return json.load(f)
                except:
                    pass
        
        # Make request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Cache successful responses
                        with open(cache_file, "w") as f:
                            json.dump(data, f)
                        return data
                    else:
                        logger.warning(f"API {api_name} returned status {response.status} for {url}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching from {api_name}: {e}")
            return None
    
    async def fetch_nfl_player_stats(self, season: int, week: int) -> List[PlayerStats]:
        """Fetch comprehensive player stats from NFL API"""
        logger.info(f"Fetching NFL player stats for {season} Week {week}")
        
        # NFL API endpoint for player stats
        url = f"{self.api_configs[