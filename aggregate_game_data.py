#!/usr/bin/env python3
"""
Aggregate all game result files into unified dataset for analysis.
Deduplicates games, normalizes format, and produces comprehensive historical file.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/ryan/code/football_betting_system")
DATA_DIRS = [
    PROJECT_ROOT / "data/referee_conspiracy",
    PROJECT_ROOT / "data/football/nfl",
]
OUTPUT_DIR = PROJECT_ROOT / "data/historical_games"
OUTPUT_FILE = OUTPUT_DIR / "nfl_games_all_aggregated.json"


def find_all_game_files() -> List[Path]:
    """Recursively find all game result JSON files."""
    files = []
    for data_dir in DATA_DIRS:
        if data_dir.exists():
            # Find game_results*.json and production_results*.json
            files.extend(data_dir.glob("game_results*.json"))
            files.extend(data_dir.glob("production_results*.json"))
    return sorted(set(files))  # Deduplicate


def load_games_from_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load games from a single file, handling various formats."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        
        # Handle dict format (keyed by game_id)
        if isinstance(data, dict):
            # Filter out non-dict values
            games = [v for v in data.values() if isinstance(v, dict)]
        # Handle list format
        elif isinstance(data, list):
            games = [g for g in data if isinstance(g, dict)]
        else:
            logger.warning(f"Unexpected format in {filepath}")
            return []
        
        return games
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return []


def normalize_game(game: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize game record to standard format."""
    return {
        "game_id": game.get("game_id", ""),
        "date": game.get("date", ""),
        "home_team": game.get("home_team", ""),
        "away_team": game.get("away_team", ""),
        "home_score": int(game.get("home_score", 0)) if game.get("home_score") else 0,
        "away_score": int(game.get("away_score", 0)) if game.get("away_score") else 0,
        "total_score": int(game.get("total_score", 0)) if game.get("total_score") else 0,
        "winner": game.get("winner", ""),
        "referee": game.get("referee", "Unknown"),
        # Pregame lines (if available)
        "spread": game.get("spread"),
        "total": game.get("total"),
        "home_ml_odds": game.get("home_ml_odds"),
        "away_ml_odds": game.get("away_ml_odds"),
        "spread_odds": game.get("spread_odds"),
        "total_odds": game.get("total_odds"),
        "spread_model_home_pct": game.get("spread_model_home_pct"),
        "total_model_over_pct": game.get("total_model_over_pct"),
    }


def deduplicate_games(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate games (by game_id + date)."""
    seen: Set[str] = set()
    unique = []
    duplicates = 0
    
    for game in games:
        key = f"{game.get('game_id', '')}:{game.get('date', '')}"
        if not key or key == ":":
            continue
        
        if key in seen:
            duplicates += 1
            continue
        
        seen.add(key)
        unique.append(game)
    
    if duplicates > 0:
        logger.info(f"Removed {duplicates} duplicate games")
    
    return unique


def aggregate_all_games() -> Dict[str, Any]:
    """Aggregate all games from all sources."""
    logger.info("="*80)
    logger.info("AGGREGATING ALL GAME DATA")
    logger.info("="*80)
    
    files = find_all_game_files()
    logger.info(f"Found {len(files)} game files to process")
    
    all_games = []
    file_stats = defaultdict(int)
    
    for filepath in files:
        games = load_games_from_file(filepath)
        normalized = [normalize_game(g) for g in games if g.get("game_id")]
        all_games.extend(normalized)
        file_stats[filepath.name] = len(normalized)
        logger.info(f"  {filepath.name}: {len(normalized)} games")
    
    logger.info(f"\nTotal games before deduplication: {len(all_games)}")
    
    # Deduplicate
    unique_games = deduplicate_games(all_games)
    logger.info(f"Total games after deduplication: {len(unique_games)}")
    
    # Sort by date
    unique_games.sort(key=lambda g: (g.get("date", ""), g.get("game_id", "")))
    
    # Analyze
    seasons = defaultdict(int)
    teams = defaultdict(int)
    referees = defaultdict(int)
    games_with_lines = 0
    
    for game in unique_games:
        # Extract season from date (YYYY-MM-DD format)
        if game.get("date"):
            season = game["date"].split("-")[0]
            seasons[season] += 1
        
        teams[game.get("home_team", "UNKNOWN")] += 1
        teams[game.get("away_team", "UNKNOWN")] += 1
        referees[game.get("referee", "Unknown")] += 1
        
        if game.get("spread") is not None and game.get("total") is not None:
            games_with_lines += 1
    
    report = {
        "total_games": len(unique_games),
        "games_with_pregame_lines": games_with_lines,
        "unique_seasons": len(seasons),
        "unique_teams": len(teams),
        "unique_referees": len(referees),
        "seasons": dict(seasons),
        "top_referees": sorted(referees.items(), key=lambda x: x[1], reverse=True)[:10],
    }
    
    logger.info("\n" + "="*80)
    logger.info("AGGREGATION REPORT")
    logger.info("="*80)
    logger.info(f"Total games: {report['total_games']}")
    logger.info(f"Games with pregame lines: {report['games_with_pregame_lines']}")
    logger.info(f"Unique seasons: {report['unique_seasons']}")
    logger.info(f"Unique teams: {report['unique_teams']}")
    logger.info(f"Unique referees: {report['unique_referees']}")
    logger.info(f"\nGames by season:")
    for season in sorted(report["seasons"].keys()):
        logger.info(f"  {season}: {report['seasons'][season]} games")
    logger.info(f"\nTop 10 referees:")
    for ref, count in report["top_referees"]:
        logger.info(f"  {ref}: {count} games")
    
    return {
        "games": unique_games,
        "report": report,
    }


def save_aggregated_data(data: Dict[str, Any]) -> None:
    """Save aggregated games to file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save games
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data["games"], f, indent=2)
    logger.info(f"\nSaved {len(data['games'])} games to {OUTPUT_FILE}")
    
    # Save report
    report_file = OUTPUT_DIR / "aggregation_report.json"
    with open(report_file, "w") as f:
        json.dump(data["report"], f, indent=2)
    logger.info(f"Saved report to {report_file}")


if __name__ == "__main__":
    data = aggregate_all_games()
    save_aggregated_data(data)
    logger.info("\n✅ Aggregation complete!")
