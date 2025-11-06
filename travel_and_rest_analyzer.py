#!/usr/bin/env python3
"""
Travel and Rest Analyzer for MLB Betting System
===============================================
Analyzes team travel patterns, fatigue, and pitcher rest days to enhance predictions.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
import httpx
import requests

logger = logging.getLogger(__name__)


class TravelAndRestAnalyzer:
    """
    Analyzes travel patterns, fatigue, and pitcher rest days for MLB teams.
    """

    def __init__(self, db_path: str = "data/travel_rest_analysis.db"):
        self.db_path = db_path
        self._init_database()

        # Travel fatigue factors
        self.travel_fatigue_weights = {
            "same_timezone": 0.0,  # No fatigue
            "1_timezone": 0.1,  # Minimal fatigue
            "2_timezones": 0.25,  # Moderate fatigue
            "3_timezones": 0.4,  # Significant fatigue
            "cross_country": 0.6,  # High fatigue (West to East or vice versa)
        }

        # Rest day impact factors
        self.pitcher_rest_impact = {
            1: -0.4,  # 1 day rest - major negative impact
            2: -0.25,  # 2 days rest - significant negative
            3: -0.1,  # 3 days rest - slight negative
            4: 0.0,  # 4 days rest - neutral
            5: 0.1,  # 5 days rest - slight positive
            6: 0.2,  # 6 days rest - good rest
            7: 0.3,  # 7+ days rest - excellent rest
        }

        # Team timezone mapping
        self.team_timezones = {
            # Pacific Time (PT)
            "Seattle Mariners": "PT",
            "Oakland Athletics": "PT",
            "San Francisco Giants": "PT",
            "Los Angeles Dodgers": "PT",
            "Los Angeles Angels": "PT",
            "San Diego Padres": "PT",
            # Mountain Time (MT)
            "Colorado Rockies": "MT",
            "Arizona Diamondbacks": "MT",
            # Central Time (CT)
            "Texas Rangers": "CT",
            "Houston Astros": "CT",
            "Kansas City Royals": "CT",
            "St. Louis Cardinals": "CT",
            "Chicago Cubs": "CT",
            "Chicago White Sox": "CT",
            "Milwaukee Brewers": "CT",
            "Minnesota Twins": "CT",
            "Detroit Tigers": "CT",
            "Cleveland Guardians": "CT",
            # Eastern Time (ET)
            "New York Yankees": "ET",
            "New York Mets": "ET",
            "Boston Red Sox": "ET",
            "Toronto Blue Jays": "ET",
            "Baltimore Orioles": "ET",
            "Tampa Bay Rays": "ET",
            "Miami Marlins": "ET",
            "Atlanta Braves": "ET",
            "Washington Nationals": "ET",
            "Philadelphia Phillies": "ET",
            "Pittsburgh Pirates": "ET",
            "Cincinnati Reds": "ET",
        }

        # Timezone order for calculating differences
        self.timezone_order = ["PT", "MT", "CT", "ET"]

        logger.info("✈️ Travel and Rest Analyzer initialized")

    def _init_database(self):
        """Initialize the travel and rest analysis database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Team travel history
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS team_travel_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT NOT NULL,
                from_city TEXT NOT NULL,
                to_city TEXT NOT NULL,
                from_timezone TEXT NOT NULL,
                to_timezone TEXT NOT NULL,
                travel_date TEXT NOT NULL,
                game_date TEXT NOT NULL,
                travel_distance_miles REAL,
                timezone_difference INTEGER,
                fatigue_factor REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Pitcher rest tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pitcher_rest_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pitcher_name TEXT NOT NULL,
                team_name TEXT NOT NULL,
                last_start_date TEXT NOT NULL,
                next_start_date TEXT,
                rest_days INTEGER,
                rest_impact_factor REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Game analysis results
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS game_travel_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                game_date TEXT NOT NULL,
                home_travel_fatigue REAL,
                away_travel_fatigue REAL,
                home_pitcher_rest_factor REAL,
                away_pitcher_rest_factor REAL,
                overall_travel_impact REAL,
                analysis_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_team_travel_team_date ON team_travel_history(team_name, travel_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pitcher_rest_pitcher ON pitcher_rest_tracking(pitcher_name, last_start_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_game_analysis_game ON game_travel_analysis(game_id)"
        )

        conn.commit()
        conn.close()

    def calculate_timezone_difference(
        self, from_timezone: str, to_timezone: str
    ) -> int:
        """Calculate timezone difference in hours."""
        try:
            from_idx = self.timezone_order.index(from_timezone)
            to_idx = self.timezone_order.index(to_timezone)
            return abs(to_idx - from_idx)
        except ValueError:
            return 0

    def get_travel_fatigue_factor(self, from_timezone: str, to_timezone: str) -> float:
        """Calculate travel fatigue factor based on timezone difference."""
        if from_timezone == to_timezone:
            return self.travel_fatigue_weights["same_timezone"]

        timezone_diff = self.calculate_timezone_difference(from_timezone, to_timezone)

        if timezone_diff == 1:
            return self.travel_fatigue_weights["1_timezone"]
        elif timezone_diff == 2:
            return self.travel_fatigue_weights["2_timezones"]
        elif timezone_diff == 3:
            return self.travel_fatigue_weights["3_timezones"]
        else:
            return self.travel_fatigue_weights["cross_country"]

    def get_pitcher_rest_impact(self, rest_days: int) -> float:
        """Get pitcher rest impact factor."""
        if rest_days <= 0:
            return self.pitcher_rest_impact[1]  # Treat as 1 day rest
        elif rest_days >= 7:
            return self.pitcher_rest_impact[7]  # Cap at 7 days
        else:
            return self.pitcher_rest_impact.get(rest_days, 0.0)

    async def record_team_travel(
        self,
        team_name: str,
        from_city: str,
        to_city: str,
        travel_date: str,
        game_date: str,
        travel_distance: float = 0,
    ):
        """Record team travel for analysis."""
        from_timezone = self.get_team_timezone(team_name)
        to_timezone = self.get_team_timezone(
            team_name
        )  # This would be updated based on destination

        # For now, we'll use the destination city to determine timezone
        # In a real implementation, you'd have a city-to-timezone mapping
        to_timezone = self.estimate_destination_timezone(to_city)

        fatigue_factor = self.get_travel_fatigue_factor(from_timezone, to_timezone)
        timezone_diff = self.calculate_timezone_difference(from_timezone, to_timezone)

        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT INTO team_travel_history
                (team_name, from_city, to_city, from_timezone, to_timezone,
                 travel_date, game_date, travel_distance_miles, timezone_difference, fatigue_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    team_name,
                    from_city,
                    to_city,
                    from_timezone,
                    to_timezone,
                    travel_date,
                    game_date,
                    travel_distance,
                    timezone_diff,
                    fatigue_factor,
                ),
            )
            await conn.commit()

        logger.info(
            f"✈️ Recorded travel: {team_name} {from_city} → {to_city} (fatigue: {fatigue_factor:.2f})"
        )

    async def record_pitcher_start(
        self, pitcher_name: str, team_name: str, start_date: str
    ):
        """Record a pitcher's start to track rest days."""
        async with aiosqlite.connect(self.db_path) as conn:
            # Update previous record with next start date
            await conn.execute(
                """
                UPDATE pitcher_rest_tracking
                SET next_start_date = ?
                WHERE pitcher_name = ? AND next_start_date IS NULL
            """,
                (start_date, pitcher_name),
            )

            # Add new record
            await conn.execute(
                """
                INSERT INTO pitcher_rest_tracking
                (pitcher_name, team_name, last_start_date, rest_days, rest_impact_factor)
                VALUES (?, ?, ?, 0, ?)
            """,
                (pitcher_name, team_name, start_date, self.pitcher_rest_impact[1]),
            )

            await conn.commit()

        logger.info(f"⚾ Recorded pitcher start: {pitcher_name} on {start_date}")

    async def calculate_team_travel_fatigue(
        self, team_name: str, game_date: str
    ) -> float:
        """Calculate team travel fatigue for a specific game."""
        try:
            # Look for recent travel within 3 days of the game
            three_days_before = (
                datetime.strptime(game_date, "%Y-%m-%d") - timedelta(days=3)
            ).strftime("%Y-%m-%d")

            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    """
                    SELECT fatigue_factor, travel_date
                    FROM team_travel_history
                    WHERE team_name = ? AND travel_date >= ?
                    ORDER BY travel_date DESC
                    LIMIT 1
                """,
                    (team_name, three_days_before),
                )

                result = await cursor.fetchone()

                if result:
                    fatigue_factor, travel_date = result

                    # Calculate days since travel
                    travel_dt = datetime.strptime(travel_date, "%Y-%m-%d")
                    game_dt = datetime.strptime(game_date, "%Y-%m-%d")
                    days_since_travel = (game_dt - travel_dt).days

                    # Fatigue decreases over time
                    if days_since_travel == 0:
                        return fatigue_factor  # Same day travel
                    elif days_since_travel == 1:
                        return fatigue_factor * 0.7  # Next day
                    elif days_since_travel == 2:
                        return fatigue_factor * 0.4  # Two days later
                    else:
                        return fatigue_factor * 0.1  # Three+ days later

                return 0.0  # No recent travel

        except Exception as e:
            logger.error(f"Error calculating travel fatigue: {e}")
            return 0.0

    async def calculate_pitcher_rest_factor(
        self, pitcher_name: str, game_date: str
    ) -> float:
        """Calculate pitcher rest factor for a specific game."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    """
                    SELECT last_start_date
                    FROM pitcher_rest_tracking
                    WHERE pitcher_name = ? AND last_start_date <= ?
                    ORDER BY last_start_date DESC
                    LIMIT 1
                """,
                    (pitcher_name, game_date),
                )

                result = await cursor.fetchone()

                if result:
                    last_start_date = result[0]
                    last_start_dt = datetime.strptime(last_start_date, "%Y-%m-%d")
                    game_dt = datetime.strptime(game_date, "%Y-%m-%d")
                    rest_days = (game_dt - last_start_dt).days

                    return self.get_pitcher_rest_impact(rest_days)

                return 0.0  # No previous start recorded

        except Exception as e:
            logger.error(f"Error calculating pitcher rest: {e}")
            return 0.0

    async def analyze_game_travel_rest(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        game_date: str,
        home_pitcher: str = None,
        away_pitcher: str = None,
    ) -> dict[str, Any]:
        """Analyze travel and rest factors for a specific game."""
        try:
            # Calculate travel fatigue for both teams
            home_travel_fatigue = await self.calculate_team_travel_fatigue(
                home_team, game_date
            )
            away_travel_fatigue = await self.calculate_team_travel_fatigue(
                away_team, game_date
            )

            # Calculate pitcher rest factors
            home_pitcher_rest = 0.0
            away_pitcher_rest = 0.0

            if home_pitcher:
                home_pitcher_rest = await self.calculate_pitcher_rest_factor(
                    home_pitcher, game_date
                )

            if away_pitcher:
                away_pitcher_rest = await self.calculate_pitcher_rest_factor(
                    away_pitcher, game_date
                )

            # Calculate overall impact
            # Positive values favor home team, negative values favor away team
            overall_impact = (home_pitcher_rest - away_pitcher_rest) + (
                away_travel_fatigue - home_travel_fatigue
            )

            # Generate analysis notes
            notes = []
            if home_travel_fatigue > 0:
                notes.append(f"Home team travel fatigue: {home_travel_fatigue:.2f}")
            if away_travel_fatigue > 0:
                notes.append(f"Away team travel fatigue: {away_travel_fatigue:.2f}")
            if home_pitcher_rest != 0:
                notes.append(f"Home pitcher rest factor: {home_pitcher_rest:+.2f}")
            if away_pitcher_rest != 0:
                notes.append(f"Away pitcher rest factor: {away_pitcher_rest:+.2f}")

            analysis = {
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "game_date": game_date,
                "home_travel_fatigue": home_travel_fatigue,
                "away_travel_fatigue": away_travel_fatigue,
                "home_pitcher_rest_factor": home_pitcher_rest,
                "away_pitcher_rest_factor": away_pitcher_rest,
                "overall_travel_rest_impact": overall_impact,
                "analysis_notes": (
                    "; ".join(notes) if notes else "No significant travel/rest factors"
                ),
                "recommendation": self.get_travel_rest_recommendation(overall_impact),
            }

            # Save analysis to database
            await self.save_game_analysis(analysis)

            logger.info(
                f"📊 Game analysis: {home_team} vs {away_team} - Impact: {overall_impact:+.3f}"
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing game travel/rest: {e}")
            return {
                "game_id": game_id,
                "error": str(e),
                "overall_travel_rest_impact": 0.0,
            }

    def get_travel_rest_recommendation(self, impact: float) -> str:
        """Get recommendation based on travel/rest impact."""
        if impact > 0.3:
            return "Strong advantage to home team"
        elif impact > 0.1:
            return "Moderate advantage to home team"
        elif impact > -0.1:
            return "Slight advantage to home team"
        elif impact > -0.3:
            return "Slight advantage to away team"
        else:
            return "Strong advantage to away team"

    async def save_game_analysis(self, analysis: dict[str, Any]):
        """Save game analysis to database."""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO game_travel_analysis
                (game_id, home_team, away_team, game_date, home_travel_fatigue,
                 away_travel_fatigue, home_pitcher_rest_factor, away_pitcher_rest_factor,
                 overall_travel_impact, analysis_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    analysis["game_id"],
                    analysis["home_team"],
                    analysis["away_team"],
                    analysis["game_date"],
                    analysis["home_travel_fatigue"],
                    analysis["away_travel_fatigue"],
                    analysis["home_pitcher_rest_factor"],
                    analysis["away_pitcher_rest_factor"],
                    analysis["overall_travel_rest_impact"],
                    analysis["analysis_notes"],
                ),
            )
            await conn.commit()

    def get_team_timezone(self, team_name: str) -> str:
        """Get timezone for a team."""
        return self.team_timezones.get(team_name, "ET")  # Default to Eastern

    def estimate_destination_timezone(self, city: str) -> str:
        """Estimate timezone based on city name."""
        city_lower = city.lower()

        if any(
            west in city_lower
            for west in [
                "seattle",
                "portland",
                "san francisco",
                "los angeles",
                "san diego",
                "oakland",
            ]
        ):
            return "PT"
        elif any(
            mountain in city_lower for mountain in ["denver", "phoenix", "salt lake"]
        ):
            return "MT"
        elif any(
            central in city_lower
            for central in [
                "chicago",
                "houston",
                "dallas",
                "kansas city",
                "milwaukee",
                "minneapolis",
            ]
        ):
            return "CT"
        else:
            return "ET"  # Default to Eastern

    async def get_travel_rest_insights(self) -> dict[str, Any]:
        """Get insights from travel and rest analysis."""
        async with aiosqlite.connect(self.db_path) as conn:
            # Get recent analysis
            cursor = await conn.execute(
                """
                SELECT COUNT(*) as total_games,
                       AVG(overall_travel_impact) as avg_impact,
                       COUNT(CASE WHEN overall_travel_impact > 0.2 THEN 1 END) as home_advantages,
                       COUNT(CASE WHEN overall_travel_impact < -0.2 THEN 1 END) as away_advantages
                FROM game_travel_analysis
                WHERE game_date >= date('now', '-30 days')
            """
            )

            result = await cursor.fetchone()

            if result:
                total_games, avg_impact, home_advantages, away_advantages = result

                return {
                    "total_games_analyzed": total_games or 0,
                    "average_travel_rest_impact": avg_impact or 0.0,
                    "home_team_advantages": home_advantages or 0,
                    "away_team_advantages": away_advantages or 0,
                    "neutral_games": (total_games or 0)
                    - (home_advantages or 0)
                    - (away_advantages or 0),
                    "home_advantage_rate": (home_advantages or 0)
                    / max(total_games or 1, 1),
                    "away_advantage_rate": (away_advantages or 0)
                    / max(total_games or 1, 1),
                }

            return {"error": "No analysis data available"}


# Integration with learning system
class TravelRestLearningIntegration:
    """Integrates travel and rest analysis with the learning system."""

    def __init__(self, travel_analyzer: TravelAndRestAnalyzer):
        self.travel_analyzer = travel_analyzer

    async def enhance_prediction_with_travel_rest(
        self, prediction: dict[str, Any], game_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Enhance a prediction with travel and rest analysis."""
        try:
            # Analyze travel and rest for this game
            analysis = await self.travel_analyzer.analyze_game_travel_rest(
                game_id=game_data.get("game_id", "unknown"),
                home_team=game_data.get("home_team", ""),
                away_team=game_data.get("away_team", ""),
                game_date=game_data.get(
                    "game_date", datetime.now().strftime("%Y-%m-%d")
                ),
                home_pitcher=game_data.get("home_pitcher"),
                away_pitcher=game_data.get("away_pitcher"),
            )

            # Calculate confidence adjustment based on travel/rest impact
            travel_rest_impact = analysis.get("overall_travel_rest_impact", 0.0)

            # Adjust confidence based on whether the prediction aligns with travel/rest factors
            predicted_winner = prediction.get("predicted_winner", "")
            home_team = game_data.get("home_team", "")

            confidence_adjustment = 0.0
            if predicted_winner == home_team and travel_rest_impact > 0:
                # Prediction favors home team and travel/rest factors support home team
                confidence_adjustment = min(
                    travel_rest_impact * 0.1, 0.05
                )  # Max 5% boost
            elif predicted_winner != home_team and travel_rest_impact < 0:
                # Prediction favors away team and travel/rest factors support away team
                confidence_adjustment = min(
                    abs(travel_rest_impact) * 0.1, 0.05
                )  # Max 5% boost
            elif predicted_winner == home_team and travel_rest_impact < 0:
                # Prediction favors home team but travel/rest factors favor away team
                confidence_adjustment = max(
                    travel_rest_impact * 0.1, -0.05
                )  # Max 5% reduction
            elif predicted_winner != home_team and travel_rest_impact > 0:
                # Prediction favors away team but travel/rest factors favor home team
                confidence_adjustment = max(
                    -travel_rest_impact * 0.1, -0.05
                )  # Max 5% reduction

            # Apply adjustment
            original_confidence = prediction.get("confidence", 0.5)
            adjusted_confidence = max(
                0.05, min(0.95, original_confidence + confidence_adjustment)
            )

            # Create enhanced prediction
            enhanced_prediction = prediction.copy()
            enhanced_prediction["confidence"] = adjusted_confidence
            enhanced_prediction["original_confidence"] = original_confidence
            enhanced_prediction["travel_rest_adjustment"] = confidence_adjustment
            enhanced_prediction["travel_rest_analysis"] = analysis

            # Add travel/rest features to the prediction
            if "features" not in enhanced_prediction:
                enhanced_prediction["features"] = {}

            enhanced_prediction["features"].update(
                {
                    "travel_rest_impact": travel_rest_impact,
                    "home_travel_fatigue": analysis.get("home_travel_fatigue", 0.0),
                    "away_travel_fatigue": analysis.get("away_travel_fatigue", 0.0),
                    "home_pitcher_rest": analysis.get("home_pitcher_rest_factor", 0.0),
                    "away_pitcher_rest": analysis.get("away_pitcher_rest_factor", 0.0),
                }
            )

            logger.info(
                f"🎯 Enhanced prediction with travel/rest: {original_confidence:.3f} → {adjusted_confidence:.3f} "
                f"(adjustment: {confidence_adjustment:+.3f})"
            )

            return enhanced_prediction

        except Exception as e:
            logger.error(f"Error enhancing prediction with travel/rest: {e}")
            return prediction


# Example usage
async def main():
    """Example usage of the travel and rest analyzer."""
    analyzer = TravelAndRestAnalyzer()

    # Record some example travel
    await analyzer.record_team_travel(
        team_name="Los Angeles Dodgers",
        from_city="Los Angeles",
        to_city="New York",
        travel_date="2025-07-20",
        game_date="2025-07-21",
        travel_distance=2789.0,
    )

    # Record pitcher starts
    await analyzer.record_pitcher_start(
        "Clayton Kershaw", "Los Angeles Dodgers", "2025-07-18"
    )
    await analyzer.record_pitcher_start("Gerrit Cole", "New York Yankees", "2025-07-19")

    # Analyze a game
    analysis = await analyzer.analyze_game_travel_rest(
        game_id="game_123",
        home_team="New York Yankees",
        away_team="Los Angeles Dodgers",
        game_date="2025-07-21",
        home_pitcher="Gerrit Cole",
        away_pitcher="Clayton Kershaw",
    )

    print("Travel and Rest Analysis:")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
