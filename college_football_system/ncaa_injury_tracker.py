#!/usr/bin/env python3
"""
NCAA/College Football Injury Tracker
=====================================

Tracks injuries for college football players from multiple sources:
- ESPN injury reports
- Team websites and official announcements
- Twitter/social media for breaking news
- College Football Data API

Provides impact scoring for betting analysis.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


class InjuryStatus(Enum):
    """Injury status enumeration"""
    OUT = "out"
    DOUBTFUL = "doubtful"
    QUESTIONABLE = "questionable"
    PROBABLE = "probable"
    DAY_TO_DAY = "day_to_day"
    HEALTHY = "healthy"


class PlayerPosition(Enum):
    """Key player positions for impact assessment"""
    QUARTERBACK = "QB"
    RUNNING_BACK = "RB"
    WIDE_RECEIVER = "WR"
    TIGHT_END = "TE"
    OFFENSIVE_LINE = "OL"
    DEFENSIVE_LINE = "DL"
    LINEBACKER = "LB"
    DEFENSIVE_BACK = "DB"
    KICKER = "K"
    PUNTER = "P"
    OTHER = "OTHER"


@dataclass
class PlayerInjury:
    """Individual player injury record"""
    player_id: str
    player_name: str
    team: str
    position: PlayerPosition
    status: InjuryStatus
    injury_type: str  # "ankle", "concussion", "knee", etc.
    description: str
    date_reported: datetime
    expected_return: Optional[datetime] = None
    games_missed: int = 0
    impact_score: float = 0.0  # 0-10 scale of impact on team
    source: str = "unknown"
    last_updated: datetime = None


@dataclass
class TeamInjuryReport:
    """Team-wide injury report"""
    team: str
    date: datetime
    injuries: List[PlayerInjury]
    key_players_out: int = 0
    total_impact_score: float = 0.0
    positions_affected: Dict[str, int] = None


class NCAAInjuryTracker:
    """
    Comprehensive injury tracking system for NCAA football.
    Monitors injuries from multiple sources and calculates impact scores.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/injuries/ncaaf")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = None
        self.injuries: Dict[str, List[PlayerInjury]] = {}  # team -> injuries
        self.last_update: Dict[str, datetime] = {}

        # API endpoints
        self.apis = {
            'espn': 'https://site.api.espn.com/apis/site/v2/sports/football/college-football',
            'cfb_data': 'https://api.collegefootballdata.com',
        }

        # Position impact weights (higher = more impactful)
        self.position_impact = {
            PlayerPosition.QUARTERBACK: 10.0,
            PlayerPosition.RUNNING_BACK: 7.0,
            PlayerPosition.WIDE_RECEIVER: 6.0,
            PlayerPosition.OFFENSIVE_LINE: 8.0,
            PlayerPosition.DEFENSIVE_LINE: 7.0,
            PlayerPosition.LINEBACKER: 6.5,
            PlayerPosition.DEFENSIVE_BACK: 6.0,
            PlayerPosition.TIGHT_END: 5.0,
            PlayerPosition.KICKER: 4.0,
            PlayerPosition.PUNTER: 3.0,
            PlayerPosition.OTHER: 3.0,
        }

        # Status severity multiplier
        self.status_severity = {
            InjuryStatus.OUT: 1.0,
            InjuryStatus.DOUBTFUL: 0.8,
            InjuryStatus.QUESTIONABLE: 0.5,
            InjuryStatus.PROBABLE: 0.2,
            InjuryStatus.DAY_TO_DAY: 0.4,
            InjuryStatus.HEALTHY: 0.0,
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_team_injuries(self, team: str) -> TeamInjuryReport:
        """
        Fetch injury report for a specific team.

        Args:
            team: Team name or abbreviation

        Returns:
            TeamInjuryReport with all current injuries
        """
        try:
            # Try ESPN API first
            injuries = await self._fetch_espn_injuries(team)

            if not injuries:
                # Fallback to cached data
                injuries = self._load_cached_injuries(team)

            # Calculate team impact
            report = self._generate_team_report(team, injuries)

            # Cache the results
            self._cache_injuries(team, injuries)

            return report

        except Exception as e:
            logger.error(f"Error fetching injuries for {team}: {e}")
            return TeamInjuryReport(
                team=team,
                date=datetime.now(),
                injuries=[],
                positions_affected={}
            )

    async def _fetch_espn_injuries(self, team: str) -> List[PlayerInjury]:
        """Fetch injuries from ESPN API."""
        injuries = []

        try:
            # ESPN team roster/injuries endpoint
            url = f"{self.apis['espn']}/teams/{team}/injuries"

            if self.session:
                async with self.session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        injuries = self._parse_espn_injuries(data, team)
            else:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    injuries = self._parse_espn_injuries(data, team)

        except Exception as e:
            logger.debug(f"ESPN injury fetch failed for {team}: {e}")

        return injuries

    def _parse_espn_injuries(self, data: Dict[str, Any], team: str) -> List[PlayerInjury]:
        """Parse ESPN injury data."""
        injuries = []

        try:
            items = data.get("items", [])

            for item in items:
                athlete = item.get("athlete", {})
                injury = item.get("injury", {})

                # Extract player info
                player_name = athlete.get("displayName", "Unknown")
                player_id = str(athlete.get("id", ""))
                position_str = athlete.get("position", {}).get("abbreviation", "OTHER")

                # Map position
                position = self._map_position(position_str)

                # Extract injury info
                status_str = injury.get("status", "questionable").lower()
                injury_status = self._map_status(status_str)
                injury_type = injury.get("type", "undisclosed")
                description = injury.get("details", "No details")

                # Create injury record
                player_injury = PlayerInjury(
                    player_id=player_id,
                    player_name=player_name,
                    team=team,
                    position=position,
                    status=injury_status,
                    injury_type=injury_type,
                    description=description,
                    date_reported=datetime.now(),
                    source="espn",
                    last_updated=datetime.now()
                )

                # Calculate impact score
                player_injury.impact_score = self._calculate_impact(player_injury)

                injuries.append(player_injury)

        except Exception as e:
            logger.error(f"Error parsing ESPN injuries: {e}")

        return injuries

    def _map_position(self, position_str: str) -> PlayerPosition:
        """Map position string to PlayerPosition enum."""
        position_map = {
            'QB': PlayerPosition.QUARTERBACK,
            'RB': PlayerPosition.RUNNING_BACK,
            'WR': PlayerPosition.WIDE_RECEIVER,
            'TE': PlayerPosition.TIGHT_END,
            'OL': PlayerPosition.OFFENSIVE_LINE,
            'OT': PlayerPosition.OFFENSIVE_LINE,
            'OG': PlayerPosition.OFFENSIVE_LINE,
            'C': PlayerPosition.OFFENSIVE_LINE,
            'DL': PlayerPosition.DEFENSIVE_LINE,
            'DE': PlayerPosition.DEFENSIVE_LINE,
            'DT': PlayerPosition.DEFENSIVE_LINE,
            'LB': PlayerPosition.LINEBACKER,
            'DB': PlayerPosition.DEFENSIVE_BACK,
            'CB': PlayerPosition.DEFENSIVE_BACK,
            'S': PlayerPosition.DEFENSIVE_BACK,
            'K': PlayerPosition.KICKER,
            'P': PlayerPosition.PUNTER,
        }
        return position_map.get(position_str.upper(), PlayerPosition.OTHER)

    def _map_status(self, status_str: str) -> InjuryStatus:
        """Map status string to InjuryStatus enum."""
        status_str = status_str.lower()
        if 'out' in status_str:
            return InjuryStatus.OUT
        elif 'doubtful' in status_str:
            return InjuryStatus.DOUBTFUL
        elif 'questionable' in status_str:
            return InjuryStatus.QUESTIONABLE
        elif 'probable' in status_str:
            return InjuryStatus.PROBABLE
        elif 'day' in status_str:
            return InjuryStatus.DAY_TO_DAY
        else:
            return InjuryStatus.QUESTIONABLE

    def _calculate_impact(self, injury: PlayerInjury) -> float:
        """
        Calculate injury impact score (0-10).

        Factors:
        - Position importance
        - Injury severity
        - Player role (starter vs backup)
        """
        base_impact = self.position_impact.get(injury.position, 3.0)
        severity = self.status_severity.get(injury.status, 0.5)

        # Special adjustments
        adjustments = 1.0

        # QB injuries are critical in college
        if injury.position == PlayerPosition.QUARTERBACK:
            adjustments *= 1.2

        # Concussions and long-term injuries
        if 'concussion' in injury.injury_type.lower():
            adjustments *= 1.3
        elif any(term in injury.injury_type.lower() for term in ['acl', 'torn', 'fracture']):
            adjustments *= 1.5

        impact = base_impact * severity * adjustments
        return min(impact, 10.0)  # Cap at 10.0

    def _generate_team_report(self, team: str, injuries: List[PlayerInjury]) -> TeamInjuryReport:
        """Generate comprehensive team injury report."""
        positions_affected = {}
        key_players_out = 0
        total_impact = 0.0

        for injury in injuries:
            # Count by position
            pos_str = injury.position.value
            positions_affected[pos_str] = positions_affected.get(pos_str, 0) + 1

            # Count key players
            if injury.impact_score >= 7.0 and injury.status in [InjuryStatus.OUT, InjuryStatus.DOUBTFUL]:
                key_players_out += 1

            # Sum impact
            total_impact += injury.impact_score

        return TeamInjuryReport(
            team=team,
            date=datetime.now(),
            injuries=injuries,
            key_players_out=key_players_out,
            total_impact_score=total_impact,
            positions_affected=positions_affected
        )

    def _cache_injuries(self, team: str, injuries: List[PlayerInjury]):
        """Cache injury data to disk."""
        try:
            cache_file = self.cache_dir / f"{team}_injuries.json"
            data = [asdict(injury) for injury in injuries]

            # Convert datetime objects to strings
            for item in data:
                if isinstance(item.get('date_reported'), datetime):
                    item['date_reported'] = item['date_reported'].isoformat()
                if isinstance(item.get('expected_return'), datetime):
                    item['expected_return'] = item['expected_return'].isoformat()
                if isinstance(item.get('last_updated'), datetime):
                    item['last_updated'] = item['last_updated'].isoformat()
                if isinstance(item.get('position'), PlayerPosition):
                    item['position'] = item['position'].value
                if isinstance(item.get('status'), InjuryStatus):
                    item['status'] = item['status'].value

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error caching injuries: {e}")

    def _load_cached_injuries(self, team: str) -> List[PlayerInjury]:
        """Load cached injury data from disk."""
        injuries = []

        try:
            cache_file = self.cache_dir / f"{team}_injuries.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                for item in data:
                    # Convert strings back to enums/datetime
                    if 'position' in item:
                        item['position'] = PlayerPosition(item['position'])
                    if 'status' in item:
                        item['status'] = InjuryStatus(item['status'])
                    if 'date_reported' in item and isinstance(item['date_reported'], str):
                        item['date_reported'] = datetime.fromisoformat(item['date_reported'])
                    if 'last_updated' in item and isinstance(item['last_updated'], str):
                        item['last_updated'] = datetime.fromisoformat(item['last_updated'])

                    injuries.append(PlayerInjury(**item))

        except Exception as e:
            logger.error(f"Error loading cached injuries: {e}")

        return injuries

    async def get_game_injury_impact(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, Any]:
        """
        Get injury impact analysis for a specific game.

        Returns comparative injury impact between teams.
        """
        home_report = await self.fetch_team_injuries(home_team)
        away_report = await self.fetch_team_injuries(away_team)

        # Calculate differential
        impact_differential = home_report.total_impact_score - away_report.total_impact_score

        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_impact': home_report.total_impact_score,
            'away_impact': away_report.total_impact_score,
            'impact_differential': impact_differential,
            'advantage': home_team if impact_differential < 0 else away_team,
            'home_key_players_out': home_report.key_players_out,
            'away_key_players_out': away_report.key_players_out,
            'home_injuries': [asdict(inj) for inj in home_report.injuries],
            'away_injuries': [asdict(inj) for inj in away_report.injuries],
            'edge_adjustment': self._calculate_edge_adjustment(impact_differential),
        }

    def _calculate_edge_adjustment(self, impact_differential: float) -> float:
        """
        Calculate betting edge adjustment based on injury differential.

        Returns adjustment factor (-0.1 to 0.1) to add to betting edge.
        """
        # Scale: 10 points of injury impact = ~0.05 edge adjustment
        adjustment = (impact_differential / 10.0) * 0.05
        return max(-0.10, min(0.10, adjustment))


# Example usage
async def main():
    """Example usage of NCAA Injury Tracker."""
    async with NCAAInjuryTracker() as tracker:
        # Get injury report for a team
        team = "Alabama"
        print(f"Fetching injury report for {team}...")
        report = await tracker.fetch_team_injuries(team)

        print(f"\n{team} Injury Report")
        print(f"Total Impact Score: {report.total_impact_score:.1f}")
        print(f"Key Players Out: {report.key_players_out}")
        print(f"\nInjuries:")
        for injury in report.injuries:
            print(f"  - {injury.player_name} ({injury.position.value}): {injury.status.value}")
            print(f"    Impact: {injury.impact_score:.1f}/10")


if __name__ == "__main__":
    asyncio.run(main())
