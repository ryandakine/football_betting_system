#!/usr/bin/env python3
"""
NFL Injury Tracker
==================
Tracks key player injuries and analyzes impact on betting lines.

Injury Impact Framework:
- QB OUT: -3 to -7 points (biggest impact)
- Top WR OUT: -1 to -3 points
- RB1 OUT: -0.5 to -2 points
- Elite Defender OUT: +1 to +3 points (favors offense)
- OL injuries: -1 to -2 points (pressure on QB)
"""

import asyncio
import json
import logging
import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/referee_conspiracy")
INJURY_CACHE_DIR = DATA_DIR / "injury_cache"
INJURY_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Position impact weights (how much player availability affects totals/spreads)
POSITION_WEIGHTS = {
    "QB": {"total": -5.0, "spread": -3.5},
    "WR1": {"total": -2.0, "spread": -1.5},
    "WR2": {"total": -1.0, "spread": -0.5},
    "RB1": {"total": -1.5, "spread": -1.0},
    "TE1": {"total": -1.0, "spread": -0.5},
    "OL": {"total": -1.5, "spread": -1.0},
    "EDGE": {"total": 2.0, "spread": 1.5},  # Defense out = more offense
    "CB1": {"total": 2.0, "spread": 1.0},
    "S": {"total": 1.5, "spread": 0.5},
}


@dataclass
class PlayerInjury:
    """Individual player injury"""
    name: str
    team: str
    position: str
    status: str  # 'OUT', 'DOUBTFUL', 'QUESTIONABLE', 'PROBABLE'
    injury: str
    
    # Impact
    importance: float = 0.0  # 0-1 scale
    total_impact: float = 0.0  # Points adjustment to total
    spread_impact: float = 0.0  # Points adjustment to spread
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TeamInjuryReport:
    """Injury report for a team"""
    team: str
    game_date: str
    
    injuries: List[PlayerInjury]
    
    # Cumulative impact
    total_impact: float = 0.0
    spread_impact: float = 0.0  # Negative = hurts team, positive = helps opponent
    severity: float = 0.0  # 0-1 overall injury severity
    
    edge_detected: bool = False
    recommendation: str = ""
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data["injuries"] = [inj.to_dict() for inj in self.injuries]
        return data


class InjuryTracker:
    """Track NFL injuries and analyze betting impact"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self
    
    async def __aexit__(self, *exc):
        if self.session:
            await self.session.close()
    
    async def fetch_team_injuries(
        self,
        team: str,
        game_date: Optional[date] = None
    ) -> TeamInjuryReport:
        """Fetch injury report for a team"""
        
        game_date = game_date or date.today()
        
        # Check cache
        cache_key = f"{team}_{game_date}"
        cache_file = INJURY_CACHE_DIR / f"injuries_{cache_key}.json"
        
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            logger.info(f"Using cached injuries for {cache_key}")
            return self._deserialize_report(cached)
        
        try:
            # Try ESPN injury report first
            injuries = await self._scrape_espn_injuries(team)
            
            if not injuries:
                # Fallback to mock/manual data
                injuries = self._get_mock_injuries(team)
            
            # Analyze impact
            report = TeamInjuryReport(
                team=team,
                game_date=game_date.isoformat(),
                injuries=injuries,
            )
            
            report = self._analyze_injury_impact(report)
            
            # Cache
            cache_file.write_text(json.dumps(report.to_dict(), indent=2))
            
            return report
        
        except Exception as e:
            logger.error(f"Failed to fetch injuries for {team}: {e}")
            return TeamInjuryReport(
                team=team,
                game_date=game_date.isoformat(),
                injuries=[],
            )
    
    async def _scrape_espn_injuries(self, team: str) -> List[PlayerInjury]:
        """Scrape ESPN injury report"""
        
        # ESPN injury page URL (example)
        url = f"https://www.espn.com/nfl/team/injuries/_/name/{team.lower()}"
        
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(f"ESPN injuries returned {resp.status} for {team}")
                    return []
                
                html = await resp.text()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            injuries = []
            
            # Find injury table (ESPN structure)
            injury_rows = soup.select('tr.Table__TR')
            
            for row in injury_rows:
                try:
                    cols = row.select('td')
                    if len(cols) < 3:
                        continue
                    
                    name = cols[0].get_text(strip=True)
                    position = cols[1].get_text(strip=True)
                    status = cols[2].get_text(strip=True).upper()
                    injury_type = cols[3].get_text(strip=True) if len(cols) > 3 else "Unknown"
                    
                    # Filter key positions
                    if not any(pos in position for pos in ["QB", "WR", "RB", "TE", "OL", "DE", "LB", "CB", "S"]):
                        continue
                    
                    # Filter status
                    if status not in ["OUT", "DOUBTFUL", "QUESTIONABLE"]:
                        continue
                    
                    injury = PlayerInjury(
                        name=name,
                        team=team,
                        position=position,
                        status=status,
                        injury=injury_type,
                    )
                    
                    injuries.append(injury)
                
                except Exception as e:
                    logger.debug(f"Could not parse injury row: {e}")
            
            logger.info(f"Scraped {len(injuries)} injuries for {team}")
            return injuries
        
        except Exception as e:
            logger.warning(f"ESPN scrape failed for {team}: {e}")
            return []
    
    def _get_mock_injuries(self, team: str) -> List[PlayerInjury]:
        """Mock injuries for testing"""
        
        # Common injury scenarios
        mock_data = {
            "BUF": [
                PlayerInjury("Josh Allen", "BUF", "QB", "QUESTIONABLE", "Shoulder"),
            ],
            "KC": [],  # Healthy
            "SF": [
                PlayerInjury("Brock Purdy", "SF", "QB", "OUT", "Elbow"),
                PlayerInjury("Deebo Samuel", "SF", "WR", "DOUBTFUL", "Hamstring"),
            ],
        }
        
        return mock_data.get(team, [])
    
    def _analyze_injury_impact(self, report: TeamInjuryReport) -> TeamInjuryReport:
        """Analyze betting impact of injuries"""
        
        total_impact = 0.0
        spread_impact = 0.0
        severity = 0.0
        
        for injury in report.injuries:
            # Determine position category
            pos_key = self._map_position(injury.position)
            
            if not pos_key:
                continue
            
            # Get base impact weights
            weights = POSITION_WEIGHTS.get(pos_key, {"total": 0, "spread": 0})
            
            # Status multiplier
            status_mult = {
                "OUT": 1.0,
                "DOUBTFUL": 0.7,
                "QUESTIONABLE": 0.3,
                "PROBABLE": 0.1,
            }.get(injury.status, 0.0)
            
            # Calculate impact
            injury.total_impact = weights["total"] * status_mult
            injury.spread_impact = weights["spread"] * status_mult
            injury.importance = abs(weights["total"]) / 5.0  # Normalize to 0-1
            
            total_impact += injury.total_impact
            spread_impact += injury.spread_impact
            severity += injury.importance * status_mult
        
        report.total_impact = total_impact
        report.spread_impact = spread_impact
        report.severity = min(1.0, severity)
        
        # Edge detection
        if abs(total_impact) >= 3.0 or abs(spread_impact) >= 2.5:
            report.edge_detected = True
            
            if total_impact < -3.0:
                report.recommendation = "LEAN_UNDER"
            elif spread_impact < -2.5:
                report.recommendation = "FADE_TEAM"
            elif total_impact > 3.0:
                report.recommendation = "LEAN_OVER"
            else:
                report.recommendation = "VALUE_OPPONENT"
        else:
            report.edge_detected = False
            report.recommendation = "NEUTRAL"
        
        return report
    
    def _map_position(self, position: str) -> Optional[str]:
        """Map full position to category"""
        
        if "QB" in position:
            return "QB"
        elif "WR" in position:
            return "WR1"  # Assume WR1 unless we have depth chart
        elif "RB" in position:
            return "RB1"
        elif "TE" in position:
            return "TE1"
        elif any(ol in position for ol in ["OT", "OG", "C"]):
            return "OL"
        elif any(edge in position for edge in ["DE", "OLB"]):
            return "EDGE"
        elif "CB" in position:
            return "CB1"
        elif "S" in position:
            return "S"
        
        return None
    
    def _deserialize_report(self, data: dict) -> TeamInjuryReport:
        """Deserialize cached report"""
        injuries = [PlayerInjury(**inj) for inj in data.pop("injuries", [])]
        return TeamInjuryReport(**data, injuries=injuries)


async def fetch_injuries_for_games(odds_file: Optional[Path] = None) -> Dict[str, TeamInjuryReport]:
    """Fetch injuries for all teams in games"""
    
    # Find most recent odds file
    if not odds_file:
        odds_files = sorted(DATA_DIR.glob("nfl_odds_*.json"))
        if not odds_files:
            logger.error("No odds files found")
            return {}
        odds_file = odds_files[-1]
    
    logger.info(f"Loading odds from {odds_file}")
    odds_data = json.loads(odds_file.read_text())
    
    # Get unique teams
    teams = set()
    for game in odds_data:
        teams.add(game.get("home_team"))
        teams.add(game.get("away_team"))
    
    teams.discard(None)
    
    # Fetch injuries for each team
    injury_reports = {}
    
    async with InjuryTracker() as tracker:
        for team in sorted(teams):
            try:
                report = await tracker.fetch_team_injuries(team)
                injury_reports[team] = report
                
                if report.injuries:
                    logger.info(
                        f"🏥 {team}: {len(report.injuries)} injuries, "
                        f"Impact: {report.total_impact:+.1f} pts, "
                        f"Severity: {report.severity:.0%}"
                    )
            
            except Exception as e:
                logger.error(f"Failed to fetch injuries for {team}: {e}")
    
    # Save injury analysis
    output_file = DATA_DIR / f"injury_analysis_{date.today()}.json"
    output_data = {team: report.to_dict() for team, report in injury_reports.items()}
    output_file.write_text(json.dumps(output_data, indent=2))
    logger.info(f"💾 Saved injury analysis to {output_file}")
    
    return injury_reports


async def main():
    """Main injury tracker runner"""
    injury_reports = await fetch_injuries_for_games()
    
    # Print summary
    print("\n" + "=" * 80)
    print("🏥 INJURY IMPACT ANALYSIS")
    print("=" * 80)
    
    significant = {
        team: report
        for team, report in injury_reports.items()
        if report.edge_detected
    }
    
    if significant:
        print(f"\n⚠️  {len(significant)} teams with significant injury impact:\n")
        
        for team, report in sorted(
            significant.items(),
            key=lambda x: x[1].severity,
            reverse=True
        ):
            print(f"🔴 {team}")
            print(f"   Severity: {report.severity:.0%}")
            print(f"   Total Impact: {report.total_impact:+.1f} points")
            print(f"   Spread Impact: {report.spread_impact:+.1f} points")
            print(f"   Recommendation: {report.recommendation}")
            print(f"   Key Injuries:")
            for inj in report.injuries:
                if inj.importance > 0.3:
                    print(f"      • {inj.name} ({inj.position}) - {inj.status}")
            print()
    else:
        print("\n✅ No significant injury impacts detected")
    
    return injury_reports


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
