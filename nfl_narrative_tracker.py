#!/usr/bin/env python3
"""
NFL Team Narrative Tracker
Builds narrative sheets for each team throughout the season (Week 1-16).
Tracks how the league/media frames each team and how it affects betting patterns.
"""

import json
from datetime import datetime
from pathlib import Path

class NFLTeamNarrativeTracker:
    """Track and analyze NFL team narratives by week"""
    
    NFL_TEAMS = {
        # AFC East
        'BUF': {'name': 'Buffalo Bills', 'division': 'AFC East', 'market': 'Buffalo, NY'},
        'MIA': {'name': 'Miami Dolphins', 'division': 'AFC East', 'market': 'Miami, FL'},
        'NE': {'name': 'New England Patriots', 'division': 'AFC East', 'market': 'Boston, MA'},
        'NYJ': {'name': 'New York Jets', 'division': 'AFC East', 'market': 'New York, NY'},
        
        # AFC North
        'BAL': {'name': 'Baltimore Ravens', 'division': 'AFC North', 'market': 'Baltimore, MD'},
        'PIT': {'name': 'Pittsburgh Steelers', 'division': 'AFC North', 'market': 'Pittsburgh, PA'},
        'CLE': {'name': 'Cleveland Browns', 'division': 'AFC North', 'market': 'Cleveland, OH'},
        'CIN': {'name': 'Cincinnati Bengals', 'division': 'AFC North', 'market': 'Cincinnati, OH'},
        
        # AFC South
        'IND': {'name': 'Indianapolis Colts', 'division': 'AFC South', 'market': 'Indianapolis, IN'},
        'TEN': {'name': 'Tennessee Titans', 'division': 'AFC South', 'market': 'Nashville, TN'},
        'HOU': {'name': 'Houston Texans', 'division': 'AFC South', 'market': 'Houston, TX'},
        'JAX': {'name': 'Jacksonville Jaguars', 'division': 'AFC South', 'market': 'Jacksonville, FL'},
        
        # AFC West
        'KC': {'name': 'Kansas City Chiefs', 'division': 'AFC West', 'market': 'Kansas City, MO'},
        'LAC': {'name': 'Los Angeles Chargers', 'division': 'AFC West', 'market': 'Los Angeles, CA'},
        'LV': {'name': 'Las Vegas Raiders', 'division': 'AFC West', 'market': 'Las Vegas, NV'},
        'DEN': {'name': 'Denver Broncos', 'division': 'AFC West', 'market': 'Denver, CO'},
        
        # NFC East
        'DAL': {'name': 'Dallas Cowboys', 'division': 'NFC East', 'market': 'Dallas, TX'},
        'PHI': {'name': 'Philadelphia Eagles', 'division': 'NFC East', 'market': 'Philadelphia, PA'},
        'WAS': {'name': 'Washington Commanders', 'division': 'NFC East', 'market': 'Washington, DC'},
        'NYG': {'name': 'New York Giants', 'division': 'NFC East', 'market': 'New York, NY'},
        
        # NFC North
        'GB': {'name': 'Green Bay Packers', 'division': 'NFC North', 'market': 'Green Bay, WI'},
        'MIN': {'name': 'Minnesota Vikings', 'division': 'NFC North', 'market': 'Minneapolis, MN'},
        'DET': {'name': 'Detroit Lions', 'division': 'NFC North', 'market': 'Detroit, MI'},
        'CHI': {'name': 'Chicago Bears', 'division': 'NFC North', 'market': 'Chicago, IL'},
        
        # NFC South
        'NO': {'name': 'New Orleans Saints', 'division': 'NFC South', 'market': 'New Orleans, LA'},
        'TB': {'name': 'Tampa Bay Buccaneers', 'division': 'NFC South', 'market': 'Tampa Bay, FL'},
        'ATL': {'name': 'Atlanta Falcons', 'division': 'NFC South', 'market': 'Atlanta, GA'},
        'CAR': {'name': 'Carolina Panthers', 'division': 'NFC South', 'market': 'Charlotte, NC'},
        
        # NFC West
        'SF': {'name': 'San Francisco 49ers', 'division': 'NFC West', 'market': 'San Francisco, CA'},
        'LAR': {'name': 'Los Angeles Rams', 'division': 'NFC West', 'market': 'Los Angeles, CA'},
        'SEA': {'name': 'Seattle Seahawks', 'division': 'NFC West', 'market': 'Seattle, WA'},
        'ARI': {'name': 'Arizona Cardinals', 'division': 'NFC West', 'market': 'Phoenix, AZ'},
    }
    
    # Narrative archetypes the NFL uses
    NARRATIVE_TYPES = {
        'SUPER_BOWL_CONTENDER': {
            'description': 'Built as clear playoff favorites',
            'markers': ['Star QB', 'High expectations', 'Media darling', 'Vegas favorite'],
            'betting_edge': 'Often overpriced early, vulnerable to upsets late season'
        },
        'DARK_HORSE': {
            'description': 'Surprise team gaining credibility',
            'markers': ['Outperforming expectations', 'Cinderella story', 'Building momentum'],
            'betting_edge': 'Lines lag reality, good value until recognized'
        },
        'DEFENDING_CHAMP': {
            'description': 'Championship team defending title',
            'markers': ['Experience advantage', 'Pressure narrative', 'Target on back'],
            'betting_edge': 'Underrated early, may underperform pressure'
        },
        'REBUILDING': {
            'description': 'Team in rebuild mode',
            'markers': ['Young players', 'Development focus', 'Tank possibility'],
            'betting_edge': 'Highly volatile, inconsistent performances'
        },
        'VETERAN_GROUP': {
            'description': 'Experienced roster chasing ring',
            'markers': ['Veteran QB', 'Aging stars', 'Now or never narrative'],
            'betting_edge': 'Motivated, but physical decline risk'
        },
        'INJURY_PLAGUED': {
            'description': 'Team dealing with injuries',
            'markers': ['Key players out', 'Backup QB', 'Season in jeopardy'],
            'betting_edge': 'Lines may overcorrect, backup can perform'
        },
        'PLAYOFF_BUBBLE': {
            'description': 'Fighting for playoff spot',
            'markers': ['Close division', 'Desperation games', 'Elevated urgency'],
            'betting_edge': 'Motivation high, but inconsistent performance'
        },
        'DIVISION_LEADER': {
            'description': 'Atop division',
            'markers': ['First place', 'Controlling narrative', 'Others chasing'],
            'betting_edge': 'Lines respect strength, look for letdowns'
        }
    }
    
    def __init__(self):
        self.narratives = {}
        self.load_or_initialize()
    
    def load_or_initialize(self):
        """Load existing narratives or create new ones"""
        narrative_file = Path('data/nfl_narratives/team_narratives.json')
        
        if narrative_file.exists():
            with open(narrative_file) as f:
                self.narratives = json.load(f)
        else:
            # Initialize all teams
            self.narratives = {
                team_code: {
                    'name': team_info['name'],
                    'division': team_info['division'],
                    'weeks': {}
                }
                for team_code, team_info in self.NFL_TEAMS.items()
            }
            self.save()
    
    def add_week_narrative(self, team_code, week, data):
        """Add narrative data for a team in a specific week"""
        if team_code not in self.narratives:
            return f"Team {team_code} not found"
        
        if 'weeks' not in self.narratives[team_code]:
            self.narratives[team_code]['weeks'] = {}
        
        self.narratives[team_code]['weeks'][str(week)] = {
            'timestamp': datetime.now().isoformat(),
            'win_loss': data.get('win_loss', '0-0'),
            'narrative_type': data.get('narrative_type', ''),
            'narrative_arc': data.get('narrative_arc', ''),
            'key_events': data.get('key_events', []),
            'media_sentiment': data.get('media_sentiment', 'neutral'),
            'playoff_implications': data.get('playoff_implications', ''),
            'betting_notes': data.get('betting_notes', ''),
            'injury_status': data.get('injury_status', []),
            'schedule_strength': data.get('schedule_strength', ''),
            'vegas_line_movement': data.get('vegas_line_movement', ''),
        }
        self.save()
        return f"Added week {week} narrative for {team_code}"
    
    def generate_team_narrative_sheet(self, team_code):
        """Generate full narrative sheet for a team (Week 1-16)"""
        if team_code not in self.narratives:
            return f"Team {team_code} not found"
        
        team_data = self.narratives[team_code]
        team_name = team_data['name']
        division = team_data['division']
        
        sheet = f"""# 📊 {team_name} NARRATIVE TRACKING SHEET
Division: {division}
Tracking Period: Weeks 1-16 + Playoffs

---

## 🎯 Season Narrative Arc

### Overall Story
[This evolves week-by-week based on performance and media narrative]

---

"""
        
        # Generate week-by-week breakdown
        for week in range(1, 17):
            week_key = str(week)
            if week_key in team_data['weeks']:
                week_data = team_data['weeks'][week_key]
                
                sheet += f"""## Week {week}

**Record**: {week_data.get('win_loss', 'TBD')}

**Narrative Type**: {week_data.get('narrative_type', 'TBD')}

**Narrative Arc**: {week_data.get('narrative_arc', 'TBD')}

**Media Sentiment**: {week_data.get('media_sentiment', 'neutral')}

**Key Events**:
"""
                for event in week_data.get('key_events', []):
                    sheet += f"- {event}\n"
                
                sheet += f"""
**Playoff Implications**: {week_data.get('playoff_implications', 'TBD')}

**Injury Status**: {', '.join(week_data.get('injury_status', [])) or 'No major injuries'}

**Schedule Strength**: {week_data.get('schedule_strength', 'TBD')}

**Vegas Line Movement**: {week_data.get('vegas_line_movement', 'TBD')}

**Betting Notes**: 
{week_data.get('betting_notes', 'TBD')}

---

"""
        
        sheet += f"""## 🏆 Final Narrative (Week 16 + Playoffs)

**Overall Season Story**: [To be determined]

**Playoff Seeding Narrative**: [To be determined]

**Key Turning Points**:
[Document 3-5 major moments that shaped season narrative]

**Betting Pattern Analysis**:
[How did narrative affect line movements and outcomes?]

**Key Insights for Future**: 
[What did we learn about this team's betting patterns?]

---

Generated: {datetime.now().strftime('%Y-%m-%d')}
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return sheet
    
    def generate_all_team_sheets(self):
        """Generate narrative sheets for all 32 teams"""
        output_dir = Path('data/nfl_narratives')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for team_code in self.NFL_TEAMS.keys():
            sheet = self.generate_team_narrative_sheet(team_code)
            file_path = output_dir / f'{team_code}_narrative.md'
            with open(file_path, 'w') as f:
                f.write(sheet)
        
        return f"Generated narrative sheets for all 32 teams"
    
    def save(self):
        """Save narratives to file"""
        output_dir = Path('data/nfl_narratives')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'team_narratives.json', 'w') as f:
            json.dump(self.narratives, f, indent=2)
    
    def get_division_narratives(self, division):
        """Get current narratives for all teams in a division"""
        teams = [code for code, info in self.NFL_TEAMS.items() if info['division'] == division]
        return {code: self.narratives.get(code, {}) for code in teams}
    
    def track_narrative_evolution(self, team_code, weeks_to_analyze=3):
        """Analyze how team's narrative is evolving"""
        if team_code not in self.narratives:
            return None
        
        team = self.narratives[team_code]
        weeks = sorted([int(w) for w in team.get('weeks', {}).keys()])
        
        if len(weeks) < weeks_to_analyze:
            return None
        
        recent_weeks = weeks[-weeks_to_analyze:]
        narratives = [team['weeks'][str(w)].get('narrative_type', '') for w in recent_weeks]
        
        return {
            'team': team_code,
            'recent_weeks': recent_weeks,
            'narrative_evolution': narratives,
            'trend': 'STABLE' if len(set(narratives)) == 1 else 'CHANGING'
        }


def main():
    import sys
    
    tracker = NFLTeamNarrativeTracker()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'generate':
            print(tracker.generate_all_team_sheets())
        elif command == 'team' and len(sys.argv) > 2:
            team_code = sys.argv[2]
            sheet = tracker.generate_team_narrative_sheet(team_code)
            print(sheet)
        elif command == 'add' and len(sys.argv) > 4:
            team_code = sys.argv[2]
            week = int(sys.argv[3])
            # Data would be passed via JSON or prompt
            print(f"Add narrative for {team_code} Week {week}")
        else:
            print("Usage:")
            print("  python nfl_narrative_tracker.py generate")
            print("  python nfl_narrative_tracker.py team <TEAM_CODE>")
    else:
        print("NFL Team Narrative Tracker initialized")
        print("Use 'python nfl_narrative_tracker.py generate' to create narrative sheets")

if __name__ == '__main__':
    main()
