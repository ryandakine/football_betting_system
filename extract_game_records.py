#!/usr/bin/env python3
import os
import re
import json
from collections import defaultdict

def parse_team_file(filepath, team):
    """Extract individual game records from team markdown file."""
    games = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find crew rotation section
    rotation_section = re.search(r'## Crew Rotation Timeline\n(.*?)##', content, re.DOTALL)
    if not rotation_section:
        return games
    
    rotation_text = rotation_section.group(1)
    
    # Parse each crew assignment
    # Format: - 2018: CrowName (1 games; weeks 3; avg margin -2.0; labels: ...)
    crew_pattern = r'- (\d+): ([^(]+) \((\d+) games; weeks ([^;]+); avg margin ([+-]?\d+\.?\d*);'
    
    for match in re.finditer(crew_pattern, rotation_text):
        year = int(match.group(1))
        crew_name = match.group(2).strip()
        num_games = int(match.group(3))
        weeks_str = match.group(4).strip()
        avg_margin = float(match.group(5))
        
        # Parse weeks (can be single or comma-separated)
        weeks = []
        for week_part in weeks_str.split(','):
            week_part = week_part.strip()
            if week_part.isdigit():
                weeks.append(int(week_part))
        
        # If we have weeks info, create individual game records
        if weeks:
            # Distribute margin across weeks (approximate)
            margin_per_game = avg_margin
            for week in weeks:
                games.append({
                    'team': team,
                    'year': year,
                    'week': week,
                    'crew': crew_name,
                    'margin': margin_per_game,
                    'num_games_in_assignment': num_games
                })
        else:
            # If no specific weeks, create one record per game
            for i in range(num_games):
                games.append({
                    'team': team,
                    'year': year,
                    'week': None,  # Unknown week
                    'crew': crew_name,
                    'margin': avg_margin,
                    'num_games_in_assignment': num_games
                })
    
    return games

def parse_narrative(filepath, team):
    """Extract narrative events (collapses, surges, OT situations)."""
    events = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find narrative section
    narrative_section = re.search(r'## Narrative Correlations\n(.*?)##', content, re.DOTALL)
    if not narrative_section:
        return events
    
    narrative_text = narrative_section.group(1)
    
    # Parse events: - Week X YEAR vs TEAM: EVENT with CREW
    event_pattern = r'- ([^\n]+)'
    
    for match in re.finditer(event_pattern, narrative_text):
        event_text = match.group(1)
        events.append({
            'team': team,
            'event': event_text
        })
    
    return events

def main():
    directory = '/home/ryan/code/football_betting_system/reports/referee_conspiracy'
    
    all_games = []
    all_events = []
    
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith('.md'):
            continue
        
        team = filename.replace('.md', '')
        filepath = os.path.join(directory, filename)
        
        games = parse_team_file(filepath, team)
        events = parse_narrative(filepath, team)
        
        all_games.extend(games)
        all_events.extend(events)
        
        print(f"✓ {team:5s}: {len(games):3d} game records extracted")
    
    # Save extracted data
    output_data = {
        'games': all_games,
        'events': all_events,
        'total_records': len(all_games)
    }
    
    with open('/home/ryan/code/football_betting_system/data/referee_conspiracy/game_records.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Extracted {len(all_games)} game records")
    print(f"✅ Extracted {len(all_events)} narrative events")
    
    # Print sample
    print("\n" + "="*100)
    print("SAMPLE GAME RECORDS")
    print("="*100)
    for game in all_games[:10]:
        print(f"{game['team']:5s} | Week {game['week'] if game['week'] else '??':2} | {game['year']} | "
              f"{game['crew']:20s} | Margin: {game['margin']:+6.1f}")
    
    print("\n" + "="*100)
    print("SAMPLE EVENTS")
    print("="*100)
    for event in all_events[:5]:
        print(f"{event['team']:5s} | {event['event']}")

if __name__ == '__main__':
    main()
