#!/usr/bin/env python3
import os
import re
from collections import defaultdict
import statistics

def parse_all_crews_by_team(directory):
    """Parse all team files and track crew assignments."""
    crew_team_map = defaultdict(lambda: defaultdict(list))  # crew -> team -> [margins]
    crew_appearances = defaultdict(int)  # crew -> total appearances
    
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith('.md'):
            continue
        
        team = filename.replace('.md', '')
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract crew rotation data
        rotation_pattern = r'- \d+: ([^(]+) \((\d+) games; weeks ([^;]+); avg margin ([+-]?\d+\.?\d*);'
        for match in re.finditer(rotation_pattern, content):
            crew_name = match.group(1).strip()
            games = int(match.group(2))
            margin = float(match.group(4))
            
            crew_team_map[crew_name][team].append(margin)
            crew_appearances[crew_name] += games
    
    return crew_team_map, crew_appearances

def analyze_crew_team_patterns(crew_team_map, crew_appearances):
    """Analyze how each crew treats different teams."""
    results = []
    
    for crew_name, team_data in crew_team_map.items():
        if crew_appearances[crew_name] < 10:  # Minimum 10 games
            continue
        
        # Analyze variance across teams
        all_margins = []
        team_avgs = []
        team_count = len(team_data)
        
        for team, margins in team_data.items():
            avg = statistics.mean(margins)
            team_avgs.append(avg)
            all_margins.extend(margins)
        
        overall_avg = statistics.mean(all_margins)
        overall_std = statistics.stdev(all_margins)
        
        # Find teams where crew has strongest bias
        team_biases = []
        for team, margins in team_data.items():
            avg = statistics.mean(margins)
            team_biases.append({
                'team': team,
                'avg_margin': avg,
                'games': len(margins),
                'deviation': avg - overall_avg
            })
        
        # Sort by deviation (how much they favor/disfavor this team)
        team_biases.sort(key=lambda x: abs(x['deviation']), reverse=True)
        
        # Determine crew type based on variance
        team_avg_variance = statistics.stdev(team_avgs) if len(team_avgs) > 1 else 0
        
        crew_type = "Consistent"
        if team_avg_variance > 8:
            crew_type = "Team-Selective"
        elif team_avg_variance > 5:
            crew_type = "Favor-Based"
        
        results.append({
            'crew': crew_name,
            'total_games': crew_appearances[crew_name],
            'teams_covered': team_count,
            'overall_avg_margin': overall_avg,
            'overall_std': overall_std,
            'team_avg_variance': team_avg_variance,
            'crew_type': crew_type,
            'top_biases': team_biases[:5],
            'bottom_biases': team_biases[-5:]
        })
    
    return results

def main():
    directory = '/home/ryan/code/football_betting_system/reports/referee_conspiracy'
    crew_team_map, crew_appearances = parse_all_crews_by_team(directory)
    results = analyze_crew_team_patterns(crew_team_map, crew_appearances)
    
    # Sort by team_avg_variance (most selective crews first)
    results.sort(key=lambda x: x['team_avg_variance'], reverse=True)
    
    print("\n" + "="*120)
    print("REFEREE CREW TEAM-SELECTIVE PATTERNS (2018-2024)")
    print("="*120)
    print("\nShowing how referee crews move between teams and their differential impact...")
    
    # Group by crew type
    crew_types = defaultdict(list)
    for r in results:
        crew_types[r['crew_type']].append(r)
    
    for ctype in ['Team-Selective', 'Favor-Based', 'Consistent']:
        if ctype not in crew_types:
            continue
        
        print(f"\n### {ctype.upper()} CREWS ###")
        print("-" * 120)
        
        for r in sorted(crew_types[ctype], key=lambda x: x['team_avg_variance'], reverse=True):
            print(f"\n{r['crew']:20s} | {r['total_games']:3d} games across {r['teams_covered']:2d} teams | "
                  f"Variance: {r['team_avg_variance']:5.2f} | Overall avg: {r['overall_avg_margin']:+6.1f}")
            
            # Show biggest favors and disfavors
            print(f"  MOST FAVORABLE FOR:")
            for bias in r['top_biases'][:3]:
                print(f"    {bias['team']:5s}: {bias['avg_margin']:+6.1f} ({bias['games']} games) | "
                      f"Deviation: {bias['deviation']:+6.1f} points")
            
            print(f"  MOST UNFAVORABLE FOR:")
            for bias in r['bottom_biases'][-3:]:
                print(f"    {bias['team']:5s}: {bias['avg_margin']:+6.1f} ({bias['games']} games) | "
                      f"Deviation: {bias['deviation']:+6.1f} points")
    
    # Summary insights
    print("\n" + "="*120)
    print("KEY INSIGHTS")
    print("="*120)
    
    team_selective = crew_types.get('Team-Selective', [])
    if team_selective:
        print(f"\n🎯 TEAM-SELECTIVE CREWS ({len(team_selective)} found):")
        print("These crews show strong biases for/against specific teams")
        for c in team_selective[:5]:
            print(f"  - {c['crew']:20s}: Variance {c['team_avg_variance']:.2f}")
    
    print(f"\n📊 Total crews analyzed: {len(results)}")
    print(f"   Team-Selective: {len(crew_types.get('Team-Selective', []))}")
    print(f"   Favor-Based: {len(crew_types.get('Favor-Based', []))}")
    print(f"   Consistent: {len(crew_types.get('Consistent', []))}")

if __name__ == '__main__':
    main()
