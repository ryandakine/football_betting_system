#!/usr/bin/env python3
import os
import re
from collections import defaultdict
import statistics

def parse_crew_data(directory):
    """Parse all team files and extract crew statistics."""
    crew_data = defaultdict(lambda: {
        'games': 0,
        'margins': [],
        'penalty_diffs': [],
        'overtime_rates': [],
        'labels': set(),
        'teams': []
    })
    
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith('.md'):
            continue
        
        team = filename.replace('.md', '')
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract crew rotation lines
        rotation_pattern = r'- \d+: ([^(]+) \((\d+) games;.*?avg margin ([+-]?\d+\.?\d*);.*?labels: ([^)]+)\)'
        for match in re.finditer(rotation_pattern, content):
            crew_name = match.group(1).strip()
            games = int(match.group(2))
            margin = float(match.group(3))
            labels = match.group(4).strip()
            
            crew_data[crew_name]['games'] += games
            crew_data[crew_name]['margins'].append(margin)
            crew_data[crew_name]['labels'].add(labels)
            crew_data[crew_name]['teams'].append(team)
        
        # Extract Style Impact data
        impact_pattern = r'- ([^(]+) \(([^)]+)\): (\d+) games, avg margin ([+-]?\d+\.?\d*), penalties on team ([\d.]+), penalty diff ([+-]?\d+\.?\d*), odds delta ([+-]?\d+\.?\d*), overtime rate ([\d.]+)%'
        for match in re.finditer(impact_pattern, content):
            crew_name = match.group(1).strip()
            penalty_diff = float(match.group(6))
            overtime_rate = float(match.group(8))
            
            if crew_name in crew_data:
                crew_data[crew_name]['penalty_diffs'].append(penalty_diff)
                crew_data[crew_name]['overtime_rates'].append(overtime_rate)
    
    return crew_data

def analyze_crews(crew_data):
    """Generate summary statistics and categorize crews."""
    results = []
    
    for crew_name, data in sorted(crew_data.items()):
        if data['games'] == 0:
            continue
        
        avg_margin = statistics.mean(data['margins']) if data['margins'] else 0
        avg_penalty_diff = statistics.mean(data['penalty_diffs']) if data['penalty_diffs'] else 0
        avg_overtime = statistics.mean(data['overtime_rates']) if data['overtime_rates'] else 0
        
        # Categorize crew type
        category = "Balanced"
        if abs(avg_margin) > 15:
            category = "Blowout" if avg_margin < 0 else "Competitive"
        elif avg_penalty_diff > 1:
            category = "Flag Heavy"
        elif avg_penalty_diff < -1:
            category = "Lenient"
        elif avg_overtime > 7:
            category = "Overtime Prone"
        
        results.append({
            'crew': crew_name,
            'games': data['games'],
            'avg_margin': avg_margin,
            'avg_penalty_diff': avg_penalty_diff,
            'avg_overtime': avg_overtime,
            'category': category,
            'labels': ', '.join(sorted(data['labels'])),
            'teams_count': len(set(data['teams']))
        })
    
    return results

def main():
    directory = '/home/ryan/code/football_betting_system/reports/referee_conspiracy'
    
    crew_data = parse_crew_data(directory)
    results = analyze_crews(crew_data)
    
    # Sort by impact
    results.sort(key=lambda x: abs(x['avg_margin']), reverse=True)
    
    print("\n" + "="*100)
    print("NFL REFEREE CREW TENDENCIES (2018-2024)")
    print("="*100)
    
    # By category
    categories = defaultdict(list)
    for r in results:
        categories[r['category']].append(r)
    
    for cat in sorted(categories.keys()):
        print(f"\n### {cat.upper()} ###")
        for r in sorted(categories[cat], key=lambda x: abs(x['avg_margin']), reverse=True):
            print(f"{r['crew']:20s} | Games: {r['games']:2d} | Margin: {r['avg_margin']:+6.1f} | Penalty Diff: {r['avg_penalty_diff']:+5.1f} | OT Rate: {r['avg_overtime']:5.1f}% | Teams: {r['teams_count']}")
    
    # Summary insights
    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    
    blowout_crews = [r for r in results if r['avg_margin'] < -12]
    competitive_crews = [r for r in results if r['avg_margin'] > 10]
    flag_heavy = [r for r in results if r['avg_penalty_diff'] > 1.5]
    
    if blowout_crews:
        print(f"\nBlowout Crews (avg margin < -12):")
        for c in blowout_crews:
            print(f"  - {c['crew']}: {c['avg_margin']:+.1f} point avg")
    
    if competitive_crews:
        print(f"\nCompetitive Crews (avg margin > +10):")
        for c in competitive_crews:
            print(f"  - {c['crew']}: {c['avg_margin']:+.1f} point avg")
    
    if flag_heavy:
        print(f"\nFlag-Heavy Crews (penalty diff > +1.5):")
        for c in flag_heavy:
            print(f"  - {c['crew']}: {c['avg_penalty_diff']:+.1f} diff")

if __name__ == '__main__':
    main()
