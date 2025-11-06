#!/usr/bin/env python3
import os
import re
from collections import defaultdict
import statistics

def parse_crew_data(directory):
    """Parse all team files and extract detailed crew game data."""
    crew_data = defaultdict(list)
    narrative_data = defaultdict(list)
    
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith('.md'):
            continue
        
        team = filename.replace('.md', '')
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract crew rotation - get all individual games
        rotation_pattern = r'- \d+: ([^(]+) \((\d+) games; weeks ([^;]+); avg margin ([+-]?\d+\.?\d*);'
        for match in re.finditer(rotation_pattern, content):
            crew_name = match.group(1).strip()
            games = int(match.group(2))
            margin = float(match.group(4))
            
            crew_data[crew_name].append({
                'team': team,
                'games': games,
                'margin': margin
            })
        
        # Extract narrative correlations for extreme games
        narrative_pattern = r'- ([^\n]+(?:collapse|surge|overtime|gut punch|blowout)[^\n]*)'
        for match in re.finditer(narrative_pattern, content, re.IGNORECASE):
            narrative = match.group(1).strip()
            # Extract crew name from narrative
            crew_match = re.search(r'([A-Z][a-z]+ (?:[A-Z][a-z]+)*)(?: stepping in)?', narrative)
            if crew_match:
                crew_name = crew_match.group(1).strip()
                narrative_data[crew_name].append({
                    'team': team,
                    'narrative': narrative
                })
    
    return crew_data, narrative_data

def analyze_extreme_patterns(crew_data):
    """Identify crews with extreme game outcomes."""
    results = []
    
    for crew_name, games in crew_data.items():
        if len(games) < 5:  # Need enough data
            continue
        
        margins = [g['margin'] for g in games]
        
        # Calculate statistics
        avg_margin = statistics.mean(margins)
        std_dev = statistics.stdev(margins) if len(margins) > 1 else 0
        min_margin = min(margins)
        max_margin = max(margins)
        range_margin = max_margin - min_margin
        
        # Count extreme games
        blowouts = len([m for m in margins if m < -20])
        blowouts_pos = len([m for m in margins if m > 20])
        close_games = len([m for m in margins if -3 <= m <= 3])
        
        # Determine pattern
        pattern = "Balanced"
        if blowouts > len(games) * 0.15:  # >15% blowouts
            pattern = "Blowout Facilitator"
        elif blowouts_pos > len(games) * 0.10:
            pattern = "Competitive Games"
        elif close_games > len(games) * 0.40:  # >40% close
            pattern = "Tight Game Handler"
        elif std_dev > 15:
            pattern = "High Variance"
        
        results.append({
            'crew': crew_name,
            'total_games': len(games),
            'avg_margin': avg_margin,
            'std_dev': std_dev,
            'range': range_margin,
            'blowouts_neg': blowouts,
            'blowouts_pos': blowouts_pos,
            'close_games': close_games,
            'close_pct': (close_games / len(games)) * 100,
            'pattern': pattern,
            'min_margin': min_margin,
            'max_margin': max_margin
        })
    
    return results

def main():
    directory = '/home/ryan/code/football_betting_system/reports/referee_conspiracy'
    
    crew_data, narrative_data = parse_crew_data(directory)
    results = analyze_extreme_patterns(crew_data)
    
    # Sort by pattern relevance
    results.sort(key=lambda x: (x['pattern'], x['std_dev']), reverse=True)
    
    print("\n" + "="*130)
    print("NFL REFEREE CREW BLOWOUT & GAME-CONTROL PATTERNS (2018-2024)")
    print("="*130)
    
    # By pattern
    patterns = defaultdict(list)
    for r in results:
        patterns[r['pattern']].append(r)
    
    for pattern in sorted(patterns.keys()):
        print(f"\n### {pattern.upper()} ###")
        for r in sorted(patterns[pattern], key=lambda x: x['std_dev'], reverse=True):
            print(f"{r['crew']:20s} | Games: {r['total_games']:3d} | AvgMargin: {r['avg_margin']:+6.1f} | StdDev: {r['std_dev']:5.1f} | "
                  f"Range: {r['range']:6.1f} | Blowouts(-/+): {r['blowouts_neg']}/{r['blowouts_pos']} | Close: {r['close_pct']:5.1f}% | "
                  f"Extremes: {r['min_margin']:+.0f} to {r['max_margin']:+.0f}")
    
    # Detailed insights
    print("\n" + "="*130)
    print("DETAILED CREW PROFILES")
    print("="*130)
    
    blowout_facilitators = [r for r in results if r['pattern'] == 'Blowout Facilitator']
    high_variance = [r for r in results if r['pattern'] == 'High Variance']
    tight_handlers = [r for r in results if r['pattern'] == 'Tight Game Handler']
    
    if blowout_facilitators:
        print(f"\n🔴 BLOWOUT FACILITATORS (>15% games with 20+ point margins):")
        for c in sorted(blowout_facilitators, key=lambda x: x['blowouts_neg'], reverse=True):
            print(f"  {c['crew']:20s}: {c['blowouts_neg']} blowout games out of {c['total_games']} ({(c['blowouts_neg']/c['total_games'])*100:.1f}%)")
    
    if high_variance:
        print(f"\n📊 HIGH VARIANCE CREWS (std dev > 15):")
        for c in sorted(high_variance, key=lambda x: x['std_dev'], reverse=True):
            print(f"  {c['crew']:20s}: StdDev {c['std_dev']:.1f}, Range {c['min_margin']:+.0f} to {c['max_margin']:+.0f}")
    
    if tight_handlers:
        print(f"\n⚖️ TIGHT GAME HANDLERS (>40% within 3 points):")
        for c in sorted(tight_handlers, key=lambda x: x['close_pct'], reverse=True):
            print(f"  {c['crew']:20s}: {c['close_games']} tight games ({c['close_pct']:.1f}%)")

if __name__ == '__main__':
    main()
