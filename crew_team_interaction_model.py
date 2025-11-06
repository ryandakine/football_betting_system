#!/usr/bin/env python3
import json
import statistics
from collections import defaultdict

def load_game_records():
    """Load extracted game records."""
    with open('/home/ryan/code/football_betting_system/data/referee_conspiracy/game_records.json', 'r') as f:
        return json.load(f)

def build_crew_team_matrix(games):
    """Build a matrix: crew x team -> statistics."""
    matrix = defaultdict(lambda: defaultdict(list))  # crew -> team -> [margins]
    
    for game in games:
        crew = game['crew']
        team = game['team']
        margin = game['margin']
        
        matrix[crew][team].append(margin)
    
    return matrix

def calculate_interaction_stats(matrix):
    """Calculate statistics for each crew-team pair."""
    stats = {}
    
    for crew, team_data in matrix.items():
        crew_stats = {}
        
        for team, margins in team_data.items():
            if len(margins) < 2:
                continue
            
            avg_margin = statistics.mean(margins)
            std_dev = statistics.stdev(margins)
            min_margin = min(margins)
            max_margin = max(margins)
            
            # Detect bias
            bias = "neutral"
            if avg_margin > 10:
                bias = "heavily_favors"
            elif avg_margin > 5:
                bias = "favors"
            elif avg_margin < -10:
                bias = "heavily_disfavors"
            elif avg_margin < -5:
                bias = "disfavors"
            
            crew_stats[team] = {
                'games': len(margins),
                'avg_margin': avg_margin,
                'std_dev': std_dev,
                'min_margin': min_margin,
                'max_margin': max_margin,
                'range': max_margin - min_margin,
                'bias': bias
            }
        
        stats[crew] = crew_stats
    
    return stats

def find_manipulation_patterns(stats, matrix):
    """Identify specific manipulation patterns."""
    patterns = defaultdict(list)
    
    for crew, team_data in stats.items():
        # Pattern 1: Extreme variance (some teams get blown out, others get favored)
        all_margins = []
        for team, data in team_data.items():
            all_margins.append(data['avg_margin'])
        
        if len(all_margins) > 5:
            variance = statistics.variance(all_margins)
            stdev = statistics.stdev(all_margins)
            
            if stdev > 8:
                patterns[crew].append({
                    'type': 'SELECTIVE_TEAM_BIAS',
                    'severity': stdev,
                    'description': f'Crew treats teams very differently (StdDev: {stdev:.1f})'
                })
        
        # Pattern 2: Extreme blowouts against specific teams
        for team, data in team_data.items():
            if data['games'] >= 2 and abs(data['avg_margin']) > 15:
                patterns[crew].append({
                    'type': 'EXTREME_TEAM_BIAS',
                    'team': team,
                    'margin': data['avg_margin'],
                    'games': data['games'],
                    'description': f'{team}: {data["avg_margin"]:+.1f} avg ({data["games"]} games)'
                })
            
            # Pattern 3: High variance within team (inconsistent handling)
            if data['std_dev'] > 12:
                patterns[crew].append({
                    'type': 'INCONSISTENT_CALLS',
                    'team': team,
                    'variance': data['std_dev'],
                    'range': data['range'],
                    'description': f'{team}: Range {data["min_margin"]:+.0f} to {data["max_margin"]:+.0f} (StdDev: {data["std_dev"]:.1f})'
                })
    
    return patterns

def main():
    data = load_game_records()
    games = data['games']
    
    print(f"\n{'='*150}")
    print("CREW-TEAM INTERACTION MATRIX ANALYSIS")
    print(f"{'='*150}")
    print(f"Total games analyzed: {len(games)}\n")
    
    matrix = build_crew_team_matrix(games)
    stats = calculate_interaction_stats(matrix)
    patterns = find_manipulation_patterns(stats, matrix)
    
    # Group crews by manipulation patterns
    selective_teams = {}
    extreme_biases = defaultdict(list)
    inconsistent = defaultdict(list)
    
    for crew, crew_patterns in patterns.items():
        for pattern in crew_patterns:
            if pattern['type'] == 'SELECTIVE_TEAM_BIAS':
                selective_teams[crew] = pattern
            elif pattern['type'] == 'EXTREME_TEAM_BIAS':
                extreme_biases[crew].append(pattern)
            elif pattern['type'] == 'INCONSISTENT_CALLS':
                inconsistent[crew].append(pattern)
    
    # Display selective team biases
    if selective_teams:
        print(f"\n🎯 CREWS WITH SELECTIVE TEAM BIASES ({len(selective_teams)} found)")
        print("-" * 150)
        for crew in sorted(selective_teams.keys(), key=lambda c: selective_teams[c]['severity'], reverse=True):
            pattern = selective_teams[crew]
            print(f"{crew:20s} | Severity: {pattern['severity']:.2f}")
            
            # Show which teams they favor/disfavor
            team_stats = stats[crew]
            favored = [(t, d['avg_margin']) for t, d in team_stats.items() if d['avg_margin'] > 8]
            disfavored = [(t, d['avg_margin']) for t, d in team_stats.items() if d['avg_margin'] < -8]
            
            if favored:
                favored.sort(key=lambda x: x[1], reverse=True)
                print(f"  FAVORS: {', '.join([f'{t}({m:+.0f})' for t, m in favored[:3]])}")
            if disfavored:
                disfavored.sort(key=lambda x: x[1])
                print(f"  DISFAVORS: {', '.join([f'{t}({m:+.0f})' for t, m in disfavored[:3]])}")
    
    # Display extreme biases
    if extreme_biases:
        print(f"\n💥 EXTREME TEAM BIASES ({sum(len(v) for v in extreme_biases.values())} found)")
        print("-" * 150)
        for crew in sorted(extreme_biases.keys()):
            biases = sorted(extreme_biases[crew], key=lambda x: abs(x['margin']), reverse=True)
            for bias in biases[:3]:  # Top 3 per crew
                direction = "HEAVILY FAVORS" if bias['margin'] > 0 else "CRUSHES"
                print(f"{crew:20s} {direction:20s} {bias['team']:5s}: {bias['margin']:+7.1f} avg ({bias['games']} games)")
    
    # Display inconsistent calls
    if inconsistent:
        print(f"\n⚡ INCONSISTENT CALL PATTERNS ({sum(len(v) for v in inconsistent.values())} found)")
        print("-" * 150)
        for crew in sorted(inconsistent.keys(), key=lambda c: max(p['variance'] for p in inconsistent[c]), reverse=True)[:10]:
            patterns_list = sorted(inconsistent[crew], key=lambda x: x['variance'], reverse=True)
            for pattern in patterns_list[:2]:
                print(f"{crew:20s} vs {pattern['team']:5s}: StdDev {pattern['variance']:5.1f} | "
                      f"Range {pattern['range']:+6.0f} | {pattern['description']}")
    
    # Summary stats
    print(f"\n{'='*150}")
    print("SUMMARY")
    print(f"{'='*150}")
    print(f"Crews analyzed: {len(stats)}")
    print(f"Crews with selective bias: {len(selective_teams)}")
    print(f"Crews with extreme team biases: {len(extreme_biases)}")
    print(f"Crews with inconsistent calls: {len(inconsistent)}")
    
    # Top suspicious crews
    suspicious_count = defaultdict(int)
    for crew, crew_patterns in patterns.items():
        suspicious_count[crew] = len(crew_patterns)
    
    print(f"\nMost suspicious crews:")
    for crew, count in sorted(suspicious_count.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {crew:20s}: {count} manipulation patterns detected")

if __name__ == '__main__':
    main()
