#!/usr/bin/env python3
import json
import statistics
from collections import defaultdict

def load_referee_data():
    """Load the referee autopsy JSON data."""
    with open('/home/ryan/code/football_betting_system/data/referee_conspiracy/referee_autopsy.json', 'r') as f:
        return json.load(f)

def analyze_crews(data):
    """Deep analysis of crew patterns using game-level data."""
    results = []
    
    for crew_name, season_data in data['referees'].items():
        # Aggregate data across all seasons/windows
        all_games = 0
        all_margins = []
        all_penalties = []
        all_points = []
        all_overtime = []
        all_flag_density = []
        
        for window_key, window_data in season_data.items():
            games = window_data.get('games', 0)
            all_games += games
            
            # Extend lists with data
            for _ in range(games):
                all_margins.append(window_data.get('margin_avg', 0))
                all_penalties.append(window_data.get('penalties_avg', 0))
                all_points.append(window_data.get('points_avg', 0))
                all_overtime.append(window_data.get('overtime_rate', 0))
                all_flag_density.append(window_data.get('flag_density_avg', 0))
        
        if all_games < 10:
            continue
        
        # Calculate statistics
        avg_margin = statistics.mean(all_margins)
        avg_penalties = statistics.mean(all_penalties)
        avg_points = statistics.mean(all_points)
        avg_overtime = statistics.mean(all_overtime) * 100  # Convert to percentage
        avg_flag_density = statistics.mean(all_flag_density)
        
        std_margin = statistics.stdev(all_margins) if len(all_margins) > 1 else 0
        
        # Seasonal variance (early vs late)
        early_margin = season_data.get('early', {}).get('margin_avg', 0)
        late_margin = season_data.get('late', {}).get('margin_avg', 0)
        seasonal_shift = abs(late_margin - early_margin)
        
        # Crew classification
        crew_type = "BALANCED"
        traits = []
        
        if abs(avg_margin) > 12:
            crew_type = "BLOWOUT"
            traits.append(f"Avg margin {avg_margin:+.1f}")
        elif avg_overtime > 10:
            crew_type = "OVERTIME_PRONE"
            traits.append(f"OT rate {avg_overtime:.1f}%")
        elif avg_penalties > 13.5:
            crew_type = "FLAG_HEAVY"
            traits.append(f"Penalties {avg_penalties:.1f}/game")
        elif avg_penalties < 10.5:
            crew_type = "LENIENT"
            traits.append(f"Penalties {avg_penalties:.1f}/game")
        
        # Detect shifting behavior
        if seasonal_shift > 3:
            traits.append(f"Seasonal shift {seasonal_shift:+.1f}")
        
        results.append({
            'crew': crew_name,
            'total_games': all_games,
            'type': crew_type,
            'avg_margin': avg_margin,
            'std_margin': std_margin,
            'avg_penalties': avg_penalties,
            'avg_points': avg_points,
            'avg_overtime': avg_overtime,
            'avg_flag_density': avg_flag_density,
            'early_margin': early_margin,
            'late_margin': late_margin,
            'seasonal_shift': seasonal_shift,
            'traits': ', '.join(traits) if traits else 'Standard'
        })
    
    return results

def main():
    data = load_referee_data()
    results = analyze_crews(data)
    
    # Group by type
    by_type = defaultdict(list)
    for r in results:
        by_type[r['type']].append(r)
    
    # Sort each category
    for t in by_type:
        by_type[t].sort(key=lambda x: abs(x['avg_margin'] if x['type'] == 'BLOWOUT' else x['avg_penalties']), reverse=True)
    
    print("\n" + "="*150)
    print("ADVANCED NFL REFEREE CREW ANALYSIS - Game-by-Game Patterns")
    print("="*150)
    
    # BLOWOUT CREWS
    if by_type['BLOWOUT']:
        print(f"\n🔴 BLOWOUT CREWS ({len(by_type['BLOWOUT'])} found)")
        print("-" * 150)
        for r in by_type['BLOWOUT']:
            print(f"{r['crew']:20s} | Games: {r['total_games']:3d} | Margin: {r['avg_margin']:+7.1f}±{r['std_margin']:5.1f} | "
                  f"Points: {r['avg_points']:5.1f} | Penalties: {r['avg_penalties']:5.1f} | OT: {r['avg_overtime']:5.1f}%")
            print(f"  └─ Early: {r['early_margin']:+.1f} | Late: {r['late_margin']:+.1f} | Shift: {r['seasonal_shift']:+.1f}")
    
    # OVERTIME CREWS
    if by_type['OVERTIME_PRONE']:
        print(f"\n⏱️ OVERTIME-PRONE CREWS ({len(by_type['OVERTIME_PRONE'])} found)")
        print("-" * 150)
        for r in by_type['OVERTIME_PRONE']:
            print(f"{r['crew']:20s} | Games: {r['total_games']:3d} | OT Rate: {r['avg_overtime']:5.1f}% | Penalties: {r['avg_penalties']:5.1f} | "
                  f"Margin: {r['avg_margin']:+7.1f} | Points: {r['avg_points']:5.1f}")
            print(f"  └─ {r['traits']}")
    
    # FLAG HEAVY
    if by_type['FLAG_HEAVY']:
        print(f"\n🚩 FLAG-HEAVY CREWS ({len(by_type['FLAG_HEAVY'])} found)")
        print("-" * 150)
        for r in by_type['FLAG_HEAVY']:
            print(f"{r['crew']:20s} | Games: {r['total_games']:3d} | Penalties: {r['avg_penalties']:5.1f} | Flag Density: {r['avg_flag_density']:.3f} | "
                  f"Margin: {r['avg_margin']:+7.1f}")
            print(f"  └─ {r['traits']}")
    
    # LENIENT
    if by_type['LENIENT']:
        print(f"\n🆓 LENIENT CREWS ({len(by_type['LENIENT'])} found)")
        print("-" * 150)
        for r in by_type['LENIENT']:
            print(f"{r['crew']:20s} | Games: {r['total_games']:3d} | Penalties: {r['avg_penalties']:5.1f} | Flag Density: {r['avg_flag_density']:.3f} | "
                  f"Margin: {r['avg_margin']:+7.1f}")
            print(f"  └─ {r['traits']}")
    
    # BALANCED
    if by_type['BALANCED']:
        print(f"\n⚪ BALANCED CREWS ({len(by_type['BALANCED'])} found)")
        print("-" * 150)
        for r in sorted(by_type['BALANCED'], key=lambda x: x['total_games'], reverse=True):
            print(f"{r['crew']:20s} | Games: {r['total_games']:3d} | Margin: {r['avg_margin']:+7.1f}±{r['std_margin']:5.1f} | "
                  f"Penalties: {r['avg_penalties']:5.1f} | OT: {r['avg_overtime']:5.1f}%")
    
    # SUMMARY
    print("\n" + "="*150)
    print("BREAKDOWN")
    print("="*150)
    total = len(results)
    for t in sorted(by_type.keys()):
        pct = (len(by_type[t]) / total) * 100
        print(f"{t:20s}: {len(by_type[t]):2d} crews ({pct:5.1f}%)")

if __name__ == '__main__':
    main()
