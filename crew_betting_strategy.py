#!/usr/bin/env python3
import re
from collections import defaultdict
import statistics
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REPORTS_DIR = BASE_DIR / "reports" / "referee_conspiracy"

def parse_crew_data(directory):
    """Parse all team files and extract crew statistics."""
    crew_data = defaultdict(list)
    
    directory_path = Path(directory)
    for filepath in sorted(directory_path.glob("*.md")):
        if not filepath.is_file():
            continue
        with filepath.open('r') as f:
            content = f.read()
        
        # Extract crew rotation data
        rotation_pattern = r'- \d+: ([^(]+) \((\d+) games; weeks ([^;]+); avg margin ([+-]?\d+\.?\d*);'
        for match in re.finditer(rotation_pattern, content):
            crew_name = match.group(1).strip()
            games = int(match.group(2))
            margin = float(match.group(4))
            
            crew_data[crew_name].append(margin)
    
    return crew_data

def classify_crews(crew_data):
    """Classify crews by betting strategy."""
    strategies = {
        'blowout_play': [],      # Favors large margins
        'under_play': [],         # Favors close/low-scoring games
        'over_play': [],          # Favors wide margins
        'prop_play': [],          # Specific prop opportunities
        'balanced': []            # No clear pattern
    }
    
    for crew_name, margins in crew_data.items():
        if len(margins) < 5:
            continue
        
        avg_margin = statistics.mean(margins)
        std_dev = statistics.stdev(margins)
        
        blowout_count = len([m for m in margins if abs(m) > 20])
        blowout_pct = (blowout_count / len(margins)) * 100
        
        close_count = len([m for m in margins if -3 <= m <= 3])
        close_pct = (close_count / len(margins)) * 100
        
        # Classify
        if blowout_pct > 15:
            strategies['blowout_play'].append({
                'crew': crew_name,
                'games': len(margins),
                'blowout_pct': blowout_pct,
                'avg_margin': avg_margin,
                'std_dev': std_dev,
                'edge': 'Play OVERS, target -7.5 or wider totals'
            })
        elif close_pct > 40:
            strategies['under_play'].append({
                'crew': crew_name,
                'games': len(margins),
                'close_pct': close_pct,
                'avg_margin': avg_margin,
                'std_dev': std_dev,
                'edge': 'Play UNDERS, target lower totals'
            })
        elif std_dev > 16:
            strategies['over_play'].append({
                'crew': crew_name,
                'games': len(margins),
                'std_dev': std_dev,
                'avg_margin': avg_margin,
                'edge': 'High variance - play OVERS on totals'
            })
        else:
            strategies['balanced'].append({
                'crew': crew_name,
                'games': len(margins),
                'avg_margin': avg_margin,
                'std_dev': std_dev,
                'edge': 'Neutral - follow the line'
            })
    
    return strategies

def main():
    directory = DEFAULT_REPORTS_DIR
    crew_data = parse_crew_data(directory)
    strategies = classify_crews(crew_data)
    
    print("\n" + "="*100)
    print("NFL REFEREE CREW BETTING STRATEGIES (2018-2024)")
    print("="*100)
    
    print("\n🔴 BLOWOUT CREWS - Play OVERS & Wide Totals")
    print("-" * 100)
    if strategies['blowout_play']:
        for c in sorted(strategies['blowout_play'], key=lambda x: x['blowout_pct'], reverse=True):
            print(f"{c['crew']:20s} | {c['games']:3d} games | {c['blowout_pct']:5.1f}% blowouts | StdDev: {c['std_dev']:5.1f}")
            print(f"  → {c['edge']}")
    else:
        print("  No blowout crews identified")
    
    print("\n🟡 HIGH VARIANCE CREWS - Play OVERS")
    print("-" * 100)
    if strategies['over_play']:
        for c in sorted(strategies['over_play'], key=lambda x: x['std_dev'], reverse=True):
            print(f"{c['crew']:20s} | {c['games']:3d} games | StdDev: {c['std_dev']:5.1f} | AvgMargin: {c['avg_margin']:+6.1f}")
            print(f"  → {c['edge']}")
    else:
        print("  No high-variance crews identified")
    
    print("\n🟢 TIGHT GAME CREWS - Play UNDERS")
    print("-" * 100)
    if strategies['under_play']:
        for c in sorted(strategies['under_play'], key=lambda x: x['close_pct'], reverse=True):
            print(f"{c['crew']:20s} | {c['games']:3d} games | {c['close_pct']:5.1f}% close games")
            print(f"  → {c['edge']}")
    else:
        print("  No tight-game crews identified")
    
    print("\n⚪ BALANCED CREWS - Follow the Line")
    print("-" * 100)
    print(f"Total balanced crews: {len(strategies['balanced'])}")
    if len(strategies['balanced']) <= 5:
        for c in sorted(strategies['balanced'], key=lambda x: x['games'], reverse=True):
            print(f"  {c['crew']:20s} | {c['games']:3d} games | AvgMargin: {c['avg_margin']:+6.1f}")
    else:
        print("  (Too many to list individually)")
    
    # Summary
    print("\n" + "="*100)
    print("QUICK REFERENCE")
    print("="*100)
    print("\n1. CHECK WHO'S OFFICIATING")
    print("2. IF BLOWOUT CREW → Play OVERS / Look for line value on wide spreads")
    print("3. IF HIGH VARIANCE → Play OVERS / Exploit bigger totals")
    print("4. IF TIGHT GAME CREW → Play UNDERS / Target lower totals")
    print("5. IF BALANCED → Follow standard analysis, crew has minimal impact")
    
    total_crews = sum(len(v) for v in strategies.values())
    print(f"\nTotal crews analyzed: {total_crews}")

if __name__ == '__main__':
    main()
