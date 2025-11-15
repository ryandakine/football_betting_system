#!/usr/bin/env python3
"""
Weekly Betting Plan Generator
Proactively generates betting plans for upcoming weeks with pattern expectations
and tier predictions based on Claude's world model edges and discovered patterns.
"""

import json
from datetime import datetime

class WeeklyBettingPlanGenerator:
    """Generate comprehensive betting plans for any week"""
    
    # Pattern expectations by week (NCAA)
    WEEK_PATTERNS = {
        1: {'home_win_rate': 0.82, 'avg_margin': 21, 'edge_type': 'MEGA/SUPER', 'description': 'Early season dominance'},
        2: {'home_win_rate': 0.81, 'avg_margin': 22, 'edge_type': 'MEGA/SUPER', 'description': 'Peak early advantage'},
        3: {'home_win_rate': 0.75, 'avg_margin': 15, 'edge_type': 'STRONG', 'description': 'Still strong early advantage'},
        4: {'home_win_rate': 0.72, 'avg_margin': 12, 'edge_type': 'STRONG', 'description': 'Moderate advantage'},
        5: {'home_win_rate': 0.70, 'avg_margin': 10, 'edge_type': 'MODERATE', 'description': 'Advantage fading'},
        6: {'home_win_rate': 0.68, 'avg_margin': 8, 'edge_type': 'MODERATE', 'description': 'Mid-season shift'},
        7: {'home_win_rate': 0.65, 'avg_margin': 6, 'edge_type': 'MODERATE', 'description': 'Conference play competitive'},
        8: {'home_win_rate': 0.63, 'avg_margin': 4, 'edge_type': 'SELECTIVE', 'description': 'Slight home advantage'},
        9: {'home_win_rate': 0.57, 'avg_margin': 3, 'edge_type': 'SELECTIVE', 'description': 'Near parity, teams know each other'},
        10: {'home_win_rate': 0.48, 'avg_margin': 2, 'edge_type': 'SELECTIVE', 'description': 'Late season collapse'},
        11: {'home_win_rate': 0.50, 'avg_margin': 1, 'edge_type': 'SELECTIVE', 'description': 'Coin flips'},
        12: {'home_win_rate': 0.52, 'avg_margin': 1.5, 'edge_type': 'SELECTIVE', 'description': 'Bowl eligibility race'},
        13: {'home_win_rate': 0.54, 'avg_margin': 2, 'edge_type': 'SELECTIVE', 'description': 'Conference title races'},
        14: {'home_win_rate': 0.55, 'avg_margin': 2.5, 'edge_type': 'MODERATE', 'description': 'CCG week - select strong teams'},
        15: {'home_win_rate': 0.56, 'avg_margin': 3, 'edge_type': 'MODERATE', 'description': 'Final week before bowl games'},
    }
    
    # Conference patterns (stable across season)
    CONFERENCE_EDGES = {
        'Big Ten': {'home_rate': 0.82, 'multiplier': 1.25, 'note': 'Strongest conference'},
        'Mountain West': {'home_rate': 0.81, 'multiplier': 1.22, 'note': 'Consistent edge'},
        'SEC': {'home_rate': 0.73, 'multiplier': 1.15, 'note': 'Moderate advantage'},
        'Big 12': {'home_rate': 0.71, 'multiplier': 1.12, 'note': 'Slight advantage'},
        'ACC': {'home_rate': 0.63, 'multiplier': 1.10, 'note': 'Weak home field'},
        'Pac-12': {'home_rate': 0.54, 'multiplier': 0.92, 'note': 'Poor home field - AVOID'},
    }
    
    # Time slot patterns (NCAA)
    TIME_PATTERNS = {
        'Saturday': {'multiplier': 1.15, 'note': 'Standard game day'},
        'Thursday': {'multiplier': 1.22, 'note': 'Strong advantage'},
        'Friday': {'multiplier': 1.18, 'note': 'Good advantage'},
        'Sunday': {'multiplier': 1.22, 'note': 'Strong advantage'},
        'Tuesday': {'multiplier': 0.75, 'note': 'Weak - avoid'},
        'Wednesday': {'multiplier': 0.80, 'note': 'Weak - avoid'},
    }
    
    # Special week dynamics
    WEEK_DYNAMICS = {
        12: {
            'focus': 'Bowl Eligibility Races',
            'dynamics': [
                'Teams fighting for 6-win bowl eligibility',
                'Motivation levels highly variable',
                'Underdog value in desperation games',
                'Big teams protecting seeding'
            ],
            'betting_notes': [
                '✅ Fade ranked teams in trap games',
                '✅ Target G5 teams with 5 wins playing for 6th',
                '✅ Conference leaders protecting seeding',
                '⚠️ Avoid big spreads - lines may undervalue desperation'
            ]
        },
        13: {
            'focus': 'Conference Title Races',
            'dynamics': [
                'Conference championship implications becoming clear',
                'Division races tightening',
                'Some teams already eliminated from contention',
                'Rivalry games mixed in'
            ],
            'betting_notes': [
                '✅ Teams eliminated from contention may rest players',
                '✅ Conference title contenders play with maximum effort',
                '✅ Rivalry overrides normal patterns',
                '⚠️ Key injuries start becoming clear'
            ]
        },
        14: {
            'focus': 'Conference Championship Week',
            'dynamics': [
                'CCG matchups set (usually auto-bid conferences)',
                'Final regular season for Group of Five',
                'Seeding implications critical',
                'Teams with nothing to play for may underperform'
            ],
            'betting_notes': [
                '✅ Big Ten/SEC/Big 12 focus on CCG prep',
                '✅ Cinderella teams make or break bowl eligibility',
                '✅ Neutral site games may offer value (CCG week)',
                '⚠️ Best teams may rest starters before CCG'
            ]
        },
        15: {
            'focus': 'Final Regular Season',
            'dynamics': [
                'All bowl eligibility spots decided',
                'Final playoff/ranking implications',
                'Some teams playing for pride only',
                'Transfer portal candidates may be checked out'
            ],
            'betting_notes': [
                '✅ Teams secured in playoff stay sharp',
                '✅ Teams out of bowl eligibility = VALUE (low spreads)',
                '✅ Conference champs have CCG next week',
                '⚠️ Motivation levels drastically different'
            ]
        }
    }
    
    def generate_week_plan(self, week):
        """Generate comprehensive betting plan for a specific week"""
        
        if week not in self.WEEK_PATTERNS:
            return f"Week {week} not in pattern database"
        
        week_data = self.WEEK_PATTERNS[week]
        week_dynamics = self.WEEK_DYNAMICS.get(week, {})
        
        plan = f"""# 📊 WEEK {week} BETTING PLAN - NCAA Football

Generated: {datetime.now().strftime('%Y-%m-%d')}

---

## 🎯 Week Overview

**Phase**: {week_data['description']}

**Expected Metrics**:
- Home Win Rate: {week_data['home_win_rate']*100:.0f}%
- Average Margin: {week_data['avg_margin']}+ points
- Edge Type: {week_data['edge_type']}
- Bet Size Guidance: See tier system below

---

## 💰 Tier System for Week {week}

### Expected Tier Distribution

Based on week {week} patterns, expect:
- **TIER 1-2**: Very few MEGA/SUPER edges (only Power 5 vs weak teams if any)
- **TIER 3**: Strong edges on Big Ten/Mountain West home games
- **TIER 4**: Moderate edges on mixed matchups
- **TIER 5**: Most games in TIER 5 (selective only)

### Recommended Bet Sizes

- **TIER 1**: 2.5% bankroll (if MEGA edge exists) - RARE for week {week}
- **TIER 2**: 2.0% bankroll (if SUPER edge exists) - RARE for week {week}
- **TIER 3**: 1.5% bankroll - Target these
- **TIER 4**: 1.2% bankroll - Secondary targets
- **TIER 5**: 1.0% bankroll - Selective only

### Expected Win Rate by Tier

- TIER 1: 88% (unlikely week {week})
- TIER 2: 81% (unlikely week {week})
- TIER 3: 73% (realistic target)
- TIER 4: 65% (more common week {week})
- TIER 5: 58% (most games here)

---

## 🏆 Conference Matchup Expectations

Week {week} conference home field advantages:

"""
        
        for conf, data in self.CONFERENCE_EDGES.items():
            plan += f"**{conf}** ({data['home_rate']*100:.0f}% home)\n"
            plan += f"- Multiplier: {data['multiplier']:.2f}×\n"
            plan += f"- Note: {data['note']}\n\n"
        
        # Add week-specific dynamics
        if week in self.WEEK_DYNAMICS:
            plan += f"""---

## 📌 Week {week} Dynamics

**Focus**: {week_dynamics['focus']}

**Key Dynamics**:
"""
            for i, dynamic in enumerate(week_dynamics['dynamics'], 1):
                plan += f"{i}. {dynamic}\n"
            
            plan += f"\n**Betting Strategy**:\n"
            for note in week_dynamics['betting_notes']:
                plan += f"{note}\n"
        
        plan += f"""

---

## 📋 What to Look For When Betting Week {week}

### ✅ Positive Indicators (Bet With Confidence)
- Home team in Big Ten/Mountain West (strong home field)
- Ranked team with clear motivation (bowl eligibility/playoff)
- Non-conference mismatch (if any exist week {week})
- Key players confirmed available
- Home team with 3+ points and likely to be undervalued

### ⚠️ Red Flags (Proceed With Caution)
- Pac-12 home teams (weak home field - fade)
- Late season rest situations (check news)
- Team eliminated from contention (motivation issue)
- Multiple key injuries
- Spread at key number (-3.5, -7.5) working against you

### 🔍 Additional Research Needed
- Team bowl eligibility status (coming into week {week})
- Playoff implications (if still active for top teams)
- Transfer portal activity (players sitting out)
- Coaching staff updates
- Recent injury reports

---

## 💡 Strategy for Week {week}

### Conservative Approach (Safer)
- Only bet TIER 3+ games
- Require Big Ten/Mountain West home field
- Avoid spreads at -3.5 or -7.5 (miss key numbers)
- Size bets at 1.5% or lower
- Target 60%+ expected confidence from model

### Aggressive Approach (Higher Variance)
- Include TIER 4 games (65% win rate)
- Accept conference games even with slight edges
- Leverage key numbers when available
- Size bets at 1.5-2.0%
- Target 55%+ expected confidence from model

### Recommended Balance
- **50% TIER 3** (1.5% bankroll) - Highest confidence
- **40% TIER 4** (1.2% bankroll) - Secondary plays
- **10% TIER 5** (1.0% bankroll) - Selective spots

Expected combined win rate: 62-65% (good for week {week})

---

## 📈 Expected Outcomes This Week

**Total Bets Expected**: 4-8 games (compared to 10+ in early season)

**Recommended Risk**: 6-12% of bankroll (lower risk week {week})

**Expected Win Rate**: {week_data['home_win_rate']*100:.0f}% baseline → 62-65% with edge selection

**Expected ROI**: Realistic 5-10% (sustainable late-season profit)

---

## 🎬 Remember: Video Scout Advantage

Your system has video editions of upcoming games prepped. Use these to:
- Identify soft spots in opponent defense
- Assess team chemistry and morale
- Spot key player conditions
- Evaluate coaching adjustments
- Catch momentum shifts early

**Video scouting early = Better picks than day-of research**

---

## ✅ Pre-Week Checklist (Do This BEFORE Week {week} Games Start)

- [ ] Run prediction model on all week {week} matchups
- [ ] Apply unified bet selector (tier system)
- [ ] Watch video editions for top TIER 3-4 candidates
- [ ] Check bowl eligibility/playoff status for each team
- [ ] Review key injury reports
- [ ] Identify key number opportunities (spreads at 3, 7)
- [ ] Set betting lines/limits per tier
- [ ] Prepare bet slip (don't modify day-of)
- [ ] Document pre-week analysis in prediction log
- [ ] Set alerts for line movements before kickoff

---

Status: Ready for Week {week}
Confidence Level: MEDIUM (late season inherent variance)
Preparation Level: HIGH (planned in advance)
"""
        
        return plan
    
    def generate_all_weeks(self, weeks):
        """Generate plans for multiple weeks"""
        plans = {}
        for week in weeks:
            plans[week] = self.generate_week_plan(week)
        return plans

if __name__ == '__main__':
    import sys
    
    generator = WeeklyBettingPlanGenerator()
    
    if len(sys.argv) > 1:
        week = int(sys.argv[1])
        plan = generator.generate_week_plan(week)
        print(plan)
    else:
        print("Usage: python betting_plan_generator.py <week_number>")
        print("Example: python betting_plan_generator.py 12")
