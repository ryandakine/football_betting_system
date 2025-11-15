#!/usr/bin/env python3
"""
UNIFIED BET SELECTOR - Multi-Sport Pattern Integration
Combines:
1. NFL division × time slot patterns (comprehensive analysis)
2. NCAA conference × day × week patterns (comprehensive analysis)
3. Claude's world model edges (early season, non-conference, key numbers)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class UnifiedBetSelector:
    """Multi-sport bet selector with all discovered patterns"""
    
    # Betting tier system based on Claude's world model edges
    BETTING_TIERS = {
        'TIER_1_MEGA_EDGE': {
            'win_rate': 0.88,
            'multiplier': 1.50,
            'bet_size': 0.025,
            'description': 'Power 5 + W1-2 + Non-Conf (HIGHEST CONFIDENCE)'
        },
        'TIER_2_SUPER_EDGE': {
            'win_rate': 0.81,
            'multiplier': 1.35,
            'bet_size': 0.020,
            'description': 'Early Season + Non-Conf (VERY HIGH CONFIDENCE)'
        },
        'TIER_3_STRONG_EDGE': {
            'win_rate': 0.73,
            'multiplier': 1.25,
            'bet_size': 0.015,
            'description': 'Big Ten/Mountain West Home + Conference (HIGH CONFIDENCE)'
        },
        'TIER_4_MODERATE_EDGE': {
            'win_rate': 0.65,
            'multiplier': 1.15,
            'bet_size': 0.012,
            'description': 'Mixed patterns (MODERATE CONFIDENCE)'
        },
        'TIER_5_SELECTIVE': {
            'win_rate': 0.58,
            'multiplier': 1.05,
            'bet_size': 0.010,
            'description': 'Late season / Weak patterns (SELECTIVE ONLY)'
        }
    }
    
    def __init__(self, ncaa_file=None, nfl_file=None, sport='ncaa'):
        self.sport = sport.lower()
        self.predictions = []
        self.patterns = {}
        self.betting_tier = None
        
        # Load predictions
        if sport == 'nfl' and nfl_file:
            self._load_predictions(nfl_file)
            self._load_nfl_patterns()
        elif sport == 'ncaa' and ncaa_file:
            self._load_predictions(ncaa_file)
            self._load_ncaa_patterns()
    
    def _load_predictions(self, filepath):
        """Load prediction data"""
        try:
            with open(filepath) as f:
                self.predictions = json.load(f)
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            self.predictions = []
    
    def _load_ncaa_patterns(self):
        """Load NCAA-specific patterns"""
        self.patterns = {
            'claude_edges': {
                'mega_edge': {
                    'criteria': ['power5', 'week_1_2', 'non_conference', 'not_neutral'],
                    'multiplier': 1.50,  # 88% win rate → 1.50x boost
                    'confidence': 0.88,
                    'description': 'Power 5 + Early + Non-Conf'
                },
                'super_edge': {
                    'criteria': ['week_1_2', 'non_conference', 'not_neutral'],
                    'multiplier': 1.35,  # 81% win rate → 1.35x boost
                    'confidence': 0.81,
                    'description': 'Early Season + Non-Conf'
                },
                'early_season': {
                    'criteria': ['week_1_2'],
                    'multiplier': 1.25,  # 81% early season
                    'confidence': 0.81,
                    'description': 'Early Season Home Advantage'
                }
            },
            'conference_patterns': {
                'Big Ten': {
                    'friday_home': 1.40,        # 100% win rate
                    'general_home': 1.25,       # 82% win rate
                    'conference_game': 1.15,
                    'neutral_penalty': 0.85
                },
                'Mountain West': {
                    'home': 1.22,               # 80% win rate
                    'early_season': 1.30,
                    'week_1_2': 1.40           # 100% W1-2
                },
                'SEC': {
                    'home': 1.15,               # 73% win rate
                    'early_season': 1.25,
                    'early_week_1': 1.38       # 94% W1
                },
                'ACC': {
                    'home': 1.10,               # 63% win rate
                    'penalty': 0.95
                },
                'Pac-12': {
                    'home': 0.92,               # 54% win rate (weak)
                    'penalty': 0.90
                }
            },
            'temporal_patterns': {
                'week_decay': {
                    1: 1.40,      # +40% boost
                    2: 1.35,      # +35% boost
                    3: 1.20,      # +20% boost
                    4: 1.15,      # +15% boost
                    5: 1.10,      # +10% boost
                    6: 1.05,      # +5% boost
                    7: 1.00,      # neutral
                    8: 0.98,      # -2% penalty
                    9: 0.95,      # -5% penalty
                    10: 0.85      # -15% penalty (collapse)
                },
                'day_of_week': {
                    'Sunday': 1.22,
                    'Thursday': 1.22,
                    'Friday': 1.18,
                    'Saturday': 1.15,
                    'Tuesday': 0.75,   # weak
                    'Wednesday': 0.80  # weak
                }
            },
            'key_numbers': {
                3: 1.08,   # 10% of games
                7: 1.06    # 8% of games
            },
            'matchup_types': {
                'non_conference': 1.30,  # 81% win rate
                'conference': 0.85       # 57% win rate (no edge)
            },
            'neutral_site_penalty': 0.65  # 42.9% → 0.65x
        }
    
    def _load_nfl_patterns(self):
        """Load NFL-specific patterns"""
        self.patterns = {
            'division_time_interactions': {
                'AFC East evening': 1.32,         # 87.5% (but late night is 100%)
                'AFC East late_night': 1.40,      # 100% edge
                'AFC South early_game': 1.40,     # 100% edge
                'AFC West evening': 1.38,         # 87.5% edge
                'AFC West afternoon': 0.92,       # 0% (avoid)
                'NFC East evening': 0.93,         # 33.3% (strong fade)
                'NFC East afternoon': 0.92,       # 0% (avoid)
                'NFC South early_afternoon': 0.94,# 37.5% (weak)
                'AFC North general': 1.00,
                'NFC North general': 1.00,
                'NFC West general': 1.00
            },
            'thursday_pattern': 1.30,             # +22.7% edge
            'time_slot_general': {
                'early_game': 1.15,
                'late_morning': 1.05,
                'afternoon': 0.92,                # 25% (weakest)
                'evening': 1.08,
                'prime_time': 1.20
            },
            'week_patterns': {
                'week_1_2': 1.15,
                'week_3_4': 1.18,                 # Peak performance
                'week_5_8': 1.05,
                'week_9': 0.92,                   # 28.6% (collapse)
                'week_11': 0.75                   # 6.7% (critical away bias)
            },
            'late_season_away_bias': {
                9: 0.85,
                10: 0.80,
                11: 0.50   # Extreme away bias
            }
        }
    
    def calculate_ncaa_score(self, game):
        """Calculate bet score for NCAA game"""
        score = 1.0
        reasons = []
        
        # Extract game info
        week = game.get('week', 7)
        home_team = game.get('predicted_winner', '')
        is_conference = game.get('is_conference_game', False)
        neutral = game.get('neutral_site', False)
        day = game.get('day_of_week', 'Saturday')
        spread = game.get('spread', 0)
        
        # Identify Power 5
        power_5_list = ['SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac']
        is_power_5 = any(p5 in home_team for p5 in power_5_list)
        
        # MEGA EDGE: Power 5 + W1-2 + Non-conference
        if is_power_5 and week <= 2 and not is_conference and not neutral:
            score *= self.patterns['claude_edges']['mega_edge']['multiplier']
            reasons.append("🚀 MEGA EDGE: P5 Early Non-Conf (+50%)")
        
        # SUPER EDGE: W1-2 + Non-conference
        elif week <= 2 and not is_conference and not neutral:
            score *= self.patterns['claude_edges']['super_edge']['multiplier']
            reasons.append("🔥 SUPER EDGE: Early Non-Conf (+35%)")
        
        # Early season boost
        elif week <= 2:
            score *= self.patterns['claude_edges']['early_season']['multiplier']
            reasons.append("✅ Early Season Boost (+25%)")
        
        # Week decay
        if week in self.patterns['temporal_patterns']['week_decay']:
            score *= self.patterns['temporal_patterns']['week_decay'][week]
            if week <= 2:
                reasons.append(f"Week {week} multiplier: +{(self.patterns['temporal_patterns']['week_decay'][week]-1)*100:.0f}%")
            elif week >= 8:
                reasons.append(f"Week {week} penalty: {(self.patterns['temporal_patterns']['week_decay'][week]-1)*100:.0f}%")
        
        # Day of week
        if day in self.patterns['temporal_patterns']['day_of_week']:
            dow_mult = self.patterns['temporal_patterns']['day_of_week'][day]
            score *= dow_mult
            if dow_mult > 1.1:
                reasons.append(f"{day}: +{(dow_mult-1)*100:.0f}%")
            elif dow_mult < 0.9:
                reasons.append(f"{day}: {(dow_mult-1)*100:.0f}%")
        
        # Conference-specific
        for conf in self.patterns['conference_patterns']:
            if conf in home_team:
                conf_mult = self.patterns['conference_patterns'][conf].get('general_home', 1.0)
                score *= conf_mult
                reasons.append(f"{conf} home: {(conf_mult-1)*100:+.0f}%")
                break
        
        # Matchup type
        if not is_conference:
            score *= self.patterns['matchup_types']['non_conference']
            reasons.append("Non-Conference: +30%")
        else:
            score *= self.patterns['matchup_types']['conference']
            reasons.append("Conference: -15% (edge disappears)")
        
        # Neutral site penalty
        if neutral:
            score *= self.patterns['neutral_site_penalty']
            reasons.append("Neutral Site: -35% (no home advantage)")
        
        # Key numbers
        if spread:
            try:
                abs_spread = abs(float(spread))
                if abs_spread == 3.0:
                    score *= self.patterns['key_numbers'][3]
                    reasons.append("Key number 3: +8%")
                elif abs_spread == 7.0:
                    score *= self.patterns['key_numbers'][7]
                    reasons.append("Key number 7: +6%")
            except:
                pass
        
        return score, reasons
    
    def calculate_nfl_score(self, game):
        """Calculate bet score for NFL game"""
        score = 1.0
        reasons = []
        
        # Extract game info
        week = game.get('week', 11)
        home_team = game.get('predicted_winner', '')
        home_division = game.get('home_division', '')
        time_slot = game.get('time_slot', 'unknown')
        day = game.get('day_of_week', 'Sunday')
        
        # Division × Time interaction
        div_time_key = f"{home_division} {time_slot}"
        if div_time_key in self.patterns['division_time_interactions']:
            score *= self.patterns['division_time_interactions'][div_time_key]
            mult = self.patterns['division_time_interactions'][div_time_key]
            reasons.append(f"{div_time_key}: {(mult-1)*100:+.0f}%")
        elif home_division in self.patterns['division_time_interactions']:
            score *= self.patterns['division_time_interactions'][home_division]
        
        # Thursday bonus
        if day == 'Thursday' and time_slot in ['evening', 'prime_time']:
            score *= self.patterns['thursday_pattern']
            reasons.append(f"Thursday night: +{(self.patterns['thursday_pattern']-1)*100:.0f}%")
        
        # Time slot general
        if time_slot in self.patterns['time_slot_general']:
            score *= self.patterns['time_slot_general'][time_slot]
        
        # Week patterns
        if week in self.patterns['week_patterns']:
            score *= self.patterns['week_patterns'][week]
            mult = self.patterns['week_patterns'][week]
            reasons.append(f"Week {week}: {(mult-1)*100:+.0f}%")
        
        # Late season away bias
        if week in self.patterns['late_season_away_bias']:
            score *= self.patterns['late_season_away_bias'][week]
            reasons.append(f"Week {week} away bias: {(self.patterns['late_season_away_bias'][week]-1)*100:.0f}%")
        
        return score, reasons
    
    def assign_tier(self, score, reasons):
        """Assign betting tier based on score and pattern detection"""
        if any('MEGA EDGE' in r for r in reasons):
            return 'TIER_1_MEGA_EDGE'
        elif any('SUPER EDGE' in r for r in reasons):
            return 'TIER_2_SUPER_EDGE'
        elif score >= 1.45:
            return 'TIER_3_STRONG_EDGE'
        elif score >= 1.20:
            return 'TIER_4_MODERATE_EDGE'
        else:
            return 'TIER_5_SELECTIVE'
    
    def select_top_10(self):
        """Select top 10 bets across all patterns"""
        if not self.predictions:
            return []
        
        scored = []
        for pred in self.predictions:
            if pred.get('actual_result') is not None:
                continue
            
            if self.sport == 'ncaa':
                score, reasons = self.calculate_ncaa_score(pred)
            else:
                score, reasons = self.calculate_nfl_score(pred)
            
            if score > 0:
                tier = self.assign_tier(score, reasons)
                tier_info = self.BETTING_TIERS[tier]
                
                scored.append({
                    'game': pred.get('game', 'Unknown'),
                    'score': score,
                    'reasons': reasons,
                    'confidence': pred.get('calibrated_confidence', 0),
                    'edge': pred.get('edge', 0),
                    'tier': tier,
                    'tier_info': tier_info
                })
        
        # Sort and return top 10
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:10]
    
    def format_output(self, top_10):
        """Display results with tier-based betting guidance"""
        if not top_10:
            print("\n⚠️  No valid bets available")
            return
        
        print("\n" + "="*140)
        print(f"🎯 TOP 10 BETS - {self.sport.upper()} (UNIFIED PATTERN ANALYSIS WITH CLAUDE'S WORLD MODEL EDGES)")
        print("="*140)
        
        # Group by tier
        tier_groups = defaultdict(list)
        for bet in top_10:
            tier_groups[bet['tier']].append(bet)
        
        # Display by tier
        tier_order = ['TIER_1_MEGA_EDGE', 'TIER_2_SUPER_EDGE', 'TIER_3_STRONG_EDGE', 
                      'TIER_4_MODERATE_EDGE', 'TIER_5_SELECTIVE']
        
        bet_num = 1
        for tier_name in tier_order:
            if tier_name not in tier_groups:
                continue
            
            tier_info = self.BETTING_TIERS[tier_name]
            print(f"\n{tier_name.replace('_', ' ')}")
            print(f"  📊 Expected Win Rate: {tier_info['win_rate']*100:.0f}%  |  Multiplier: {tier_info['multiplier']}×  |  Bet Size: {tier_info['bet_size']*100:.1f}% bankroll")
            print(f"  Description: {tier_info['description']}")
            print("-" * 140)
            print(f"{'#':<3} {'Game':<45} {'Score':<8} {'Confidence':<12} {'Edge Factors':<60}")
            print("-" * 140)
            
            for bet in tier_groups[tier_name]:
                game = bet['game'][:44]
                score = f"{bet['score']:.2f}"
                conf = f"{bet['confidence']*100:.0f}%"
                factors = " | ".join(bet['reasons'][:3])[:58]
                
                print(f"{bet_num:<3} {game:<45} {score:<8} {conf:<12} {factors:<60}")
                bet_num += 1
        
        print("\n" + "="*140)
        print("✅ INTEGRATED ANALYSIS:")
        print(f"   • NFL Patterns: Division×Time interactions, Thursday bonus, week progression, late-season away bias")
        print(f"   • NCAA Patterns: Conference×day×week interactions, early season edge, non-conference boost, key numbers")
        print(f"   • Claude's World Model: MEGA EDGE detection (88% win rate), SUPER EDGE (81%), tier-based sizing")
        print("="*140)

def main():
    sport = sys.argv[1].lower() if len(sys.argv) > 1 else 'ncaa'
    
    if sport == 'nfl':
        pred_file = 'data/predictions/nfl_prediction_log.json'
        selector = UnifiedBetSelector(nfl_file=pred_file, sport='nfl')
    else:
        pred_file = 'data/predictions/prediction_log.json'
        selector = UnifiedBetSelector(ncaa_file=pred_file, sport='ncaa')
    
    print(f"🤖 Unified Bet Selector - {sport.upper()}")
    print(f"📊 Loading predictions from: {pred_file}")
    
    if not selector.predictions:
        print("❌ No predictions found")
        sys.exit(1)
    
    print(f"✅ Loaded {len(selector.predictions)} predictions")
    print("📈 Calculating unified scores (NFL patterns + NCAA patterns + Claude edges)...")
    
    top_10 = selector.select_top_10()
    selector.format_output(top_10)

if __name__ == '__main__':
    main()
