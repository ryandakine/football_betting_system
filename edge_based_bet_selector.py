#!/usr/bin/env python3
"""
Edge-Based Bet Selector - Uses World Model Patterns
Implements the discovered edges from deep NCAA analysis
"""

import json
from datetime import datetime
from collections import defaultdict

class EdgeBasedBetSelector:
    def __init__(self, predictions_file, bankroll=80):
        with open(predictions_file) as f:
            self.predictions = json.load(f)
        self.bankroll = bankroll
        
    def calculate_edge_score(self, game):
        """
        World model edge scoring based on discovered patterns
        Returns: (edge_score, edge_reasons)
        """
        score = 0
        reasons = []
        
        # Extract game features
        week = game.get('week', 11)  # Default to current week
        home_team = game.get('predicted_winner', '')
        is_conference = game.get('is_conference_game', False)
        neutral = game.get('neutral_site', False)
        conference = game.get('conference', '')
        
        # Identify if Power 5
        power_5 = ['SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12']
        is_power_5 = any(p5 in home_team or conference == p5 for p5 in power_5)
        
        # MEGA EDGE: Power 5 + Early Season + Non-Conference
        # 88% win rate, +27.1 point margin
        if is_power_5 and week <= 2 and not is_conference and not neutral:
            score += 40
            reasons.append("🚀 MEGA EDGE: P5 Early Non-Conf (88% win rate)")
        
        # SUPER EDGE: Early Season + Non-Conference
        # 81.7% win rate, +22.1 point margin
        elif week <= 2 and not is_conference and not neutral:
            score += 30
            reasons.append("🔥 SUPER EDGE: Early Non-Conf (81% win rate)")
        
        # Early season general home advantage
        # 81.2% win rate weeks 1-2
        elif week <= 2:
            score += 20
            reasons.append("✅ Early Season Home Advantage (81% win rate)")
        
        # Conference-specific home field strength
        if 'Big Ten' in conference or 'Big Ten' in home_team:
            score += 12
            reasons.append("Big Ten home field (82% win rate)")
        elif 'Mountain West' in conference:
            score += 10
            reasons.append("Mountain West home field (80% win rate)")
        elif conference in ['SEC']:
            score += 8
            reasons.append("SEC home field (73% win rate)")
        elif conference in ['Big 12']:
            score += 7
            reasons.append("Big 12 home field (71% win rate)")
        elif 'Pac-12' in conference or 'Pac-12' in home_team:
            score -= 5
            reasons.append("⚠️ Pac-12 weak home field (54% win rate)")
        
        # Late season conference game penalty
        # Only 55.8% home wins, 2.4 point avg margin
        if week >= 8 and is_conference:
            score -= 12
            reasons.append("⚠️ Late season conference (55% win rate only)")
        
        # Neutral site major penalty
        # Only 42.9% "home" wins
        if neutral:
            score -= 20
            reasons.append("⚠️ Neutral site - no home advantage")
        
        # Non-conference advantage (regardless of week)
        # 81% vs 57% conference
        if not is_conference and week > 2:
            score += 10
            reasons.append("Non-conference mismatch advantage")
        
        return score, reasons
    
    def check_key_number(self, spread):
        """Check if spread is on a key number (3 or 7)"""
        if spread is None:
            return 0, []
        
        abs_spread = abs(spread)
        
        # Getting +3 or +7 is valuable (push protection)
        if abs_spread == 3.0:
            return 5, ["Key number 3 (10% of games)"]
        elif abs_spread == 7.0:
            return 5, ["Key number 7 (8% of games)"]
        
        # Laying -3.5 or -7.5 is dangerous (missing key number)
        elif abs_spread == 3.5:
            return -3, ["⚠️ Just past key 3"]
        elif abs_spread == 7.5:
            return -3, ["⚠️ Just past key 7"]
        
        return 0, []
    
    def select_bets(self, max_bets=10):
        """
        Select bets using world model edge analysis
        """
        scored_bets = []
        
        for pred in self.predictions:
            # Calculate edge score
            edge_score, edge_reasons = self.calculate_edge_score(pred)
            
            # Check key numbers
            spread = pred.get('spread')
            kn_score, kn_reasons = self.check_key_number(spread)
            edge_score += kn_score
            edge_reasons.extend(kn_reasons)
            
            # Get base confidence
            confidence = pred.get('calibrated_confidence', 0)
            
            # Combine edge score with model confidence
            # Edge score can boost or reduce confidence
            final_confidence = confidence + (edge_score / 100)
            final_confidence = max(0.0, min(0.95, final_confidence))
            
            # Calculate strategic value
            strategic_score = (edge_score * 0.6) + (confidence * 100 * 0.4)
            
            scored_bets.append({
                'game': pred['game'],
                'pick': pred['predicted_winner'],
                'spread': spread,
                'base_confidence': confidence,
                'edge_score': edge_score,
                'final_confidence': final_confidence,
                'strategic_score': strategic_score,
                'edge_reasons': edge_reasons,
                'hours_until_game': pred.get('hours_until_game', 999)
            })
        
        # Filter to positive edge only
        scored_bets = [b for b in scored_bets if b['edge_score'] >= 0]
        
        # Sort by strategic score
        scored_bets.sort(key=lambda x: x['strategic_score'], reverse=True)
        
        # Select top bets with portfolio optimization
        selected = self._optimize_portfolio(scored_bets, max_bets)
        
        return selected
    
    def _optimize_portfolio(self, scored_bets, max_bets):
        """Optimize betting portfolio"""
        portfolio = []
        time_slots = defaultdict(int)
        total_risk = 0
        
        for bet in scored_bets:
            if len(portfolio) >= max_bets:
                break
            
            # Kelly sizing based on final confidence
            confidence = bet['final_confidence']
            if confidence < 0.60:
                continue
            
            bet_size = self.bankroll * 0.015 * (confidence / 0.65)
            
            # Boost size for mega edges
            if bet['edge_score'] >= 30:
                bet_size *= 1.5  # 50% larger for mega edges
                
            bet_size = min(bet_size, self.bankroll * 0.03)  # Cap at 3%
            
            # Risk management
            if total_risk + bet_size > self.bankroll * 0.35:
                bet_size = self.bankroll * 0.35 - total_risk
                if bet_size < 0.5:
                    continue
            
            # Time slot management
            hours = bet['hours_until_game']
            time_slot = int(hours / 4) * 4
            if time_slots[time_slot] >= 4:
                continue
            
            bet['bet_size'] = round(bet_size, 2)
            portfolio.append(bet)
            time_slots[time_slot] += 1
            total_risk += bet_size
        
        return portfolio
    
    def display_recommendations(self, portfolio):
        """Display betting plan with edge analysis"""
        print("=" * 80)
        print("🎯 EDGE-BASED BETTING PLAN - NCAA")
        print("=" * 80)
        print()
        
        total_risk = sum(b['bet_size'] for b in portfolio)
        mega_edges = sum(1 for b in portfolio if b['edge_score'] >= 30)
        strong_edges = sum(1 for b in portfolio if 20 <= b['edge_score'] < 30)
        
        print(f"Total Bets: {len(portfolio)}")
        print(f"  🚀 Mega Edges: {mega_edges} (88% expected)")
        print(f"  🔥 Strong Edges: {strong_edges} (75-81% expected)")
        print(f"Total Risk: ${total_risk:.2f} ({total_risk/self.bankroll*100:.1f}% of bankroll)")
        print()
        
        # Group by edge tier
        mega = [b for b in portfolio if b['edge_score'] >= 30]
        strong = [b for b in portfolio if 20 <= b['edge_score'] < 30]
        moderate = [b for b in portfolio if 10 <= b['edge_score'] < 20]
        other = [b for b in portfolio if b['edge_score'] < 10]
        
        for tier_name, tier_bets in [
            ("🚀 MEGA EDGES (88% expected)", mega),
            ("🔥 STRONG EDGES (75-81%)", strong),
            ("✅ MODERATE EDGES (65-73%)", moderate),
            ("📊 OTHER BETS (60-65%)", other)
        ]:
            if not tier_bets:
                continue
            
            print(f"\n{tier_name}")
            print("-" * 80)
            
            for bet in tier_bets:
                print(f"\n{bet['game']}")
                print(f"  Pick: {bet['pick']} {bet.get('spread', '')}")
                print(f"  Base Confidence: {bet['base_confidence']*100:.0f}%")
                print(f"  Edge Score: {bet['edge_score']}/100")
                print(f"  Final Confidence: {bet['final_confidence']*100:.0f}%")
                print(f"  💰 Bet Size: ${bet['bet_size']:.2f}")
                print(f"  Edge Factors:")
                for reason in bet['edge_reasons']:
                    print(f"    • {reason}")

if __name__ == "__main__":
    import sys
    
    # Use prediction log
    pred_file = 'prediction_log.json'
    if len(sys.argv) > 1:
        pred_file = sys.argv[1]
    
    try:
        selector = EdgeBasedBetSelector(pred_file, bankroll=80)
        
        print("🧠 Analyzing with World Model Edge Detection...")
        portfolio = selector.select_bets(max_bets=10)
        
        selector.display_recommendations(portfolio)
        
        # Save plan
        with open('edge_based_betting_plan.json', 'w') as f:
            json.dump(portfolio, f, indent=2)
        
        print("\n" + "=" * 80)
        print("💾 Betting plan saved to: edge_based_betting_plan.json")
        print("=" * 80)
        
    except FileNotFoundError:
        print(f"❌ Prediction file not found: {pred_file}")
        print("Usage: python edge_based_bet_selector.py [prediction_file.json]")
