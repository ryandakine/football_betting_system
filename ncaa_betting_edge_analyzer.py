#!/usr/bin/env python3
"""
NCAA Betting Edge Analyzer - Complete Integration
==================================================

Combines ALL edges into one comprehensive analysis:
1. CLV Tracking - Are you beating closing line?
2. Line Shopping - Best line across books?
3. Key Numbers - Right side of 3/7/10?
4. Trap Detection - Sharp vs public money?
5. 12-Model Ensemble - Game analysis

USAGE:
    from ncaa_betting_edge_analyzer import BettingEdgeAnalyzer

    analyzer = BettingEdgeAnalyzer()

    # Comprehensive analysis
    analysis = analyzer.analyze_bet(
        game="Toledo @ Bowling Green",
        team="Toledo",
        your_line=-2.5,
        lines_available={
            'DraftKings': -3.0,
            'FanDuel': -2.5,
            'BetMGM': -3.5
        },
        closing_line=-4.0,  # Optional: for CLV tracking
        handle_data={  # Optional: for trap detection
            'expected': 0.60,
            'actual': 0.85,
            'moneyline': -150
        }
    )

    print(analysis['recommendation'])
"""

from typing import Dict, List, Optional
from ncaa_clv_tracker import CLVTracker
from ncaa_line_shopping import LineShoppingModule
from ncaa_key_numbers import KeyNumberAnalyzer
from ncaa_trap_detection import NCAATrapDetector


class BettingEdgeAnalyzer:
    """
    Complete betting edge analysis

    Integrates all proven edges + trap detection
    """

    def __init__(self):
        self.clv_tracker = CLVTracker()
        self.line_shopper = LineShoppingModule()
        self.key_analyzer = KeyNumberAnalyzer()
        self.trap_detector = NCAATrapDetector()

    def analyze_bet(
        self,
        game: str,
        team: str,
        your_line: float,
        lines_available: Optional[Dict[str, float]] = None,
        closing_line: Optional[float] = None,
        handle_data: Optional[Dict] = None,
        model_prediction: Optional[float] = None,
        model_confidence: Optional[float] = None
    ) -> Dict:
        """
        Comprehensive bet analysis

        Args:
            game: Game description
            team: Team you're betting on
            your_line: Line you're considering
            lines_available: Dict of book -> spread
            closing_line: Closing line (for CLV)
            handle_data: {expected, actual, moneyline} for trap detection
            model_prediction: Your model's predicted spread
            model_confidence: Model confidence (0-1)

        Returns:
            Comprehensive analysis with recommendations
        """

        print(f"\n{'='*80}")
        print(f"🎯 COMPREHENSIVE BETTING EDGE ANALYSIS")
        print(f"{'='*80}\n")
        print(f"Game: {game}")
        print(f"Team: {team}")
        print(f"Your Line: {your_line:+.1f}")
        print()

        analysis = {
            'game': game,
            'team': team,
            'your_line': your_line,
            'edges': {},
            'warnings': [],
            'total_edge_score': 0,
            'recommendation': ''
        }

        # 1. KEY NUMBER ANALYSIS (Always available)
        print(f"{'='*80}")
        print(f"1️⃣  KEY NUMBER ANALYSIS")
        print(f"{'='*80}\n")

        key_analysis = self.key_analyzer.analyze_spread(your_line, team)
        analysis['edges']['key_numbers'] = {
            'on_good_side': key_analysis.on_good_side,
            'nearest_key': key_analysis.nearest_key_number,
            'ev_adjustment': key_analysis.ev_adjustment,
            'recommendation': key_analysis.recommendation
        }

        print(key_analysis.recommendation)
        if key_analysis.warning:
            print(f"\n⚠️  {key_analysis.warning}")
            analysis['warnings'].append(key_analysis.warning)

        # Add to edge score
        if key_analysis.on_good_side:
            analysis['total_edge_score'] += key_analysis.ev_adjustment

        print()

        # 2. LINE SHOPPING (If lines available)
        if lines_available:
            print(f"{'='*80}")
            print(f"2️⃣  LINE SHOPPING ANALYSIS")
            print(f"{'='*80}\n")

            # Add all lines to shopper
            for book, spread in lines_available.items():
                self.line_shopper.add_line(book, team, spread, game_key=game)

            best_line_result = self.line_shopper.get_best_line(team)

            if best_line_result:
                analysis['edges']['line_shopping'] = {
                    'best_book': best_line_result.book,
                    'best_line': best_line_result.value,
                    'your_line': your_line,
                    'advantage': abs(best_line_result.value - your_line),
                    'crosses_key_number': best_line_result.crosses_key_number
                }

                if abs(best_line_result.value - your_line) < 0.1:
                    print(f"✅ You have BEST LINE available!")
                    print(f"   {best_line_result.book}: {best_line_result.value:+.1f}")
                    analysis['total_edge_score'] += 2.0
                else:
                    print(f"⚠️  BETTER LINE AVAILABLE:")
                    print(f"   Your line: {your_line:+.1f}")
                    print(f"   Best line: {best_line_result.book} at {best_line_result.value:+.1f}")
                    print(f"   Missing: {abs(best_line_result.value - your_line):.1f} points of value")
                    analysis['warnings'].append(
                        f"Better line available: {best_line_result.book} at {best_line_result.value:+.1f}"
                    )
            print()

        # 3. CLV TRACKING (If closing line available)
        if closing_line:
            print(f"{'='*80}")
            print(f"3️⃣  CLOSING LINE VALUE (CLV)")
            print(f"{'='*80}\n")

            clv = your_line - closing_line
            analysis['edges']['clv'] = {
                'your_line': your_line,
                'closing_line': closing_line,
                'clv': clv
            }

            print(f"Your line: {your_line:+.1f}")
            print(f"Closing line: {closing_line:+.1f}")
            print(f"CLV: {clv:+.1f} points\n")

            if clv >= 2.0:
                print(f"🔥 ELITE CLV (+{clv:.1f}) - Sharp level!")
                analysis['total_edge_score'] += 10.0
            elif clv >= 1.0:
                print(f"✅ GOOD CLV (+{clv:.1f}) - Above sharp average")
                analysis['total_edge_score'] += 5.0
            elif clv >= 0.5:
                print(f"👍 POSITIVE CLV (+{clv:.1f}) - Beating market")
                analysis['total_edge_score'] += 2.0
            elif clv >= 0:
                print(f"😐 NEUTRAL CLV ({clv:+.1f})")
            else:
                print(f"❌ NEGATIVE CLV ({clv:+.1f}) - Getting bad price")
                analysis['warnings'].append(f"Negative CLV: {clv:+.1f} points")
                analysis['total_edge_score'] += clv * 2  # Penalize negative CLV

            print()

        # 4. TRAP DETECTION (If handle data available)
        if handle_data:
            print(f"{'='*80}")
            print(f"4️⃣  TRAP DETECTION (Sharp vs Public)")
            print(f"{'='*80}\n")

            trap_signal = self.trap_detector.analyze_game(
                home_ml=handle_data.get('moneyline', -150),
                actual_handle=handle_data.get('actual', 0.5),
                line_opened=handle_data.get('opening', your_line),
                line_current=your_line
            )

            analysis['edges']['trap_detection'] = {
                'signal': trap_signal.signal,
                'trap_score': trap_signal.trap_score,
                'sharp_side': trap_signal.sharp_side,
                'reasoning': trap_signal.reasoning
            }

            print(f"Signal: {trap_signal.signal}")
            print(f"Trap Score: {trap_signal.trap_score}")
            print(f"Sharp Side: {trap_signal.sharp_side}")
            print(f"\n{trap_signal.reasoning}\n")

            # Add to edge score
            if trap_signal.trap_score <= -60:
                # Strong trap - fade public
                print(f"🚨 STRONG TRAP DETECTED - Sharps fading public!")
                analysis['total_edge_score'] += 5.0
            elif trap_signal.trap_score >= 60:
                # Sharp consensus
                print(f"✅ SHARP CONSENSUS - Smart money aligned!")
                analysis['total_edge_score'] += 5.0

            print()

        # 5. MODEL PREDICTION (If available)
        if model_prediction is not None:
            print(f"{'='*80}")
            print(f"5️⃣  MODEL PREDICTION")
            print(f"{'='*80}\n")

            edge_vs_line = abs(model_prediction - your_line)
            analysis['edges']['model'] = {
                'prediction': model_prediction,
                'confidence': model_confidence,
                'edge_vs_line': edge_vs_line
            }

            print(f"Model Prediction: {model_prediction:+.1f}")
            print(f"Your Line: {your_line:+.1f}")
            print(f"Edge: {edge_vs_line:.1f} points")

            if model_confidence:
                print(f"Confidence: {model_confidence:.0%}")

                if model_confidence >= 0.75:
                    print(f"✅ HIGH CONFIDENCE - Model strongly agrees")
                    analysis['total_edge_score'] += 5.0
                elif model_confidence >= 0.65:
                    print(f"👍 GOOD CONFIDENCE")
                    analysis['total_edge_score'] += 3.0

            print()

        # FINAL RECOMMENDATION
        print(f"{'='*80}")
        print(f"🎯 FINAL RECOMMENDATION")
        print(f"{'='*80}\n")

        print(f"Total Edge Score: {analysis['total_edge_score']:+.1f}")
        print()

        # Generate recommendation
        if analysis['total_edge_score'] >= 15.0:
            recommendation = "🔥 STRONG BET - Multiple edges aligned!"
        elif analysis['total_edge_score'] >= 10.0:
            recommendation = "✅ GOOD BET - Several positive edges"
        elif analysis['total_edge_score'] >= 5.0:
            recommendation = "👍 PLAYABLE - Some edge detected"
        elif analysis['total_edge_score'] >= 0:
            recommendation = "😐 MARGINAL - Consider passing"
        else:
            recommendation = "❌ PASS - Multiple warning signs"

        analysis['recommendation'] = recommendation
        print(recommendation)

        # Print warnings
        if analysis['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in analysis['warnings']:
                print(f"   • {warning}")

        print(f"\n{'='*80}\n")

        return analysis


def main():
    """Demo comprehensive edge analyzer"""

    print("NCAA Comprehensive Edge Analyzer Demo\n")

    analyzer = BettingEdgeAnalyzer()

    # Example: Perfect bet (all edges aligned)
    print("🎯 Example: STRONG BET - All edges aligned\n")

    analysis = analyzer.analyze_bet(
        game="Toledo @ Bowling Green",
        team="Toledo",
        your_line=-2.5,
        lines_available={
            'DraftKings': -3.0,
            'FanDuel': -2.5,
            'BetMGM': -3.5
        },
        closing_line=-4.5,  # Great CLV!
        handle_data={
            'moneyline': -150,
            'expected': 0.60,
            'actual': 0.85,  # Trap!
            'opening': -130
        },
        model_prediction=-4.2,
        model_confidence=0.78
    )

    print(f"\n📊 Final Analysis:")
    print(f"   Recommendation: {analysis['recommendation']}")
    print(f"   Edge Score: {analysis['total_edge_score']:+.1f}")
    print(f"   Edges Detected: {len(analysis['edges'])}")


if __name__ == "__main__":
    main()
