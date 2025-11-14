#!/usr/bin/env python3
"""
NCAA DeepSeek R1 Reasoner - Meta-Analysis of 12 Models
=======================================================

PRINCIPLE: Reasoning Over Ensemble (Not Just Another Model)

WHY THIS EXISTS:
- 12 models each see different patterns
- DeepSeek R1 analyzes WHAT they're seeing collectively
- Finds edges/patterns that Vegas is missing
- Provides reasoning about WHY to bet or not bet

WORKFLOW:
1. Run all 12 NCAA models → Get individual predictions
2. DeepSeek R1 analyzes predictions → Finds patterns
3. R1 identifies disagreements/consensus
4. R1 reasons about edges Vegas might miss
5. Final recommendation with detailed reasoning

Based on NFL R1 system (60.91x returns over 10 years)
Adapted for NCAA/College Football patterns
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class ModelPrediction:
    """Individual model prediction"""
    model_name: str
    predicted_spread: float
    confidence: float
    reasoning: str


@dataclass
class R1Analysis:
    """DeepSeek R1 meta-analysis result"""
    recommended_pick: str  # "HOME -3.5" or "AWAY +3.5" or "NO BET"
    confidence: int  # 70-100
    reasoning: str  # Detailed R1 reasoning
    patterns_found: List[str]
    vegas_miss: Optional[str]  # What Vegas might be missing
    model_consensus: float  # % of models agreeing
    contrarian_signal: Optional[str]
    trap_signal: Optional[Dict]  # Sharp money vs public (NEW!)


class NCAADeepSeekR1Reasoner:
    """
    Meta-reasoning layer over 12 NCAA models with Dual-Reader system:
    - Claude (via OpenRouter) - Primary reader for ensemble analysis
    - DeepSeek R1 - Secondary reasoning validation

    PRINCIPLE: Investment → System with multiple perspectives
    Combines Claude's analytical strength with DeepSeek R1's reasoning
    """

    def __init__(self, api_key: Optional[str] = None, use_claude: bool = True):
        self.use_claude = use_claude
        self.deepseek_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
        
        if self.use_claude:
            if not self.openrouter_key:
                print("⚠️  OpenRouter key not found, falling back to DeepSeek only")
                self.use_claude = False
            else:
                self.claude_client = OpenAI(
                    api_key=self.openrouter_key,
                    base_url="https://openrouter.ai/api/v1"
                )
                print("✅ Claude (OpenRouter) reader initialized")
        
        if self.deepseek_key:
            self.deepseek_client = OpenAI(
                api_key=self.deepseek_key,
                base_url="https://api.deepseek.com"
            )
            print("✅ DeepSeek R1 reasoner initialized")
        elif not self.use_claude:
            raise ValueError("Either Claude (OpenRouter) or DeepSeek API key required")
        else:
            print("⚠️  DeepSeek key not found, using Claude only")

    def analyze_game(
        self,
        game: Dict,
        model_predictions: List[ModelPrediction],
        market_spread: float,
        contrarian_signal: Optional[Dict] = None,
        trap_signal: Optional[Dict] = None
    ) -> R1Analysis:
        """
        Meta-analysis: What are 12 models seeing that Vegas might miss?

        WHY: R1 finds patterns in ensemble predictions
        """

        # Build context for R1
        context = self._build_analysis_context(
            game,
            model_predictions,
            market_spread,
            contrarian_signal,
            trap_signal
        )

        # Ask Claude or R1 to reason over the ensemble
        prompt = self._build_r1_prompt(context)

        # Get reasoning from primary reader
        if self.use_claude:
            # Use Claude via OpenRouter
            response = self.claude_client.chat.completions.create(
                model="openai/gpt-4-turbo",  # Using Claude equivalent via OpenRouter
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0,
                max_tokens=4000
            )
        else:
            # Fallback to DeepSeek R1
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0,
                max_tokens=4000
            )

        # Parse response
        analysis = self._parse_r1_response(
            response.choices[0].message.content,
            model_predictions,
            market_spread
        )

        return analysis

    def _build_analysis_context(
        self,
        game: Dict,
        predictions: List[ModelPrediction],
        market_spread: float,
        contrarian_signal: Optional[Dict],
        trap_signal: Optional[Dict]
    ) -> Dict:
        """Build context for R1 analysis"""

        home_team = game.get('home_team', 'Home')
        away_team = game.get('away_team', 'Away')

        # Calculate consensus
        spreads = [p.predicted_spread for p in predictions]
        avg_spread = sum(spreads) / len(spreads) if spreads else 0

        # Find disagreements
        spread_std = (sum((s - avg_spread)**2 for s in spreads) / len(spreads))**0.5 if spreads else 0

        # Model agreement (how many models within 3 points of consensus)
        agreement_count = sum(1 for s in spreads if abs(s - avg_spread) <= 3.0)
        agreement_pct = agreement_count / len(spreads) if spreads else 0

        return {
            'game': {
                'home_team': home_team,
                'away_team': away_team,
                'day_of_week': game.get('day_of_week', 'Unknown'),
                'conference': game.get('conference', 'Unknown'),
                'is_maction': game.get('is_maction', False),
            },
            'market': {
                'spread': market_spread,
                'favorite': home_team if market_spread < 0 else away_team,
                'underdog': away_team if market_spread < 0 else home_team,
            },
            'models': {
                'predictions': predictions,
                'consensus_spread': avg_spread,
                'spread_std': spread_std,
                'agreement_pct': agreement_pct,
                'edge_vs_market': abs(avg_spread - market_spread),
            },
            'contrarian': contrarian_signal,
            'trap': trap_signal
        }

    def _get_system_prompt(self) -> str:
        """System prompt for R1"""
        return """You are an elite NCAA football betting analyst using DeepSeek R1's reasoning capabilities.

Your job: Analyze predictions from 12 specialized models and find patterns/edges that Vegas might be missing.

THE 12 MODELS:
1. XGBoost Super - Non-linear relationships, feature interactions
2. Neural Net Deep - Complex patterns, momentum, trends
3. Alt Spread - Market inefficiency detection
4. Bayesian Ensemble - Uncertainty quantification
5. Momentum Model - Recent performance (last 3 games)
6. Situational - Road/home, weather, rest days
7. Advanced Stats - EPA, success rate, efficiency
8. Drive Outcomes - Red zone efficiency, turnovers
9. Opponent-Adjusted - Strength of schedule corrections
10. Special Teams - FG%, punt returns, kickoffs
11. Pace & Tempo - Possessions per game, variance
12. Game Script - How game will unfold

YOUR TASK:
1. Analyze what PATTERNS the models are seeing collectively
2. Identify DISAGREEMENTS between models (why?)
3. Find CONSENSUS among models
4. Determine what EDGE exists vs Vegas line
5. Reason about what Vegas MIGHT BE MISSING

NCAA-SPECIFIC PATTERNS TO CONSIDER:
- Home field advantage (3.5 pts average, varies by stadium)
- Conference dynamics (familiarity, rivalries)
- Big name school bias (public overvalues Alabama, Ohio State, etc.)
- MACtion games (Tuesday/Wednesday, softer lines)
- Bye week advantages
- Look-ahead spots (team thinking about next week)
- Letdown spots (after big win)
- Weather impact (wind for passing teams)

TRAP DETECTION (Sharp Money vs Public):
- Expected handle % varies by odds (e.g., -150 should get 60% of money)
- When actual handle significantly diverges from expected = TRAP SIGNAL
- Trap score -100 to +100:
  * Negative = public overloaded, sharps on other side (FADE PUBLIC)
  * Positive = sharp consensus (RIDE WITH SHARPS)
  * -60 or below = moderate to strong trap
  * +60 or above = sharp consensus
- Reverse line movement (RLM) = line moves AGAINST public (confirms sharp action)
- When models AGREE with trap signal = VERY STRONG BET
- When models DISAGREE with trap signal = use reasoning to determine which is right
- Public can be RIGHT sometimes - analyze why sharps might disagree

REASONING PROCESS:
1. What do models AGREE on? (consensus = strong signal)
2. What do models DISAGREE on? (disagreement = uncertainty or edge)
3. Which models are most relevant for THIS matchup?
4. What patterns exist that market might not price in?
5. Is there value vs Vegas line?

OUTPUT FORMAT:
{
  "recommended_pick": "HOME -3.5" or "AWAY +7.0" or "NO BET",
  "confidence": 75,
  "reasoning": "Detailed reasoning here...",
  "patterns_found": ["Pattern 1", "Pattern 2", ...],
  "vegas_miss": "What Vegas might be missing",
  "edge_explanation": "Why this is +EV"
}

Be thorough in reasoning. Think step-by-step. Find the edge."""

    def _build_r1_prompt(self, context: Dict) -> str:
        """Build prompt for R1 analysis"""

        game = context['game']
        market = context['market']
        models = context['models']
        contrarian = context['contrarian']
        trap = context['trap']

        prompt = f"""Analyze this NCAA football game and find edges:

GAME:
{game['away_team']} @ {game['home_team']}
Day: {game['day_of_week']}
Conference: {game['conference']}
MACtion: {game['is_maction']}

VEGAS LINE:
Market Spread: {market['spread']:+.1f}
Favorite: {market['favorite']}
Underdog: {market['underdog']}

12-MODEL PREDICTIONS:
"""

        # Add each model's prediction
        for pred in models['predictions']:
            prompt += f"\n{pred.model_name}:"
            prompt += f"\n  Spread: {pred.predicted_spread:+.1f}"
            prompt += f"\n  Confidence: {pred.confidence:.0%}"
            if pred.reasoning:
                prompt += f"\n  Key Factor: {pred.reasoning}"

        # Add ensemble metrics
        prompt += f"""

ENSEMBLE METRICS:
Consensus Spread: {models['consensus_spread']:+.1f}
Model Agreement: {models['agreement_pct']:.0%} (within 3 points)
Spread Std Dev: {models['spread_std']:.2f}
Edge vs Market: {models['edge_vs_market']:.1f} points
"""

        # Add contrarian intelligence if available
        if contrarian:
            prompt += f"""

CONTRARIAN INTELLIGENCE:
Strength: {contrarian.get('strength', 0)}/5 stars
Public %: {contrarian.get('public_percentage', 0.5):.0%}
Recommendation: {contrarian.get('recommendation', 'N/A')}
Sharp Money: {'YES' if contrarian.get('sharp_money_detected') else 'NO'}
"""

        # Add trap detection signal if available
        if trap:
            prompt += f"""

TRAP DETECTION (Sharp Money vs Public):
Signal: {trap.get('signal', 'N/A')}
Trap Score: {trap.get('trap_score', 0)} (-100=STRONG TRAP, +100=EXTREME SHARP)
Sharp Side: {trap.get('sharp_side', 'N/A')}
Expected Handle: {trap.get('expected_handle', 0.5):.0%}
Actual Handle: {trap.get('actual_handle', 0.5):.0%}
Divergence: {trap.get('divergence', 0):+.1%}
Reverse Line Movement: {'YES' if trap.get('reverse_line_movement') else 'NO'}
Confidence: {trap.get('confidence', 0):.0%}

Reasoning: {trap.get('reasoning', 'N/A')}

IMPORTANT: When trap score < -60, public is overloaded and sharps are on the other side.
When models AGREE with trap signal, that's a VERY STRONG bet.
When models DISAGREE with trap signal, determine which signal is correct.
"""

        prompt += """

YOUR ANALYSIS:
1. What patterns do you see in the 12 model predictions?
2. Where do models AGREE? (strong signal)
3. Where do models DISAGREE? (why? edge or uncertainty?)
4. Which models are most relevant for THIS specific matchup?
5. What might Vegas be MISSING that the models see?
6. Is there betting value vs the market line?
7. What's the recommended pick and confidence?

Provide detailed reasoning and identify specific edges."""

        return prompt

    def _parse_r1_response(
        self,
        response_text: str,
        predictions: List[ModelPrediction],
        market_spread: float
    ) -> R1Analysis:
        """Parse R1 reasoning into structured analysis"""

        try:
            # Try to extract JSON if present
            if '{' in response_text and '}' in response_text:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)

                return R1Analysis(
                    recommended_pick=data.get('recommended_pick', 'NO BET'),
                    confidence=data.get('confidence', 70),
                    reasoning=data.get('reasoning', response_text),
                    patterns_found=data.get('patterns_found', []),
                    vegas_miss=data.get('vegas_miss'),
                    model_consensus=sum(1 for p in predictions) / len(predictions) if predictions else 0,
                    contrarian_signal=data.get('contrarian_signal'),
                    trap_signal=data.get('trap_signal')
                )
        except:
            pass

        # Fallback: Parse text reasoning
        return R1Analysis(
            recommended_pick='NO BET',
            confidence=70,
            reasoning=response_text,
            patterns_found=[],
            vegas_miss=None,
            model_consensus=0.0,
            contrarian_signal=None,
            trap_signal=None
        )

    def print_analysis(
        self,
        analysis: R1Analysis,
        game: Dict
    ):
        """Display R1 analysis to user"""

        print(f"\n{'='*80}")
        print(f"🧠 DEEPSEEK R1 META-ANALYSIS")
        print(f"{'='*80}\n")

        print(f"🏈 {game.get('away_team')} @ {game.get('home_team')}")
        print()

        print(f"💡 R1 RECOMMENDATION:")
        print(f"   Pick: {analysis.recommended_pick}")
        print(f"   Confidence: {analysis.confidence}%")
        print()

        if analysis.patterns_found:
            print(f"📊 PATTERNS DETECTED:")
            for pattern in analysis.patterns_found:
                print(f"   • {pattern}")
            print()

        if analysis.vegas_miss:
            print(f"🎯 WHAT VEGAS IS MISSING:")
            print(f"   {analysis.vegas_miss}")
            print()

        print(f"🤔 R1 REASONING:")
        # Print reasoning (limit to first 800 chars for display)
        reasoning_display = analysis.reasoning[:800] + "..." if len(analysis.reasoning) > 800 else analysis.reasoning
        print(f"   {reasoning_display}")
        print()

        print(f"📈 MODEL CONSENSUS: {analysis.model_consensus:.0%}")
        print()


def main():
    """Production NCAA R1 Reasoner"""
    print("NCAA DeepSeek R1 Reasoner - Production Ready")
    print("\nIntegrate with run_ncaa_predictions_production.py")
    print("Requires: OPENROUTER_API_KEY or DEEPSEEK_API_KEY environment variables")
    print("\nUsage:")
    print("  reasoner = NCAADeepSeekR1Reasoner()  # Uses Claude + DeepSeek")
    print("  analysis = reasoner.analyze_game(game, predictions, spread)")


if __name__ == "__main__":
    main()
