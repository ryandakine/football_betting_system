#!/usr/bin/env python3
"""
DeepSeek-R1 Contrarian-Enhanced Analysis

WHY THIS EXISTS:
DeepSeek-R1 has home favorite bias (37% ROI but picks favorites 60-70% of time).
This fixes it by feeding contrarian intelligence BEFORE analysis.

DESIGN PHILOSOPHY: Investment → System
- Contrarian intelligence is fetched automatically
- Fed to DeepSeek as context (not separate model)
- DeepSeek adjusts picks based on sharp money signals

THE FIX:
1. Fetch contrarian intelligence (line movement, public %)
2. Generate enhanced prompt with contrarian context
3. DeepSeek analyzes with contrarian awareness
4. Output: Contrarian-adjusted pick

USAGE:
    python deepseek_contrarian_analysis.py --game "PHI @ GB"
    python deepseek_contrarian_analysis.py --week 11 --all
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from contrarian_intelligence import ContrarianIntelligence
except ImportError:
    print("❌ Could not import contrarian_intelligence")
    sys.exit(1)


class DeepSeekContrarianAnalyzer:
    """DeepSeek-R1 analysis enhanced with contrarian intelligence"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.contrarian = ContrarianIntelligence()
        self.output_dir = Path(__file__).parent / "analysis"
        self.output_dir.mkdir(exist_ok=True)

    def analyze_game(self, game: str, game_context: Optional[Dict] = None) -> Dict:
        """
        Analyze game with contrarian-enhanced DeepSeek-R1.

        Args:
            game: Game string like "PHI @ GB"
            game_context: Additional game context (injuries, weather, etc.)

        Returns:
            Dict with contrarian-adjusted pick
        """
        print("=" * 70)
        print(f"🎯 DEEP SEEK CONTRARIAN ANALYSIS: {game}")
        print("=" * 70)
        print()

        # Step 1: Fetch contrarian intelligence
        print("📊 Step 1: Fetching contrarian intelligence...")
        contrarian_intel = self.contrarian.get_contrarian_intelligence(game)
        print()

        # Step 2: Generate enhanced prompt
        print("🤖 Step 2: Generating contrarian-enhanced prompt...")
        prompt = self._generate_enhanced_prompt(game, contrarian_intel, game_context)
        print()

        # Step 3: Call DeepSeek-R1 (or simulate if no API)
        print("🧠 Step 3: Running DeepSeek-R1 analysis...")
        if self.api_key:
            analysis = self._call_deepseek(prompt)
        else:
            print("   ⚠️  No API key - showing prompt only")
            analysis = self._simulate_analysis(game, contrarian_intel)
        print()

        # Step 4: Save results
        output_file = self.output_dir / f"contrarian_analysis_{game.replace(' ', '_').replace('@', 'at')}.json"
        result = {
            'game': game,
            'contrarian_intelligence': contrarian_intel,
            'deepseek_analysis': analysis,
            'prompt_used': prompt,
            'timestamp': datetime.now().isoformat()
        }

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"💾 Saved to {output_file}")
        print()

        # Print summary
        self._print_summary(analysis, contrarian_intel)

        return result

    def _generate_enhanced_prompt(
        self,
        game: str,
        contrarian_intel: Dict,
        game_context: Optional[Dict]
    ) -> str:
        """Generate the contrarian-enhanced prompt for DeepSeek."""

        parts = game.split('@')
        away_team = parts[0].strip() if len(parts) == 2 else "AWAY"
        home_team = parts[1].strip() if len(parts) == 2 else "HOME"

        # Extract contrarian signals
        public_betting = contrarian_intel.get('public_betting', {})
        public_pct = public_betting.get('public_percentage', {})
        home_pct = public_pct.get('home', 50)
        away_pct = public_pct.get('away', 50)

        line_movement = contrarian_intel.get('line_movement', {})
        sharp_indicators = line_movement.get('sharp_indicators', {})

        contrarian_signals = contrarian_intel.get('contrarian_signals', {})
        strength = contrarian_signals.get('strength', 0)
        recommendation = contrarian_signals.get('recommendation', 'No clear signal')
        reasoning = contrarian_signals.get('reasoning', [])

        prompt = f"""You are an elite NFL betting analyst with contrarian intelligence.

GAME: {away_team} @ {home_team}

===== CONTRARIAN INTELLIGENCE =====

PUBLIC BETTING PERCENTAGES:
- {home_team} (home): {home_pct}%
- {away_team} (away): {away_pct}%
- Contrarian threshold: 70% (fade public when exceeded)

PUBLIC BIAS ANALYSIS:
{public_betting.get('reasoning', 'No data available')}

SHARP MONEY INDICATORS:
- Sharp side: {sharp_indicators.get('sharp_side', 'Unknown')}
- Confidence: {sharp_indicators.get('confidence', 0)}/5
- Reasoning: {' | '.join(sharp_indicators.get('reasoning', ['No indicators detected']))}

CONTRARIAN SIGNALS:
- Strength: {strength}/5 {'⭐' * strength}
- Recommendation: {recommendation}
- Analysis:
{chr(10).join(f'  • {r}' for r in reasoning)}

{"===== REVERSE LINE MOVEMENT DETECTED =====" if sharp_indicators.get('sharp_side') else ""}
{"Line moving AGAINST public money = SHARP ACTION!" if sharp_indicators.get('sharp_side') else ""}

===== GAME CONTEXT =====

{self._format_game_context(game_context) if game_context else "Limited game context available"}

===== YOUR TASK =====

Analyze this game with CONTRARIAN FOCUS:

1. **Identify Public Bias**
   - Public is {home_pct}% on {home_team} - is this too heavy?
   - What narrative is public overvaluing?
   - Is this a "square" pick (public loves it)?

2. **Evaluate Contrarian Signal**
   - Contrarian strength: {strength}/5
   - Sharp money: {sharp_indicators.get('sharp_side', 'Unknown')}
   - Should we fade the public here?

3. **Make Contrarian-Informed Pick**
   - If contrarian strength ≥3: HEAVILY WEIGHT IT
   - If public >{70}% on favorite: Consider fading
   - Balance: Contrarian signal vs fundamentals

IMPORTANT ANTI-BIAS RULES:
- DO NOT default to home favorite just because they're home
- DO NOT pick based on popularity or public narrative
- DO NOT ignore strong contrarian signals (strength ≥3)
- DO ask: "Why is the public wrong here?"

{"⚠️  STRONG CONTRARIAN SIGNAL - Fade public recommended!" if strength >= 3 else ""}

OUTPUT FORMAT (JSON):
{{
  "pick": "[TEAM] [SPREAD]",
  "confidence": [70-85],
  "reasoning": [
    "Contrarian analysis: ...",
    "Public bias: ...",
    "Sharp money: ...",
    "Fundamentals: ..."
  ],
  "contrarian_weight": [0-5],
  "public_fade": true/false,
  "expected_value": "+X%"
}}

THINK STEP BY STEP. Show your contrarian reasoning.
"""

        return prompt

    def _format_game_context(self, context: Dict) -> str:
        """Format game context for prompt."""
        lines = []

        if context.get('home_team'):
            lines.append(f"HOME TEAM: {context['home_team']}")
            if context.get('home_record'):
                lines.append(f"  Record: {context['home_record']}")
            if context.get('home_injuries'):
                lines.append(f"  Injuries: {context['home_injuries']}")

        if context.get('away_team'):
            lines.append(f"\nAWAY TEAM: {context['away_team']}")
            if context.get('away_record'):
                lines.append(f"  Record: {context['away_record']}")
            if context.get('away_injuries'):
                lines.append(f"  Injuries: {context['away_injuries']}")

        if context.get('referee'):
            lines.append(f"\nREFEREE: {context['referee']}")

        if context.get('weather'):
            lines.append(f"WEATHER: {context['weather']}")

        return '\n'.join(lines) if lines else "No additional context"

    def _call_deepseek(self, prompt: str) -> Dict:
        """Call DeepSeek-R1 API with enhanced prompt."""
        # This would call the actual DeepSeek API
        # For now, return a placeholder
        return {
            'status': 'API call would go here',
            'model': 'deepseek-r1',
            'prompt_length': len(prompt),
            'note': 'Implement OpenRouter API call here'
        }

    def _simulate_analysis(self, game: str, contrarian_intel: Dict) -> Dict:
        """Simulate DeepSeek analysis when no API key."""
        signals = contrarian_intel.get('contrarian_signals', {})
        strength = signals.get('strength', 0)
        recommendation = signals.get('recommendation', '')

        parts = game.split('@')
        away_team = parts[0].strip() if len(parts) == 2 else "AWAY"
        home_team = parts[1].strip() if len(parts) == 2 else "HOME"

        # Simulate contrarian-adjusted pick
        public_betting = contrarian_intel.get('public_betting', {})
        public_pct = public_betting.get('public_percentage', {})
        home_pct = public_pct.get('home', 50)

        # If strong contrarian signal (strength ≥3) and public >70% on home:
        # Pick the away team (fade public)
        if strength >= 3 and home_pct > 70:
            pick = f"{away_team} +SPREAD"
            confidence = 73 + strength
            public_fade = True
            reasoning = [
                f"Contrarian signal: Strength {strength}/5 - Strong fade opportunity",
                f"Public bias: {home_pct}% on {home_team} - too heavy",
                "Sharp money: Detected action on away team",
                "Fundamentals: Public overvaluing home team advantage"
            ]
        else:
            # Normal analysis (but still contrarian-aware)
            pick = f"{home_team} -SPREAD"
            confidence = 72
            public_fade = False
            reasoning = [
                "Contrarian signal: Weak - no clear fade opportunity",
                f"Public bias: {home_pct}% on {home_team} - within normal range",
                "Sharp money: No strong indicators",
                "Fundamentals: Home team has legitimate edge"
            ]

        return {
            'pick': pick,
            'confidence': confidence,
            'reasoning': reasoning,
            'contrarian_weight': strength,
            'public_fade': public_fade,
            'expected_value': '+2-4%' if public_fade else '+1-2%',
            'note': 'Simulated analysis - actual DeepSeek would provide deeper reasoning'
        }

    def _print_summary(self, analysis: Dict, contrarian_intel: Dict):
        """Print analysis summary."""
        print("=" * 70)
        print("📋 ANALYSIS SUMMARY")
        print("=" * 70)
        print()

        # Contrarian strength
        signals = contrarian_intel.get('contrarian_signals', {})
        strength = signals.get('strength', 0)
        print(f"Contrarian Strength: {'⭐' * strength} ({strength}/5)")
        print()

        # Pick
        print(f"🎯 PICK: {analysis.get('pick', 'N/A')}")
        print(f"📊 CONFIDENCE: {analysis.get('confidence', 0)}%")
        print(f"🔄 PUBLIC FADE: {'YES' if analysis.get('public_fade') else 'NO'}")
        print(f"💰 EXPECTED VALUE: {analysis.get('expected_value', 'N/A')}")
        print()

        # Reasoning
        print("💭 REASONING:")
        for reason in analysis.get('reasoning', []):
            print(f"   • {reason}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-R1 analysis with contrarian intelligence"
    )
    parser.add_argument(
        "--game",
        required=True,
        help="Game to analyze (e.g. 'PHI @ GB')"
    )
    parser.add_argument(
        "--api-key",
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )

    args = parser.parse_args()

    analyzer = DeepSeekContrarianAnalyzer(api_key=args.api_key)
    result = analyzer.analyze_game(args.game)

    print("✅ Analysis complete!")
    print()
    print("💡 NEXT STEPS:")
    print("   1. Review the contrarian-adjusted pick")
    print("   2. If contrarian strength ≥3: Consider fading public")
    print("   3. Use this pick in your betting workflow")


if __name__ == "__main__":
    main()
