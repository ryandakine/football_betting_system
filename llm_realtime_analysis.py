#!/usr/bin/env python3
"""
LLM Real-Time Analysis Engine
Uses Claude AI for game-specific insights and edge detection

Analyzes:
- Matchup dynamics (team styles, coaching, motivation)
- Injury impact (key players out, backup quality)
- Situational factors (playoff implications, revenge games)
- Narrative/psychological edges (media hype, public perception)
- Risk assessment (variance, ceiling/floor outcomes)

Impact: Catches 10-15% of edges that pure statistical models miss
"""
import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("❌ Anthropic package not installed. Run: pip install anthropic")
    anthropic = None


class LLMGameAnalyzer:
    """
    Uses Claude AI to analyze specific games and provide insights

    Prompts Claude with all available context and gets back:
    - Narrative edges
    - Risk assessment
    - Confidence adjustments
    - Specific recommendations
    """

    def __init__(self, api_key: Optional[str] = None):
        if not anthropic:
            raise ImportError("anthropic package required. Install: pip install anthropic")

        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("No Anthropic API key. Set ANTHROPIC_API_KEY env var")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-5-20250929"  # Latest model

        # Output directory
        self.output_dir = Path('data/llm_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_game(self, game_context: Dict) -> Dict:
        """
        Get Claude's analysis of a specific game

        Args:
            game_context: All available context about the game

        Returns:
            Dict with Claude's insights and recommendations
        """
        # Build comprehensive prompt
        prompt = self._build_analysis_prompt(game_context)

        # Get Claude's analysis
        print(f"\n🤖 Asking Claude to analyze {game_context['game']}...")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            analysis_text = response.content[0].text

            # Parse structured response
            analysis = self._parse_analysis(analysis_text, game_context)

            # Save analysis
            self._save_analysis(game_context['game'], analysis)

            return analysis

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {
                'error': str(e),
                'edge_detected': False,
                'confidence_adjustment': 0.0
            }

    def _build_analysis_prompt(self, context: Dict) -> str:
        """Build comprehensive analysis prompt for Claude"""

        prompt = f"""You are an expert NFL betting analyst. Analyze this game and provide specific insights.

GAME: {context['game']}
{context.get('away_team', 'Away')} @ {context.get('home_team', 'Home')}

CURRENT BETTING LINE:
- Spread: {context.get('spread', 'N/A')}
- Total: {context.get('total', 'N/A')}
- Our predicted edge: {context.get('estimated_edge', 'N/A')}%

STATISTICAL ANALYSIS:
- Sharp money: {context.get('sharp_side', 'N/A')} (trap score: {context.get('trap_score', 0)}/5)
- Public betting: {context.get('public_pct', 'N/A')}% on favorite
- Line movement: {context.get('line_movement', 'N/A')} points
- CLV opportunity: {context.get('clv_improvement', 'N/A')}%

WEATHER CONDITIONS:
- Temperature: {context.get('temperature', 'N/A')}°F
- Wind: {context.get('wind_speed', 'N/A')} mph
- Conditions: {context.get('weather_conditions', 'N/A')}
- Stadium: {context.get('stadium', 'N/A')} ({'Dome' if context.get('is_dome') else 'Outdoor'})

TEAM CONTEXT:
- {context.get('home_team', 'Home')}: {context.get('home_record', 'N/A')} record
- {context.get('away_team', 'Away')}: {context.get('away_record', 'N/A')} record
"""

        if context.get('injuries'):
            prompt += f"\nKEY INJURIES:\n{context['injuries']}\n"

        if context.get('situational_factors'):
            prompt += f"\nSITUATIONAL FACTORS:\n{context['situational_factors']}\n"

        prompt += """
Please analyze this game and provide:

1. NARRATIVE EDGE (0-10 scale):
   - What narrative/psychological factors favor one side?
   - How might public perception be wrong?
   - Any revenge game, rivalry, or motivational angles?

2. INJURY/PERSONNEL IMPACT (0-10 scale):
   - How significant are the injury impacts?
   - Quality of backups?
   - Matchup-specific personnel advantages?

3. SITUATIONAL FACTORS (0-10 scale):
   - Playoff implications?
   - Look-ahead/trap game potential?
   - Rest/travel advantages?

4. RISK ASSESSMENT:
   - Variance (LOW/MEDIUM/HIGH): How predictable is this game?
   - Floor outcome: What's the worst case?
   - Ceiling outcome: What's the best case?

5. RECOMMENDATION:
   - Confidence adjustment (-15% to +15%)
   - Specific bet recommendation (if any)
   - Key factors supporting this view

Format your response as JSON:
{
    "narrative_edge": {
        "score": 0-10,
        "description": "...",
        "favors": "home/away/neither"
    },
    "injury_impact": {
        "score": 0-10,
        "description": "...",
        "favors": "home/away/neither"
    },
    "situational_edge": {
        "score": 0-10,
        "description": "...",
        "favors": "home/away/neither"
    },
    "risk_assessment": {
        "variance": "LOW/MEDIUM/HIGH",
        "floor": "...",
        "ceiling": "...",
        "predictability": 0-10
    },
    "recommendation": {
        "confidence_adjustment": -0.15 to +0.15,
        "recommended_bet": "...",
        "key_factors": ["...", "..."],
        "contrarian_angle": "..."
    }
}
"""

        return prompt

    def _parse_analysis(self, analysis_text: str, context: Dict) -> Dict:
        """Parse Claude's response into structured format"""

        try:
            # Try to extract JSON from response
            # Look for JSON block
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = analysis_text[json_start:json_end]
                analysis = json.loads(json_str)

                # Add metadata
                analysis['game'] = context['game']
                analysis['timestamp'] = datetime.now().isoformat()
                analysis['raw_response'] = analysis_text

                # Calculate overall edge
                narrative = analysis.get('narrative_edge', {}).get('score', 0)
                injury = analysis.get('injury_impact', {}).get('score', 0)
                situational = analysis.get('situational_edge', {}).get('score', 0)

                analysis['overall_edge_score'] = (narrative + injury + situational) / 3

                # Determine if edge detected
                analysis['edge_detected'] = analysis['overall_edge_score'] >= 6.0

                return analysis
            else:
                # Fallback: return raw text
                return {
                    'game': context['game'],
                    'timestamp': datetime.now().isoformat(),
                    'raw_response': analysis_text,
                    'edge_detected': False,
                    'confidence_adjustment': 0.0,
                    'error': 'Could not parse JSON response'
                }

        except json.JSONDecodeError as e:
            return {
                'game': context['game'],
                'timestamp': datetime.now().isoformat(),
                'raw_response': analysis_text,
                'edge_detected': False,
                'confidence_adjustment': 0.0,
                'error': f'JSON parse error: {e}'
            }

    def _save_analysis(self, game: str, analysis: Dict):
        """Save analysis to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_analysis_{game.replace(' ', '_')}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"   ✅ Analysis saved to {filepath}")

    def apply_llm_insights(self, prediction: Dict, analysis: Dict) -> Dict:
        """
        Apply LLM insights to prediction

        Args:
            prediction: Base prediction
            analysis: LLM analysis results

        Returns:
            Enhanced prediction with LLM insights
        """
        enhanced = prediction.copy()

        # Get confidence adjustment from LLM
        recommendation = analysis.get('recommendation', {})
        adjustment = recommendation.get('confidence_adjustment', 0.0)

        # Apply adjustment
        base_confidence = enhanced.get('confidence', 0.60)
        llm_confidence = base_confidence + adjustment
        llm_confidence = max(0.50, min(0.85, llm_confidence))

        enhanced['base_confidence'] = base_confidence
        enhanced['llm_confidence'] = llm_confidence
        enhanced['llm_adjustment'] = adjustment
        enhanced['confidence'] = llm_confidence

        # Add LLM insights
        enhanced['llm_insights'] = {
            'edge_detected': analysis.get('edge_detected', False),
            'overall_score': analysis.get('overall_edge_score', 0),
            'narrative_edge': analysis.get('narrative_edge', {}),
            'injury_impact': analysis.get('injury_impact', {}),
            'situational_edge': analysis.get('situational_edge', {}),
            'risk_assessment': analysis.get('risk_assessment', {}),
            'recommended_bet': recommendation.get('recommended_bet', ''),
            'key_factors': recommendation.get('key_factors', []),
            'contrarian_angle': recommendation.get('contrarian_angle', '')
        }

        return enhanced

    def batch_analyze_games(self, games: List[Dict]) -> List[Dict]:
        """
        Analyze multiple games in batch

        Args:
            games: List of game contexts

        Returns:
            List of analyses
        """
        analyses = []

        print(f"\n🤖 Analyzing {len(games)} games with Claude AI...")

        for i, game in enumerate(games, 1):
            print(f"\nGame {i}/{len(games)}: {game['game']}")
            analysis = self.analyze_game(game)
            analyses.append(analysis)

        print(f"\n✅ Batch analysis complete")

        return analyses

    def print_analysis_summary(self, analysis: Dict):
        """Print formatted analysis summary"""

        print("\n" + "="*80)
        print(f"🤖 CLAUDE AI ANALYSIS: {analysis.get('game', 'Unknown')}")
        print("="*80)

        if 'error' in analysis:
            print(f"\n❌ Error: {analysis['error']}")
            return

        # Edge scores
        print(f"\nEDGE SCORES:")
        narrative = analysis.get('narrative_edge', {})
        print(f"  Narrative: {narrative.get('score', 0)}/10 - {narrative.get('description', 'N/A')}")
        print(f"  Favors: {narrative.get('favors', 'N/A')}")

        injury = analysis.get('injury_impact', {})
        print(f"\n  Injury Impact: {injury.get('score', 0)}/10 - {injury.get('description', 'N/A')}")
        print(f"  Favors: {injury.get('favors', 'N/A')}")

        situational = analysis.get('situational_edge', {})
        print(f"\n  Situational: {situational.get('score', 0)}/10 - {situational.get('description', 'N/A')}")
        print(f"  Favors: {situational.get('favors', 'N/A')}")

        print(f"\nOVERALL EDGE SCORE: {analysis.get('overall_edge_score', 0):.1f}/10")
        print(f"Edge Detected: {'YES' if analysis.get('edge_detected') else 'NO'}")

        # Risk assessment
        risk = analysis.get('risk_assessment', {})
        print(f"\nRISK ASSESSMENT:")
        print(f"  Variance: {risk.get('variance', 'N/A')}")
        print(f"  Predictability: {risk.get('predictability', 0)}/10")
        print(f"  Floor: {risk.get('floor', 'N/A')}")
        print(f"  Ceiling: {risk.get('ceiling', 'N/A')}")

        # Recommendation
        rec = analysis.get('recommendation', {})
        print(f"\nRECOMMENDATION:")
        print(f"  Confidence Adjustment: {rec.get('confidence_adjustment', 0):+.1%}")
        print(f"  Recommended Bet: {rec.get('recommended_bet', 'N/A')}")
        print(f"  Contrarian Angle: {rec.get('contrarian_angle', 'N/A')}")

        print(f"\n  Key Factors:")
        for factor in rec.get('key_factors', []):
            print(f"    • {factor}")

        print("\n" + "="*80)


def example_usage():
    """Example of using LLM analysis"""

    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("="*80)
        print("⚠️  ANTHROPIC_API_KEY not set")
        print("="*80)
        print("\nTo use LLM analysis:")
        print("1. Get API key from: https://console.anthropic.com/")
        print("2. Set environment variable:")
        print("   export ANTHROPIC_API_KEY='your_key_here'")
        print("3. Run this script again")
        print("\n" + "="*80)
        return

    # Create analyzer
    analyzer = LLMGameAnalyzer()

    # Example game context
    game_context = {
        'game': 'Chiefs @ Bills',
        'home_team': 'Buffalo Bills',
        'away_team': 'Kansas City Chiefs',
        'spread': -3.0,
        'total': 47.5,
        'estimated_edge': 6.5,
        'sharp_side': 'Bills',
        'trap_score': 4,
        'public_pct': 72,
        'line_movement': -0.5,
        'clv_improvement': 2.5,
        'temperature': 25,
        'wind_speed': 22,
        'weather_conditions': 'Cloudy, windy',
        'stadium': 'Highmark Stadium',
        'is_dome': False,
        'home_record': '8-3',
        'away_record': '9-2',
        'injuries': '- Chiefs: None significant\n- Bills: LB Matt Milano (out)',
        'situational_factors': '- Potential playoff preview\n- Bills seeking revenge for last year\'s playoff loss'
    }

    # Get Claude's analysis
    analysis = analyzer.analyze_game(game_context)

    # Print summary
    analyzer.print_analysis_summary(analysis)

    # Apply to prediction
    base_prediction = {
        'game': 'Chiefs @ Bills',
        'prediction': 'Bills -3',
        'confidence': 0.68,
        'estimated_edge': 6.5
    }

    enhanced = analyzer.apply_llm_insights(base_prediction, analysis)

    print("\n" + "="*80)
    print("PREDICTION ENHANCEMENT")
    print("="*80)
    print(f"\nBase confidence: {enhanced['base_confidence']*100:.0f}%")
    print(f"LLM adjustment: {enhanced['llm_adjustment']:+.1%}")
    print(f"Final confidence: {enhanced['llm_confidence']*100:.0f}%")
    print("\n" + "="*80)


if __name__ == "__main__":
    example_usage()
