#!/usr/bin/env python3
"""
LLM Meta-Reasoner: Combines 12 Model Outputs Using AI Reasoning
================================================================

This meta-model takes predictions from all 12 base models and uses
an LLM to reason through conflicts and find optimal weighting.

Advantages:
- No training data needed
- Explainable reasoning (shows why it made decision)
- Finds complex interactions humans miss
- Uses your 5 free LLM calls strategically

Usage:
    python llm_meta_reasoner.py --game "BAL @ PIT" --week 11
"""

import argparse
import json
import os
from typing import Dict, List, Tuple
from pathlib import Path


class LLMMetaReasoner:
    """Uses LLM to intelligently combine 12 model predictions."""

    def __init__(self, llm_provider="openrouter"):
        self.llm_provider = llm_provider
        self.model_history = []  # Track which models are most accurate

    def combine_predictions(
        self,
        game: str,
        model_outputs: Dict[str, Dict],
        game_context: Dict
    ) -> Dict:
        """
        Combine 12 model predictions using LLM reasoning.

        Args:
            game: Game identifier (e.g., "BAL @ PIT")
            model_outputs: Dict of {model_name: {prediction, confidence, reasoning}}
            game_context: Game details (weather, injuries, referee, etc.)

        Returns:
            Dict with consensus prediction, confidence, and reasoning
        """

        # Build prompt for LLM
        prompt = self._build_meta_reasoning_prompt(game, model_outputs, game_context)

        # Call LLM (using cheapest available)
        response = self._call_llm(prompt)

        # Parse LLM response
        consensus = self._parse_llm_response(response)

        return consensus

    def _build_meta_reasoning_prompt(
        self,
        game: str,
        model_outputs: Dict[str, Dict],
        game_context: Dict
    ) -> str:
        """Build detailed prompt for LLM meta-reasoning."""

        prompt = f"""You are a professional sports betting analyst with access to 12 specialized prediction models.

GAME: {game}

GAME CONTEXT:
"""

        # Add context
        for key, value in game_context.items():
            prompt += f"- {key}: {value}\n"

        prompt += "\n" + "="*80 + "\n"
        prompt += "12 MODEL PREDICTIONS:\n"
        prompt += "="*80 + "\n\n"

        # Add each model's prediction
        for model_name, output in model_outputs.items():
            pred = output.get('prediction', 'N/A')
            conf = output.get('confidence', 0)
            reason = output.get('reasoning', 'No reasoning provided')

            prompt += f"**{model_name.upper()}**: {pred} ({conf:.0f}% confidence)\n"
            prompt += f"   Reasoning: {reason}\n\n"

        # Add conflict detection
        conflicts = self._detect_conflicts(model_outputs)
        if conflicts:
            prompt += "="*80 + "\n"
            prompt += "⚠️ CONFLICTS DETECTED:\n"
            prompt += "="*80 + "\n\n"
            for conflict in conflicts:
                prompt += f"- {conflict}\n"
            prompt += "\n"

        # Add historical model accuracy (if available)
        if self.model_history:
            prompt += "="*80 + "\n"
            prompt += "📊 HISTORICAL MODEL ACCURACY:\n"
            prompt += "="*80 + "\n\n"
            for model_name, accuracy in sorted(
                self.model_history,
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                prompt += f"- {model_name}: {accuracy:.1f}% accurate\n"
            prompt += "\n"

        # Add meta-reasoning instructions
        prompt += "="*80 + "\n"
        prompt += "YOUR TASK:\n"
        prompt += "="*80 + "\n\n"

        prompt += """Analyze all 12 models and provide:

1. **CONFLICT RESOLUTION**: When models disagree, explain which to trust and why
   - Consider game-specific factors (weather, injuries, matchups)
   - Identify which models are most relevant for THIS specific game
   - Explain any interactions between factors (e.g., wind + division rivalry)

2. **CONSENSUS PREDICTION**: Provide final prediction with confidence (0-100%)
   - Bet recommendation: OVER/UNDER or SPREAD or MONEYLINE
   - Exact line/number to bet
   - Confidence level (80%+ = strong bet, 75-79% = medium bet, <75% = pass)

3. **REASONING**: Step-by-step explanation of your decision
   - Which models you weighted most heavily
   - Why you resolved conflicts the way you did
   - What interactions you identified
   - Why this bet has edge over Vegas line

4. **BET SIZING**: Given $100 bankroll
   - 80%+ confidence: Bet $6 (3 units)
   - 75-79% confidence: Bet $4 (2 units)
   - <75% confidence: PASS

Think step by step. Find patterns humans miss. Identify non-linear interactions.

OUTPUT FORMAT:
{
    "prediction": "HOME COVERS -3.5" or "OVER 45.5" or "AWAY ML",
    "confidence": 78,
    "bet_amount": 4,
    "reasoning": "Detailed explanation...",
    "key_factors": ["Factor 1", "Factor 2", "Factor 3"],
    "models_trusted": ["model_1", "model_2"],
    "models_ignored": ["model_3", "model_4"],
    "edge_identified": "Specific edge over Vegas line"
}
"""

        return prompt

    def _detect_conflicts(self, model_outputs: Dict[str, Dict]) -> List[str]:
        """Detect when models strongly disagree."""
        conflicts = []

        # Extract predictions
        predictions = []
        for model_name, output in model_outputs.items():
            pred = output.get('prediction', '')
            conf = output.get('confidence', 0)

            if 'OVER' in pred or 'over' in pred:
                predictions.append((model_name, 'OVER', conf))
            elif 'UNDER' in pred or 'under' in pred:
                predictions.append((model_name, 'UNDER', conf))
            elif 'HOME' in pred or 'home' in pred:
                predictions.append((model_name, 'HOME', conf))
            elif 'AWAY' in pred or 'away' in pred:
                predictions.append((model_name, 'AWAY', conf))

        # Find conflicts
        over_models = [m for m, p, c in predictions if p == 'OVER']
        under_models = [m for m, p, c in predictions if p == 'UNDER']
        home_models = [m for m, p, c in predictions if p == 'HOME']
        away_models = [m for m, p, c in predictions if p == 'AWAY']

        if over_models and under_models:
            conflicts.append(
                f"Total conflict: {', '.join(over_models)} say OVER but "
                f"{', '.join(under_models)} say UNDER"
            )

        if home_models and away_models:
            conflicts.append(
                f"Spread conflict: {', '.join(home_models)} favor HOME but "
                f"{', '.join(away_models)} favor AWAY"
            )

        return conflicts

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API (OpenRouter, Anthropic, etc.)."""

        # For now, return placeholder
        # You'll replace this with actual API call

        if self.llm_provider == "openrouter":
            # Use OpenRouter free tier
            try:
                import requests

                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    return self._fallback_simple_consensus(prompt)

                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "meta-llama/llama-3.2-3b-instruct:free",  # Free tier
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    }
                )

                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                else:
                    return self._fallback_simple_consensus(prompt)

            except Exception as e:
                print(f"LLM call failed: {e}")
                return self._fallback_simple_consensus(prompt)

        return self._fallback_simple_consensus(prompt)

    def _fallback_simple_consensus(self, prompt: str) -> str:
        """Fallback: Simple weighted average if LLM unavailable."""
        return json.dumps({
            "prediction": "CONSENSUS UNAVAILABLE - USE SIMPLE AVERAGE",
            "confidence": 50,
            "bet_amount": 0,
            "reasoning": "LLM unavailable, using fallback",
            "key_factors": [],
            "models_trusted": [],
            "models_ignored": [],
            "edge_identified": "None"
        })

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response into structured output."""

        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return {
                    "prediction": "PARSE ERROR",
                    "confidence": 50,
                    "bet_amount": 0,
                    "reasoning": response,
                    "key_factors": [],
                    "models_trusted": [],
                    "models_ignored": [],
                    "edge_identified": "Parse error"
                }
        except json.JSONDecodeError:
            return {
                "prediction": "PARSE ERROR",
                "confidence": 50,
                "bet_amount": 0,
                "reasoning": response,
                "key_factors": [],
                "models_trusted": [],
                "models_ignored": [],
                "edge_identified": "Parse error"
            }


def main():
    """Test the LLM meta-reasoner."""
    parser = argparse.ArgumentParser(description="LLM Meta-Reasoner")
    parser.add_argument("--game", required=True, help="Game to analyze")
    parser.add_argument("--week", type=int, required=True, help="Week number")
    args = parser.parse_args()

    # Example: Mock 12 model outputs
    model_outputs = {
        "referee_intelligence": {
            "prediction": "HOME COVERS -3.5",
            "confidence": 73,
            "reasoning": "Referee Bill Vinovich assigned - home team 67% ATS with him"
        },
        "public_sentiment": {
            "prediction": "AWAY COVERS +3.5",
            "confidence": 68,
            "reasoning": "82% of public on home team - contrarian opportunity"
        },
        "weather_model": {
            "prediction": "UNDER 45.5",
            "confidence": 71,
            "reasoning": "35 mph winds forecasted - passing game limited"
        },
        # ... add other 9 models
    }

    game_context = {
        "home_team": args.game.split('@')[1].strip(),
        "away_team": args.game.split('@')[0].strip(),
        "spread": -3.5,
        "total": 45.5,
        "weather": "35 mph winds, 40°F",
        "referee": "Bill Vinovich",
        "injuries": "Home: RB1 OUT, Away: CB1 Questionable"
    }

    reasoner = LLMMetaReasoner()
    consensus = reasoner.combine_predictions(args.game, model_outputs, game_context)

    print("\n" + "="*80)
    print("🤖 LLM META-REASONER CONSENSUS")
    print("="*80)
    print(json.dumps(consensus, indent=2))


if __name__ == "__main__":
    main()
