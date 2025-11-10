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

    def __init__(self, llm_provider="openrouter", llm_model="deepseek-r1"):
        """
        Initialize meta-reasoner.

        Args:
            llm_provider: API provider (openrouter, local, etc.)
            llm_model: Which model to use:
                - "deepseek-r1": DeepSeek R1 (FREE, best reasoning)
                - "mistral-7b": Mistral 7B (FREE, fast)
                - "mixtral-8x7b": Mixtral 8x7B (FREE, powerful)
                - "all": Run all three and compare
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
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
            If llm_model="all", returns dict with all three model outputs for comparison
        """

        # Build prompt for LLM
        prompt = self._build_meta_reasoning_prompt(game, model_outputs, game_context)

        # If "all" mode, run all three models and compare
        if self.llm_model == "all":
            return self._run_all_models_comparison(prompt)

        # Otherwise, call single LLM
        response = self._call_llm(prompt, self.llm_model)

        # Parse LLM response
        consensus = self._parse_llm_response(response)
        consensus['llm_model_used'] = self.llm_model

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

    def _run_all_models_comparison(self, prompt: str) -> Dict:
        """
        Run all three LLM models and compare their outputs.

        Returns:
            Dict with comparison of all three models' predictions
        """
        print("\n" + "="*80)
        print("🤖 RUNNING ALL 3 LLM MODELS FOR COMPARISON")
        print("="*80 + "\n")

        models = ["deepseek-r1", "mistral-7b", "mixtral-8x7b"]
        results = {}

        for model in models:
            print(f"🔄 Calling {model}...")
            try:
                response = self._call_llm(prompt, model)
                consensus = self._parse_llm_response(response)
                consensus['llm_model_used'] = model
                results[model] = consensus
                print(f"✅ {model}: {consensus.get('prediction', 'N/A')} ({consensus.get('confidence', 0)}%)")
            except Exception as e:
                print(f"❌ {model} failed: {e}")
                results[model] = {
                    "prediction": "ERROR",
                    "confidence": 0,
                    "error": str(e)
                }

        # Analyze agreement/disagreement
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON ANALYSIS")
        print("="*80 + "\n")

        predictions = [r.get('prediction', 'N/A') for r in results.values()]
        confidences = [r.get('confidence', 0) for r in results.values()]

        # Check if models agree
        unique_predictions = set(predictions)
        if len(unique_predictions) == 1:
            print("✅ ALL 3 MODELS AGREE!")
            print(f"   Consensus: {predictions[0]}")
            print(f"   Avg Confidence: {sum(confidences) / len(confidences):.0f}%")
        else:
            print("⚠️  MODELS DISAGREE:")
            for model, pred, conf in zip(models, predictions, confidences):
                print(f"   {model}: {pred} ({conf}%)")

        # Return compiled results
        return {
            "comparison_mode": True,
            "models": results,
            "agreement": len(unique_predictions) == 1,
            "consensus_prediction": max(set(predictions), key=predictions.count) if predictions else "N/A",
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "all_predictions": predictions,
            "all_confidences": confidences
        }

    def _call_llm(self, prompt: str, model_name: str = None) -> str:
        """
        Call LLM API.

        Args:
            prompt: The prompt to send
            model_name: Which model to use (deepseek-r1, mistral-7b, mixtral-8x7b)

        Returns:
            LLM response text
        """

        if model_name is None:
            model_name = self.llm_model

        # Map model names to API endpoints
        model_mapping = {
            "deepseek-r1": "deepseek/deepseek-r1",
            "mistral-7b": "mistralai/mistral-7b-instruct:free",
            "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct:free"
        }

        api_model = model_mapping.get(model_name, "deepseek/deepseek-r1")

        if self.llm_provider == "openrouter":
            # Use OpenRouter free tier
            try:
                import requests

                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    print(f"⚠️  No OPENROUTER_API_KEY found, using fallback")
                    return self._fallback_simple_consensus(prompt)

                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": api_model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                else:
                    print(f"❌ LLM API error: {response.status_code} - {response.text}")
                    return self._fallback_simple_consensus(prompt)

            except Exception as e:
                print(f"❌ LLM call failed: {e}")
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
    parser.add_argument("--game", required=True, help="Game to analyze (e.g., 'PHI @ GB')")
    parser.add_argument("--week", type=int, required=True, help="Week number")
    parser.add_argument("--model",
                        default="deepseek-r1",
                        choices=["deepseek-r1", "mistral-7b", "mixtral-8x7b", "all"],
                        help="Which LLM model to use (default: deepseek-r1)")
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

    reasoner = LLMMetaReasoner(llm_model=args.model)
    consensus = reasoner.combine_predictions(args.game, model_outputs, game_context)

    print("\n" + "="*80)
    print(f"🤖 LLM META-REASONER RESULTS (Model: {args.model.upper()})")
    print("="*80)

    if consensus.get('comparison_mode'):
        # All three models - show detailed comparison
        print("\n📊 DETAILED COMPARISON:\n")
        for model_name, result in consensus['models'].items():
            print(f"\n{'='*60}")
            print(f"🔹 {model_name.upper()}")
            print(f"{'='*60}")
            print(f"Prediction: {result.get('prediction', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 0)}%")
            print(f"Bet Amount: ${result.get('bet_amount', 0)}")
            print(f"Reasoning: {result.get('reasoning', 'N/A')[:200]}...")

        print("\n" + "="*80)
        print("🎯 FINAL VERDICT")
        print("="*80)
        print(f"Agreement: {'✅ YES' if consensus['agreement'] else '⚠️  NO'}")
        print(f"Consensus Prediction: {consensus['consensus_prediction']}")
        print(f"Average Confidence: {consensus['avg_confidence']:.0f}%")
    else:
        # Single model - show full output
        print(json.dumps(consensus, indent=2))


if __name__ == "__main__":
    main()
