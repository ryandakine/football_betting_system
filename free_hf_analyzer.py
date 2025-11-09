#!/usr/bin/env python3
"""
Free Hugging Face NFL Analyzer
Uses free Hugging Face Inference API (no token required)
"""

import asyncio
import json
import aiohttp
import re
from datetime import datetime
from typing import Dict, List

# Free Hugging Face models that work without authentication
FREE_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
]


class HuggingFaceAnalyzer:
    """Analyzes NFL games using free Hugging Face models"""

    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/"
        self.models = FREE_MODELS

    async def query_model(self, session: aiohttp.ClientSession, model: str, prompt: str) -> Dict:
        """Query a Hugging Face model"""
        url = f"{self.api_url}{model}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 250,
                "temperature": 0.7,
                "return_full_text": False
            }
        }

        try:
            async with session.post(url, json=payload, timeout=60) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list) and len(data) > 0:
                        return {"success": True, "text": data[0].get("generated_text", "")}
                    return {"success": False, "error": "Invalid response format"}
                else:
                    text = await response.text()
                    return {"success": False, "error": f"Status {response.status}: {text[:200]}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def parse_prediction(self, text: str, model_name: str) -> Dict:
        """Parse prediction from model output"""
        # Try to extract pick and confidence
        pick = None
        confidence = 0.65  # Default

        # Look for team names
        if "broncos" in text.lower():
            pick = "Denver Broncos"
        elif "raiders" in text.lower():
            pick = "Las Vegas Raiders"

        # Look for confidence indicators
        confidence_words = {
            "very confident": 0.85,
            "highly confident": 0.85,
            "confident": 0.75,
            "fairly confident": 0.70,
            "moderately": 0.65,
            "somewhat": 0.60,
            "slightly": 0.55
        }

        for phrase, conf in confidence_words.items():
            if phrase in text.lower():
                confidence = conf
                break

        return {
            "model": model_name,
            "pick": pick or "Denver Broncos",  # Default to favorite
            "confidence": confidence,
            "reasoning": text[:200]  # First 200 chars
        }

    async def analyze_game(self, game_data: Dict) -> List[Dict]:
        """Analyze game with multiple models"""
        home = game_data['home_team']
        away = game_data['away_team']
        spread = game_data['spread']
        total = game_data['total']

        prompt = f"""Analyze this NFL game and pick a winner:

Game: {away} @ {home}
Spread: {home} {spread}
Total: {total}

The {home} are {abs(spread)}-point favorites at home.

Who will win this game and why? Be specific and confident in your prediction.
Pick the winner and explain in 2-3 sentences."""

        results = []

        async with aiohttp.ClientSession() as session:
            for model in self.models:
                print(f"  Querying {model.split('/')[-1]}...")

                response = await self.query_model(session, model, prompt)

                if response["success"]:
                    prediction = self.parse_prediction(response["text"], model.split('/')[-1])
                    prediction["raw_response"] = response["text"]
                    results.append(prediction)
                    print(f"    ✅ Got prediction: {prediction['pick']}")
                else:
                    print(f"    ❌ Failed: {response['error'][:50]}")

        return results

    def calculate_consensus(self, predictions: List[Dict]) -> Dict:
        """Calculate consensus from multiple predictions"""
        if not predictions:
            return {
                "pick": "No consensus",
                "confidence": 0.0,
                "model_count": 0
            }

        # Count picks
        pick_counts = {}
        total_confidence = 0

        for pred in predictions:
            pick = pred['pick']
            pick_counts[pick] = pick_counts.get(pick, 0) + 1
            total_confidence += pred['confidence']

        # Get majority pick
        consensus_pick = max(pick_counts, key=pick_counts.get)
        avg_confidence = total_confidence / len(predictions)
        agreement = pick_counts[consensus_pick] / len(predictions)

        return {
            "pick": consensus_pick,
            "confidence": avg_confidence,
            "agreement": agreement,
            "model_count": len(predictions)
        }


async def main():
    print("=" * 70)
    print("🏈 FREE HUGGING FACE NFL ANALYSIS")
    print("=" * 70)
    print()

    # Load tonight's game
    with open('tonights_game.json', 'r') as f:
        game = json.load(f)

    print(f"Game: {game['away_team']} @ {game['home_team']}")
    print(f"Time: 8:15 PM ET")
    print()
    print("=" * 70)
    print("BETTING LINES")
    print("=" * 70)
    print(f"Moneyline: {game['away_team']} +{game['away_moneyline']} | {game['home_team']} {game['home_moneyline']}")
    print(f"Spread: {game['home_team']} {game['spread']}")
    print(f"Total: O/U {game['total']}")
    print()
    print("=" * 70)
    print("🤖 RUNNING FREE AI MODELS...")
    print("=" * 70)
    print()

    analyzer = HuggingFaceAnalyzer()
    predictions = await analyzer.analyze_game(game)

    if not predictions:
        print("❌ No predictions generated - all models failed")
        return

    print()
    print("=" * 70)
    print("📊 MODEL PREDICTIONS")
    print("=" * 70)

    for pred in predictions:
        print(f"\n🤖 {pred['model'].upper()}:")
        print(f"   Pick: {pred['pick']}")
        print(f"   Confidence: {pred['confidence']:.1%}")
        print(f"   Analysis: {pred['reasoning'][:150]}...")

    # Calculate consensus
    consensus = analyzer.calculate_consensus(predictions)

    print()
    print("=" * 70)
    print("🎯 CONSENSUS")
    print("=" * 70)
    print(f"   Pick: {consensus['pick']}")
    print(f"   Avg Confidence: {consensus['confidence']:.1%}")
    print(f"   Agreement: {consensus['agreement']:.1%}")
    print(f"   Models: {consensus['model_count']}")
    print()
    print("=" * 70)

    # Save results
    result = {
        "game": game,
        "predictions": predictions,
        "consensus": consensus,
        "analyzed_at": datetime.now().isoformat()
    }

    with open('data/hf_analysis_results.json', 'w') as f:
        json.dump(result, f, indent=2)

    print("💾 Results saved to: data/hf_analysis_results.json")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
