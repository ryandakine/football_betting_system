#!/usr/bin/env python3
"""
Hugging Face NFL Analyzer with Token
Uses authenticated Hugging Face Inference API
"""

import asyncio
import json
import aiohttp
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Good free models that work with tokens
MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
]


async def query_hf_model(session, model, prompt, token):
    """Query Hugging Face Inference API with token"""
    url = f"https://api-inference.huggingface.co/models/{model}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False
        }
    }

    try:
        async with session.post(url, headers=headers, json=payload, timeout=60) as response:
            if response.status == 200:
                data = await response.json()
                if isinstance(data, list) and len(data) > 0:
                    text = data[0].get("generated_text", "")
                    return {"success": True, "text": text, "model": model.split('/')[-1]}
                return {"success": False, "error": "Invalid response", "model": model.split('/')[-1]}
            else:
                error_text = await response.text()
                return {"success": False, "error": f"{response.status}: {error_text[:100]}", "model": model.split('/')[-1]}
    except Exception as e:
        return {"success": False, "error": str(e), "model": model.split('/')[-1]}


def parse_prediction(text, model_name):
    """Extract prediction from model response"""
    text_lower = text.lower()

    # Determine pick
    broncos_mentions = text_lower.count("broncos") + text_lower.count("denver")
    raiders_mentions = text_lower.count("raiders") + text_lower.count("las vegas")

    if broncos_mentions > raiders_mentions:
        pick = "Denver Broncos"
    elif raiders_mentions > broncos_mentions:
        pick = "Las Vegas Raiders"
    else:
        pick = "Denver Broncos"  # Default to favorite

    # Determine confidence
    confidence = 0.70  # Default
    if any(word in text_lower for word in ["very confident", "highly", "strong"]):
        confidence = 0.85
    elif any(word in text_lower for word in ["confident", "likely", "expect"]):
        confidence = 0.75
    elif any(word in text_lower for word in ["could", "might", "possibly"]):
        confidence = 0.60

    return {
        "model": model_name,
        "pick": pick,
        "confidence": confidence,
        "reasoning": text[:200]
    }


async def main():
    print("=" * 70)
    print("🏈 HUGGING FACE NFL ANALYSIS (WITH TOKEN)")
    print("=" * 70)
    print()

    # Get token
    token = os.getenv('HUGGINGFACE_API_TOKEN')
    if not token:
        print("❌ No Hugging Face token found in .env")
        return

    print(f"Using token: {token[:20]}...")

    # Load game data
    with open('tonights_game.json', 'r') as f:
        game = json.load(f)

    print()
    print(f"Game: {game['away_team']} @ {game['home_team']}")
    print(f"Spread: {game['home_team']} {game['spread']}")
    print(f"Total: {game['total']}")
    print()
    print("=" * 70)
    print("🤖 QUERYING AI MODELS...")
    print("=" * 70)
    print()

    prompt = f"""Analyze this NFL game and predict the winner:

{game['away_team']} @ {game['home_team']}
Spread: {game['home_team']} {game['spread']}
Total: {game['total']}

The {game['home_team']} are {abs(game['spread'])}-point favorites at home.

Who will win and cover the spread? Explain your prediction in 2-3 sentences."""

    predictions = []

    async with aiohttp.ClientSession() as session:
        for model in MODELS:
            print(f"  Querying {model.split('/')[-1]}...")
            result = await query_hf_model(session, model, prompt, token)

            if result["success"]:
                pred = parse_prediction(result["text"], result["model"])
                pred["raw"] = result["text"]
                predictions.append(pred)
                print(f"    ✅ {pred['pick']} ({pred['confidence']:.0%})")
            else:
                print(f"    ❌ Failed: {result['error'][:60]}")

    if not predictions:
        print("\n❌ All models failed - check token permissions")
        return

    print()
    print("=" * 70)
    print("📊 PREDICTIONS")
    print("=" * 70)

    for pred in predictions:
        print(f"\n🤖 {pred['model'].upper()}")
        print(f"   Pick: {pred['pick']}")
        print(f"   Confidence: {pred['confidence']:.0%}")
        print(f"   Analysis: {pred['reasoning'][:120]}...")

    # Consensus
    picks = [p['pick'] for p in predictions]
    consensus_pick = max(set(picks), key=picks.count)
    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
    agreement = picks.count(consensus_pick) / len(predictions)

    print()
    print("=" * 70)
    print("🎯 CONSENSUS")
    print("=" * 70)
    print(f"   Pick: {consensus_pick}")
    print(f"   Confidence: {avg_conf:.0%}")
    print(f"   Agreement: {agreement:.0%} ({picks.count(consensus_pick)}/{len(predictions)} models)")
    print("=" * 70)

    # Save
    result = {
        "game": game,
        "predictions": predictions,
        "consensus": {
            "pick": consensus_pick,
            "confidence": avg_conf,
            "agreement": agreement
        }
    }

    with open('data/hf_predictions.json', 'w') as f:
        json.dump(result, f, indent=2)

    print("\n💾 Saved to: data/hf_predictions.json")


if __name__ == "__main__":
    asyncio.run(main())
