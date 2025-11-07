#!/usr/bin/env python3
"""
Free GGUF Model NFL Analyzer
Uses local GGUF models (no API costs!)
Requires: llama-cpp-python
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️  llama-cpp-python not installed. Install with: pip install llama-cpp-python")


class GGUFAnalyzer:
    """NFL analyzer using free local GGUF models"""

    def __init__(self):
        self.models_dir = Path("models/gguf")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Define available GGUF models (download these manually or via script)
        self.available_models = {
            "llama3-8b": "llama-3-8b-instruct.Q4_K_M.gguf",
            "mistral-7b": "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
            "phi3": "phi-3-mini-4k-instruct.Q4_K_M.gguf",
        }

    def find_local_models(self) -> List[str]:
        """Find downloaded GGUF models"""
        found = []
        for name, filename in self.available_models.items():
            filepath = self.models_dir / filename
            if filepath.exists():
                found.append((name, str(filepath)))
                print(f"✅ Found: {name} ({filepath.name})")
            else:
                print(f"❌ Missing: {name} ({filename})")
        return found

    def load_model(self, model_path: str) -> Llama:
        """Load a GGUF model"""
        print(f"Loading model: {model_path}")

        # Load with optimized settings
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window
            n_threads=4,  # CPU threads
            n_gpu_layers=0,  # Set to 35+ if you have GPU
            verbose=False
        )

        return llm

    def create_prompt(self, game_data: Dict) -> str:
        """Create analysis prompt"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert NFL analyst. Analyze the game and pick a winner.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze this NFL game:

{game_data['away_team']} @ {game_data['home_team']}
Spread: {game_data['home_team']} {game_data['spread']}
Total: {game_data['total']}

The {game_data['home_team']} are {abs(game_data['spread'])}-point favorites at home.

Pick the winner and explain why in 2-3 sentences. Be specific about which team will win.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt

    def parse_response(self, text: str, model_name: str, game_data: Dict) -> Dict:
        """Parse model response"""
        text_lower = text.lower()

        # Count team mentions
        home_lower = game_data['home_team'].lower()
        away_lower = game_data['away_team'].lower()

        home_count = text_lower.count(home_lower.split()[-1])  # Use last word (e.g., "Broncos")
        away_count = text_lower.count(away_lower.split()[-1])

        # Determine pick
        if home_count > away_count:
            pick = game_data['home_team']
        elif away_count > home_count:
            pick = game_data['away_team']
        else:
            # Default to favorite if unclear
            pick = game_data['home_team']

        # Determine confidence
        confidence = 0.70
        if any(word in text_lower for word in ["very confident", "highly", "definitely", "strong"]):
            confidence = 0.85
        elif any(word in text_lower for word in ["confident", "likely", "should", "expect"]):
            confidence = 0.75
        elif any(word in text_lower for word in ["could", "might", "possibly", "maybe"]):
            confidence = 0.60

        return {
            "model": model_name,
            "pick": pick,
            "confidence": confidence,
            "reasoning": text[:250].strip()
        }

    def analyze_game(self, game_data: Dict) -> List[Dict]:
        """Analyze game with available GGUF models"""
        if not LLAMA_CPP_AVAILABLE:
            print("❌ llama-cpp-python not installed!")
            print("Install with: pip install llama-cpp-python")
            return []

        print("\n" + "="*70)
        print("🤖 RUNNING LOCAL GGUF MODELS")
        print("="*70)
        print()

        # Find available models
        local_models = self.find_local_models()

        if not local_models:
            print("\n❌ No GGUF models found!")
            print("\nDownload models from:")
            print("  https://huggingface.co/models?library=gguf")
            print("\nRecommended models:")
            for name, filename in self.available_models.items():
                print(f"  • {name}: {filename}")
            print(f"\nSave to: {self.models_dir}/")
            return []

        predictions = []
        prompt = self.create_prompt(game_data)

        print(f"\n{'='*70}")
        print("Analyzing with local models...")
        print(f"{'='*70}\n")

        for model_name, model_path in local_models:
            try:
                print(f"🔄 {model_name}...")

                # Load model
                llm = self.load_model(model_path)

                # Generate prediction
                output = llm(
                    prompt,
                    max_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["<|eot_id|>", "<|end_of_text|>"],
                    echo=False
                )

                response_text = output['choices'][0]['text']

                # Parse prediction
                prediction = self.parse_response(response_text, model_name, game_data)
                prediction['raw_response'] = response_text
                predictions.append(prediction)

                print(f"  ✅ {prediction['pick']} ({prediction['confidence']:.0%})")

            except Exception as e:
                print(f"  ❌ Error: {str(e)[:60]}")
                continue

        return predictions

    def calculate_consensus(self, predictions: List[Dict]) -> Dict:
        """Calculate consensus from predictions"""
        if not predictions:
            return {
                "pick": "No consensus",
                "confidence": 0.0,
                "agreement": 0.0,
                "model_count": 0
            }

        # Count picks
        picks = [p['pick'] for p in predictions]
        pick_counts = {}
        for pick in picks:
            pick_counts[pick] = pick_counts.get(pick, 0) + 1

        # Consensus
        consensus_pick = max(pick_counts, key=pick_counts.get)
        avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
        agreement = pick_counts[consensus_pick] / len(predictions)

        return {
            "pick": consensus_pick,
            "confidence": avg_confidence,
            "agreement": agreement,
            "model_count": len(predictions)
        }


def main():
    print("=" * 70)
    print("🏈 FREE LOCAL GGUF NFL ANALYZER")
    print("=" * 70)
    print()

    # Load game data
    with open('tonights_game.json', 'r') as f:
        game = json.load(f)

    print(f"Game: {game['away_team']} @ {game['home_team']}")
    print(f"Spread: {game['home_team']} {game['spread']}")
    print(f"Total: {game['total']}")

    # Create analyzer
    analyzer = GGUFAnalyzer()

    # Analyze
    predictions = analyzer.analyze_game(game)

    if not predictions:
        print("\n❌ No predictions generated")
        return

    # Display results
    print("\n" + "="*70)
    print("📊 PREDICTIONS")
    print("="*70)

    for pred in predictions:
        print(f"\n🤖 {pred['model'].upper()}")
        print(f"   Pick: {pred['pick']}")
        print(f"   Confidence: {pred['confidence']:.0%}")
        print(f"   Analysis: {pred['reasoning'][:120]}...")

    # Consensus
    consensus = analyzer.calculate_consensus(predictions)

    print("\n" + "="*70)
    print("🎯 CONSENSUS")
    print("="*70)
    print(f"   Pick: {consensus['pick']}")
    print(f"   Confidence: {consensus['confidence']:.0%}")
    print(f"   Agreement: {consensus['agreement']:.0%} ({int(consensus['agreement'] * consensus['model_count'])}/{consensus['model_count']} models)")
    print("="*70)

    # Save results
    result = {
        "game": game,
        "predictions": predictions,
        "consensus": consensus,
        "analyzed_at": datetime.now().isoformat()
    }

    with open('data/gguf_predictions.json', 'w') as f:
        json.dump(result, f, indent=2)

    print("\n💾 Saved to: data/gguf_predictions.json")


if __name__ == "__main__":
    main()
