#!/usr/bin/env python3
"""
Hybrid Football AI Overlord
===========================

Primary path: Google Colab GPUs running the full Hugging Face ensemble.
Fallback: Local Linux execution (GPU if available, CPU otherwise).

Use:
  python3 main.py --ai-service colab --ai-gpu A100 --ai-vram 40
  python3 main.py --local-only
  python3 main.py gradio  (launches UI on localhost:7860)

Expose env:
  HUGGINGFACE_API_TOKEN=...  (already wired via api_config/get_api_keys)

This script:
  * Detects Colab vs local
  * Optionally calls a remote Colab endpoint (if COLAB_HYBRID_ENDPOINT is set)
  * Falls back to local pipelines instantly on failure
  * Provides CLI + Gradio interface
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from transformers import pipeline

import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("HybridFootballAI")

# =====================================================================
# Runtime Detection / Config
# =====================================================================

IS_COLAB = "google.colab" in sys.modules
DEFAULT_COLAB_GPU = "A100"
DEFAULT_COLAB_VRAM = 40   # GB


def detect_local_device() -> Tuple[str, bool]:
    """Return (device_name, cuda_enabled)."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return name, True
    return "CPU", False


def load_env() -> None:
    """Ensure .env is loaded so the Hugging Face token is available."""
    try:
        from api_config import get_api_keys
        keys = get_api_keys()
        if not keys.get("huggingface"):
            LOGGER.warning("⚠️ Hugging Face token missing; run huggingface-cli login")
    except Exception as exc:  # noqa: broad-except
        LOGGER.warning("⚠️ Could not load api_config - %s", exc)


load_env()

# =====================================================================
# Colab Remote Invocation
# =====================================================================


class ColabInvoker:
    """Optional remote call to a Colab REST endpoint."""

    def __init__(self) -> None:
        self.endpoint = os.getenv("COLAB_HYBRID_ENDPOINT")
        self.session = requests.Session() if self.endpoint else None

    def available(self) -> bool:
        return bool(self.session and self.endpoint)

    def invoke(self, payload: Dict[str, Any], timeout: int = 120) -> Optional[Dict[str, Any]]:
        if not self.available():
            return None
        try:
            LOGGER.info("📡 Dispatching payload to Colab endpoint: %s", self.endpoint)
            resp = self.session.post(self.endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: broad-except
            LOGGER.warning("⚠️ Colab endpoint failed: %s", exc)
            return None


# =====================================================================
# Local Model Suite
# =====================================================================


class LocalModelSuite:
    """Loads light-weight Hugging Face pipelines locally."""

    def __init__(self) -> None:
        device = 0 if torch.cuda.is_available() else -1
        LOGGER.info("🧠 Initializing local model suite on %s", "CUDA" if device == 0 else "CPU")

        self.generators = pipeline(
            "text-generation",
            model="gpt2",
            device=device,
            max_length=512,
            do_sample=True,
            temperature=0.9,
        )
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
        )
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device,
        )

    def analyze(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        LOGGER.info("⚙️ Running local pipelines...")
        generated = self.generators(
            text + "\nStrategic response:",
            max_length=400,
            num_return_sequences=1,
            pad_token_id=self.generators.tokenizer.eos_token_id,
        )[0]["generated_text"]

        sentiment = self.sentiment(text)[0]
        labels = ["aggressive blitz", "zone coverage", "run defense", "pass rush"]
        zero_shot = self.zero_shot(text, labels)

        return {
            "mode": "local",
            "text_generation": generated,
            "sentiment": sentiment,
            "tactics": zero_shot,
            "context": context,
        }


# =====================================================================
# Hybrid AI Engine
# =====================================================================


class HybridFootballAI:
    """Brains of the operation, handles Colab vs local fallback."""

    def __init__(self, prefer_colab: bool = True) -> None:
        self.colab = ColabInvoker()
        self.local_suite = LocalModelSuite()
        self.prefer_colab = prefer_colab

    def _prepare_payload(self, text: str, stats_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        stats_preview = None
        if stats_df is not None:
            stats_preview = stats_df.head(10).to_dict(orient="records")
        return {
            "input_text": text,
            "stats_preview": stats_preview,
            "meta": {
                "prefer_colab": self.prefer_colab,
            }
        }

    def run(self, text: str, stats_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        payload = self._prepare_payload(text, stats_df)

        if self.prefer_colab and self.colab.available():
            LOGGER.info("🛰️ Attempting Colab ensemble inference...")
            result = self.colab.invoke(payload)
            if result:
                result["mode"] = "colab"
                result.setdefault("logs", []).append("Executed on Colab ensemble.")
                LOGGER.info("✅ Colab run complete.")
                return result
            LOGGER.warning("⚠️ Colab attempt failed; falling back to local execution.")

        LOGGER.info("🛠️ Using local fallback.")
        result = self.local_suite.analyze(text, {"from_csv": bool(stats_df is not None)})
        result.setdefault("logs", []).append("Executed locally (fallback).")
        return result


# =====================================================================
# CLI / Gradio Interface
# =====================================================================


def analyze_text(args: argparse.Namespace) -> None:
    prefer_colab = not args.local_only
    engine = HybridFootballAI(prefer_colab=prefer_colab)
    stats_df = None
    if args.csv:
        stats_df = pd.read_csv(args.csv)
    result = engine.run(args.text, stats_df)
    print(json.dumps(result, indent=2))


def launch_gradio(args: argparse.Namespace) -> None:
    prefer_colab = not args.local_only
    engine = HybridFootballAI(prefer_colab=prefer_colab)

    def _predict(play_text: str, stats_file) -> Dict[str, Any]:
        stats_df = None
        if stats_file is not None:
            stats_df = pd.read_csv(stats_file.name)
        return engine.run(play_text, stats_df)

    with gr.Blocks(title="Hybrid Football AI Overlord") as demo:
        gr.Markdown("## Hybrid Football AI Overlord\nUnleash GPU fury or fallback to local tactics.")
        play_text = gr.Textbox(label="Game Situation / Playbook Text", lines=8)
        stats_upload = gr.File(label="Optional Stats CSV")
        run_btn = gr.Button("Analyze (No Mercy)")
        output_json = gr.JSON(label="Analysis Output")
        run_btn.click(fn=_predict, inputs=[play_text, stats_upload], outputs=[output_json])

    LOGGER.info("🎛️ Launching Gradio UI on http://127.0.0.1:7860")
    demo.launch(server_port=7860, share=False)


def main() -> None:
    # Initialize crew adjustment system for automatic NFL prediction adjustments
    try:
        from init_crew_adjustments import initialize_crew_system
        initialize_crew_system()
    except Exception as e:
        LOGGER.warning(f"⚠️ Crew adjustment system initialization failed: {e}")
    
    parser = argparse.ArgumentParser(description="Run hybrid football AI analysis")
    parser.add_argument("--text", type=str, default="QB scramble left, no huddle.", help="Game situation text")
    parser.add_argument("--csv", type=str, help="Path to stats CSV input")
    parser.add_argument("--local-only", action="store_true", help="Force local execution; ignore Colab even if available.")
    parser.add_argument("--ai-service", type=str, default=os.getenv("AI_GPU_SERVICE", "colab"), help="Documentation flag")
    parser.add_argument("--ai-gpu", type=str, default=os.getenv("AI_GPU_TYPE", DEFAULT_COLAB_GPU), help="Documentation flag")
    parser.add_argument("--ai-vram", type=int, default=int(os.getenv("AI_GPU_VRAM", DEFAULT_COLAB_VRAM)), help="Documentation flag")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("gradio", help="Launch Gradio web UI")
    subparsers.add_parser("info", help="Show runtime info and bail")

    args = parser.parse_args()

    if args.command == "gradio":
        launch_gradio(args)
    elif args.command == "info":
        device_name, cuda = detect_local_device()
        info = {
            "is_colab": IS_COLAB,
            "colab_endpoint": os.getenv("COLAB_HYBRID_ENDPOINT"),
            "local_device": device_name,
            "cuda_available": cuda,
        }
        print(json.dumps(info, indent=2))
    else:
        analyze_text(args)


if __name__ == "__main__":
    main()
