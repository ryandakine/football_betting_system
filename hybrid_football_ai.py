#!/usr/bin/env python3
"""
Hybrid Football AI Engine
=========================

Primary inference runs on a remote Google Colab GPU ensemble (via REST endpoint that
the user exposes). Automatic fallback executes locally using lightweight Hugging Face
pipelines so the workflow never dies because of rate limits or quota nonsense.

This module exposes `run_hybrid_analysis` and `get_shared_engine` for reuse inside
the Tkinter GUI or command-line utilities.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple

import pandas as pd
import requests
import torch
from transformers import pipeline

LOGGER = logging.getLogger(__name__)

# Try to ensure .env is loaded so HUGGINGFACE_API_TOKEN is in the environment.
def _prime_env() -> None:
    try:
        from api_config import get_api_keys  # pylint: disable=import-error

        keys = get_api_keys()
        if keys.get("huggingface"):
            LOGGER.debug("Hugging Face key detected via api_config loader.")
    except Exception as exc:  # noqa: broad-except
        LOGGER.debug("api_config loader failed (non-fatal): %s", exc)


_prime_env()


def _detect_device() -> Tuple[int, str]:
    """Return (device_index, label)."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        LOGGER.info("CUDA detected: %s", name)
        return 0, name
    LOGGER.info("CUDA not available, using CPU fallback.")
    return -1, "CPU"


@dataclass
class ColabInvoker:
    """Thin wrapper around a Colab REST endpoint."""

    endpoint: Optional[str] = os.getenv("COLAB_HYBRID_ENDPOINT")

    def available(self) -> bool:
        return bool(self.endpoint)

    def invoke(self, payload: Dict[str, Any], timeout: int = 180) -> Optional[Dict[str, Any]]:
        if not self.endpoint:
            return None
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: broad-except
            LOGGER.warning("Colab invocation failed: %s", exc)
            return None


class LocalModelSuite:
    """Lightweight set of local Hugging Face pipelines."""

    def __init__(self) -> None:
        device, label = _detect_device()
        self.device = device
        LOGGER.info("Initialising local pipelines on %s", label)

        self.generator = pipeline(
            "text-generation",
            model="gpt2",
            device=device,
            max_length=512,
            do_sample=True,
            temperature=0.95,
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

    def analyze(self, play_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        LOGGER.info("Running local hybrid analysis pipelines.")
        generated = self.generator(
            play_text + "\nStrategic response:",
            max_length=380,
            num_return_sequences=1,
            pad_token_id=self.generator.tokenizer.eos_token_id,
        )[0]["generated_text"]

        sentiment = self.sentiment(play_text)[0]
        labels = ["aggressive blitz", "zone coverage", "pass rush", "run stuffing", "deep coverage"]
        zero_shot = self.zero_shot(play_text, labels)

        return {
            "mode": "local",
            "text_generation": generated,
            "sentiment": sentiment,
            "tactical_labels": zero_shot,
            "context": context or {},
            "logs": ["Executed on local pipelines."],
        }


class HybridFootballAI:
    """Primary controller for hybrid execution."""

    def __init__(self, prefer_colab: bool = True) -> None:
        self.prefer_colab = prefer_colab
        self.colab = ColabInvoker()
        self.local_suite = LocalModelSuite()

    def run(self, play_text: str, stats_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        payload = self._build_payload(play_text, stats_df)
        if self.prefer_colab and self.colab.available():
            LOGGER.info("Attempting Colab GPU ensemble run...")
            result = self.colab.invoke(payload)
            if result:
                result.setdefault("logs", []).append("Executed on Colab ensemble.")
                result.setdefault("mode", "colab")
                return result
            LOGGER.warning("Colab GPU run failed, falling back to local execution.")

        LOGGER.info("Using local fallback execution.")
        return self.local_suite.analyze(play_text, payload.get("context"))

    def _build_payload(self, play_text: str, stats_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        stats_preview: Optional[List[Dict[str, Any]]] = None
        if stats_df is not None and not stats_df.empty:
            stats_preview = stats_df.head(15).to_dict(orient="records")
        return {
            "play_text": play_text,
            "context": {
                "stats_preview": stats_preview,
                "prefer_colab": self.prefer_colab,
            },
        }


_shared_engine: Optional[HybridFootballAI] = None


def get_shared_engine(prefer_colab: bool = True) -> HybridFootballAI:
    """Return a cached HybridFootballAI instance."""
    global _shared_engine  # noqa: global-statement
    if _shared_engine is None or _shared_engine.prefer_colab != prefer_colab:
        _shared_engine = HybridFootballAI(prefer_colab=prefer_colab)
    return _shared_engine


def run_hybrid_analysis(play_text: str, stats_df: Optional[pd.DataFrame], prefer_colab: bool = True) -> Dict[str, Any]:
    """Convenience entrypoint used by the GUI."""
    engine = get_shared_engine(prefer_colab=prefer_colab)
    return engine.run(play_text, stats_df)


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Football AI CLI")
    parser.add_argument("--text", type=str, required=True, help="Play description text")
    parser.add_argument("--csv", type=str, help="Optional stats CSV")
    parser.add_argument("--local-only", action="store_true", help="Force local fallback")
    args = parser.parse_args()

    df = pd.read_csv(args.csv) if args.csv else None
    result = run_hybrid_analysis(args.text, df, prefer_colab=not args.local_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _cli()
