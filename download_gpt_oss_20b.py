#!/usr/bin/env python3
"""
Helper script to download DavidAU's 20B uncensored GGUF checkpoint for local use.

Usage:
    HF_TOKEN=your_token python download_gpt_oss_20b.py

The model will be stored under `models/gguf/` by default.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf"
FILENAME = "OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf"
OUTPUT_DIR = Path("models/gguf")


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN environment variable is not set.\n"
            "Create a Hugging Face access token with repository permissions and run:\n"
            "  export HF_TOKEN=hf_xxx\n"
            "before executing this script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, token=token, local_dir=str(OUTPUT_DIR))
    print(f"Model downloaded to {model_path}")


if __name__ == "__main__":
    main()
