#!/usr/bin/env python3
"""
Download GGUF Models for Local NFL Analysis
Downloads quantized models from Hugging Face
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm


# Recommended GGUF models (good balance of quality and speed)
MODELS = {
    "llama3-8b": {
        "url": "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "filename": "llama-3-8b-instruct.Q4_K_M.gguf",
        "size_gb": 4.9
    },
    "mistral-7b": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        "size_gb": 4.4
    },
    "phi3": {
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "filename": "phi-3-mini-4k-instruct.Q4_K_M.gguf",
        "size_gb": 2.4
    }
}


def download_file(url: str, filepath: Path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filepath, 'wb') as f, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            progress_bar.update(size)


def main():
    print("=" * 70)
    print("🤖 GGUF MODEL DOWNLOADER FOR NFL ANALYSIS")
    print("=" * 70)
    print()

    # Create models directory
    models_dir = Path("models/gguf")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Available models to download:\n")
    for i, (name, info) in enumerate(MODELS.items(), 1):
        print(f"{i}. {name}")
        print(f"   Size: ~{info['size_gb']:.1f} GB")
        print(f"   File: {info['filename']}")
        print()

    print(f"Models will be saved to: {models_dir.absolute()}")
    print()

    # Ask which models to download
    choice = input("Download: (1) All models, (2) Select specific, (3) Exit: ").strip()

    if choice == "3":
        print("Exiting...")
        return

    to_download = []

    if choice == "1":
        to_download = list(MODELS.keys())
    elif choice == "2":
        print("\nEnter model numbers (comma-separated, e.g., 1,3):")
        selected = input("> ").strip()
        try:
            indices = [int(x.strip()) for x in selected.split(',')]
            model_names = list(MODELS.keys())
            to_download = [model_names[i-1] for i in indices if 1 <= i <= len(model_names)]
        except:
            print("Invalid input!")
            return
    else:
        print("Invalid choice!")
        return

    if not to_download:
        print("No models selected!")
        return

    # Download models
    print("\n" + "=" * 70)
    print("DOWNLOADING MODELS")
    print("=" * 70)
    print()

    total_size = sum(MODELS[name]['size_gb'] for name in to_download)
    print(f"Total download size: ~{total_size:.1f} GB")
    print()

    for name in to_download:
        info = MODELS[name]
        filepath = models_dir / info['filename']

        if filepath.exists():
            print(f"✅ {name} already exists, skipping...")
            continue

        print(f"📥 Downloading {name}...")
        try:
            download_file(info['url'], filepath)
            print(f"✅ {name} downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")

    print("\n" + "=" * 70)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 70)
    print()
    print("You can now run:")
    print("  python3 gguf_nfl_analyzer.py")
    print()


if __name__ == "__main__":
    main()
