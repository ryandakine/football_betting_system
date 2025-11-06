#!/usr/bin/env python3
"""
Practical GGUF Ensemble Setup
============================

Sets up a practical ensemble of smaller, efficient GGUF models for football betting.
These models are sized 2-4GB each and can run together on most systems.
"""

import os
import sys
import time
from pathlib import Path
import logging
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimized model selection: smaller, faster, diverse
RECOMMENDED_MODELS = [
    {
        "name": "mistral_7b_instruct_q4",
        "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_gb": 4.1,
        "specialty": "General instruction following and reasoning",
        "performance": 4.2
    },
    {
        "name": "codellama_7b_instruct_q4", 
        "repo": "TheBloke/CodeLlama-7B-Instruct-GGUF",
        "filename": "codellama-7b-instruct.Q4_K_M.gguf",
        "size_gb": 3.8,
        "specialty": "Analytical reasoning and structured output",
        "performance": 4.0
    },
    {
        "name": "openchat_7b_q4",
        "repo": "TheBloke/openchat-3.5-0106-GGUF", 
        "filename": "openchat-3.5-0106.Q4_K_M.gguf",
        "size_gb": 3.9,
        "specialty": "Conversational analysis and nuanced reasoning",
        "performance": 4.1
    },
    {
        "name": "neural_chat_7b_q4",
        "repo": "TheBloke/neural-chat-7B-v3-3-GGUF",
        "filename": "neural-chat-7b-v3-3.Q4_K_M.gguf", 
        "size_gb": 3.9,
        "specialty": "Sports analysis and betting insights",
        "performance": 3.9
    },
    {
        "name": "dolphin_mistral_7b_q4",
        "repo": "TheBloke/dolphin-2.6-mistral-7B-GGUF",
        "filename": "dolphin-2.6-mistral-7b.Q4_K_M.gguf",
        "size_gb": 4.0,
        "specialty": "Uncensored analysis and contrarian views", 
        "performance": 4.0
    }
]

def check_system_resources():
    """Check if system can handle the ensemble."""
    try:
        import psutil
        total_ram = psutil.virtual_memory().total / (1024**3)  # GB
        available_ram = psutil.virtual_memory().available / (1024**3)
        
        logger.info(f"💾 System RAM: {total_ram:.1f}GB total, {available_ram:.1f}GB available")
        
        # Estimate needs: models + system overhead + inference
        total_model_size = sum(model["size_gb"] for model in RECOMMENDED_MODELS)
        estimated_need = total_model_size * 1.5  # Add 50% overhead
        
        if available_ram < estimated_need:
            logger.warning(f"⚠️ Tight on RAM: need ~{estimated_need:.1f}GB, have {available_ram:.1f}GB")
            logger.info("Consider downloading fewer models or adding more RAM")
            return False
        else:
            logger.info(f"✅ RAM looks good: need ~{estimated_need:.1f}GB, have {available_ram:.1f}GB")
            return True
            
    except ImportError:
        logger.warning("psutil not available - can't check RAM")
        return True

def download_model(model_config, force=False):
    """Download a single model."""
    model_path = Path(f"models/gguf/{model_config['filename']}")
    
    if model_path.exists() and not force:
        size_gb = model_path.stat().st_size / (1024**3)
        logger.info(f"✅ {model_config['name']} already exists ({size_gb:.1f}GB)")
        return model_path
    
    logger.info(f"📥 Downloading {model_config['name']} ({model_config['size_gb']}GB)")
    logger.info(f"    Purpose: {model_config['specialty']}")
    
    try:
        # Ensure directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        downloaded_path = hf_hub_download(
            repo_id=model_config["repo"],
            filename=model_config["filename"],
            cache_dir="models/cache",
            local_dir="models/gguf",
            local_dir_use_symlinks=False
        )
        
        # Verify download
        if model_path.exists():
            actual_size = model_path.stat().st_size / (1024**3)
            logger.info(f"✅ Downloaded {model_config['name']} ({actual_size:.1f}GB)")
            return model_path
        else:
            logger.error(f"❌ Download failed for {model_config['name']}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to download {model_config['name']}: {e}")
        return None

def test_model_loading(model_path, model_name):
    """Quick test that model can be loaded."""
    try:
        from llama_cpp import Llama
        
        logger.info(f"🧪 Testing {model_name}...")
        
        # Load with minimal resources for testing
        test_model = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_gpu_layers=0,  # CPU for safety during test
            verbose=False,
        )
        
        # Quick inference test
        response = test_model(
            "Analyze: Cowboys vs Giants. JSON:",
            max_tokens=50,
            temperature=0.1,
        )
        
        logger.info(f"✅ {model_name} loaded and tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ {model_name} test failed: {e}")
        return False

def create_ensemble_config():
    """Create configuration file for the ensemble."""
    config = {
        "gguf_ensemble": {
            "models_dir": "./models/gguf/",
            "max_concurrent_models": 3,  # Load 3 at a time to manage memory
            "rotation_strategy": "performance_weighted",
            "models": []
        }
    }
    
    for model in RECOMMENDED_MODELS:
        model_path = Path(f"models/gguf/{model['filename']}")
        if model_path.exists():
            config["gguf_ensemble"]["models"].append({
                "name": model["name"],
                "filename": model["filename"],
                "specialty": model["specialty"], 
                "performance_weight": model["performance"],
                "size_gb": model["size_gb"],
                "max_context": 4096,
                "temperature": 0.3,
                "top_p": 0.9
            })
    
    config_path = Path("models/ensemble_config.json")
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Created ensemble config: {config_path}")
    return config

def main():
    """Main setup function."""
    print("🏈 Setting up Practical GGUF Ensemble for Football Betting")
    print("=" * 60)
    
    # Check system resources
    if not check_system_resources():
        response = input("\n⚠️ System resources may be tight. Continue anyway? (y/N): ").lower()
        if response != 'y':
            print("Setup cancelled. Consider upgrading RAM or selecting fewer models.")
            return
    
    # Check llama-cpp-python
    try:
        from llama_cpp import Llama
        logger.info("✅ llama-cpp-python is available")
    except ImportError:
        logger.error("❌ llama-cpp-python not installed")
        logger.info("Install with: pip install llama-cpp-python")
        return
    
    # Interactive model selection
    print(f"\n📦 Available models ({len(RECOMMENDED_MODELS)} total):")
    for i, model in enumerate(RECOMMENDED_MODELS, 1):
        print(f"{i}. {model['name']} ({model['size_gb']}GB) - {model['specialty']}")
    
    print("\nRecommended: Download 3-4 models for best diversity")
    selection = input("\nSelect models to download (e.g., 1,2,3 or 'all'): ").strip()
    
    if selection.lower() == 'all':
        selected_models = RECOMMENDED_MODELS
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_models = [RECOMMENDED_MODELS[i] for i in indices if 0 <= i < len(RECOMMENDED_MODELS)]
        except:
            logger.error("Invalid selection format")
            return
    
    if not selected_models:
        logger.error("No models selected")
        return
    
    # Download selected models
    successful_downloads = []
    total_size = sum(m["size_gb"] for m in selected_models)
    
    print(f"\n📥 Downloading {len(selected_models)} models (~{total_size:.1f}GB total)")
    print("This may take 10-30 minutes depending on your connection...")
    
    for model in selected_models:
        model_path = download_model(model)
        if model_path:
            if test_model_loading(model_path, model["name"]):
                successful_downloads.append(model)
        print()  # Spacing
    
    # Create ensemble configuration
    if successful_downloads:
        config = create_ensemble_config()
        
        print(f"\n🎉 Ensemble Setup Complete!")
        print("=" * 40)
        print(f"✅ Downloaded {len(successful_downloads)} models")
        print(f"📁 Models stored in: models/gguf/")
        print(f"⚙️ Config created: models/ensemble_config.json")
        
        total_downloaded = sum(m["size_gb"] for m in successful_downloads)
        print(f"💾 Total size: {total_downloaded:.1f}GB")
        
        print(f"\n🚀 Next steps:")
        print("1. python test_practical_ensemble.py")
        print("2. Integrate with your football betting system")
        print("3. Models will auto-rotate based on performance")
        
    else:
        print("\n❌ No models downloaded successfully")
        print("Check your internet connection and try again")

if __name__ == "__main__":
    main()