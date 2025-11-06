#!/usr/bin/env python3
"""
GGUF Model Setup Script for Football Betting System
=================================================

This script helps you set up the DavidAU 20B GGUF model for use as a fallback
in your football betting ensemble system.
"""

import os
import sys
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_llama_cpp():
    """Check if llama-cpp-python is installed and working."""
    try:
        from llama_cpp import Llama
        logger.info("✅ llama-cpp-python is installed and available")
        return True
    except ImportError:
        logger.error("❌ llama-cpp-python is not installed")
        logger.info("Install it with:")
        logger.info("  pip install --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/ llama-cpp-python")
        return False

def check_model_file():
    """Check if the GGUF model file exists in the expected location."""
    possible_paths = [
        "./models/gguf/OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf",
        "~/Downloads/OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf",
        "~/Desktop/OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf",
    ]
    
    for path_str in possible_paths:
        path = Path(path_str).expanduser()
        if path.exists():
            logger.info(f"✅ Found GGUF model at: {path}")
            return path
    
    logger.error("❌ GGUF model file not found")
    logger.info("Expected locations:")
    for path_str in possible_paths:
        logger.info(f"  - {Path(path_str).expanduser()}")
    
    logger.info("\nTo download the model:")
    logger.info("1. Go to: https://huggingface.co/DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf")
    logger.info("2. Download: OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf")
    logger.info("3. Copy it to: ./models/gguf/")
    
    return None

def setup_model_directory():
    """Ensure the models/gguf directory exists."""
    models_dir = Path("./models/gguf")
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Model directory ready: {models_dir.absolute()}")
    return models_dir

def copy_model_to_correct_location(source_path, target_dir):
    """Copy the GGUF model to the correct location."""
    target_path = target_dir / "OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf"
    
    if target_path.exists():
        logger.info(f"✅ Model already in correct location: {target_path}")
        return target_path
    
    if source_path and source_path.exists():
        logger.info(f"📁 Copying model from {source_path} to {target_path}")
        try:
            shutil.copy2(source_path, target_path)
            logger.info("✅ Model copied successfully")
            return target_path
        except Exception as e:
            logger.error(f"❌ Failed to copy model: {e}")
            return None
    
    return None

def test_model_loading(model_path):
    """Test that the GGUF model can be loaded successfully."""
    try:
        from llama_cpp import Llama
        logger.info(f"🧪 Testing model loading: {model_path}")
        
        # Try to load with minimal resources for testing
        test_model = Llama(
            model_path=str(model_path),
            n_ctx=512,  # Small context for testing
            n_gpu_layers=0,  # CPU only for safety
            verbose=False,
        )
        
        # Try a simple inference
        response = test_model(
            "Test prompt",
            max_tokens=10,
            temperature=0.1,
        )
        
        logger.info("✅ Model loaded and tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        return False

def update_env_file(model_path):
    """Update .env file with GGUF model path."""
    env_file = Path(".env")
    env_lines = []
    gguf_line_found = False
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_lines = f.readlines()
    
    # Update or add GGUF_MODEL_PATH
    new_lines = []
    for line in env_lines:
        if line.strip().startswith('GGUF_MODEL_PATH='):
            new_lines.append(f'GGUF_MODEL_PATH={model_path}\n')
            gguf_line_found = True
        else:
            new_lines.append(line)
    
    if not gguf_line_found:
        new_lines.append(f'GGUF_MODEL_PATH={model_path}\n')
    
    with open(env_file, 'w') as f:
        f.writelines(new_lines)
    
    logger.info(f"✅ Updated .env file with GGUF_MODEL_PATH={model_path}")

def test_ensemble_integration():
    """Test that the ensemble system can find and use the GGUF model."""
    try:
        from huggingface_cloud_gpu import CloudGPUAIEnsemble
        
        # Test configuration
        config = {
            "gguf_fallback_path": "./models/gguf/OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf"
        }
        
        ensemble = CloudGPUAIEnsemble(config=config)
        
        if ensemble.gguf_model:
            logger.info("✅ Ensemble system successfully loaded GGUF model!")
            return True
        else:
            logger.warning("⚠️ Ensemble system initialized but GGUF model not loaded")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ensemble integration test failed: {e}")
        return False

def print_usage_instructions():
    """Print instructions for using the GGUF model."""
    logger.info("\n" + "="*60)
    logger.info("🎯 GGUF MODEL SETUP COMPLETE!")
    logger.info("="*60)
    logger.info("Your football betting system now has GGUF fallback support.")
    logger.info("\nTo use it:")
    logger.info("1. The GGUF model will automatically be used as fallback")
    logger.info("2. If cloud models fail, the system switches to local GGUF")
    logger.info("3. You can force GGUF-only mode by setting cloud models to fail")
    logger.info("\nTest it:")
    logger.info("  python -c \"from huggingface_cloud_gpu import CloudGPUAIEnsemble; e=CloudGPUAIEnsemble(); print('GGUF loaded:', bool(e.gguf_model))\"")
    logger.info("\nMonitor usage:")
    logger.info("  - Look for '🛡️ Falling back to local GGUF model' in logs")
    logger.info("  - GGUF responses will have 'gguf_fallback' in models_used")
    
def main():
    """Main setup function."""
    print("🚀 Setting up GGUF model for football betting system...")
    print("-" * 60)
    
    # Step 1: Check llama-cpp-python
    if not check_llama_cpp():
        sys.exit(1)
    
    # Step 2: Setup model directory
    models_dir = setup_model_directory()
    
    # Step 3: Check for existing model file
    model_path = check_model_file()
    
    # Step 4: Copy model if found elsewhere
    if model_path and str(model_path) != str(models_dir / "OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf"):
        model_path = copy_model_to_correct_location(model_path, models_dir)
    elif not model_path:
        target_path = models_dir / "OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf"
        if target_path.exists():
            model_path = target_path
            logger.info(f"✅ Found model at target location: {model_path}")
    
    if not model_path or not model_path.exists():
        logger.error("❌ Cannot proceed without GGUF model file")
        logger.info("\nNext steps:")
        logger.info("1. Download the model from Google Drive or HuggingFace")
        logger.info("2. Copy it to: ./models/gguf/OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf")
        logger.info("3. Run this script again")
        sys.exit(1)
    
    # Step 5: Test model loading
    if not test_model_loading(model_path):
        logger.error("❌ Model test failed - check your installation")
        sys.exit(1)
    
    # Step 6: Update environment
    update_env_file(model_path)
    
    # Step 7: Test ensemble integration
    if test_ensemble_integration():
        logger.info("✅ Full integration test passed!")
    else:
        logger.warning("⚠️ Integration test had issues - check logs")
    
    # Step 8: Print usage instructions
    print_usage_instructions()

if __name__ == "__main__":
    main()