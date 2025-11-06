#!/usr/bin/env python3
"""
Quick GGUF Ensemble Test
=======================

Test script to verify your 5 downloaded GGUF models work with the
backtesting pipeline before running the full Colab GPU validation.
"""

import asyncio
import time
from pathlib import Path
from gguf_ensemble_backtest import GGUFEnsembleBacktester

async def quick_model_test():
    """Quick test of all 5 GGUF models"""
    print("🚀 QUICK GGUF MODEL TEST")
    print("=" * 40)
    
    backtester = GGUFEnsembleBacktester()
    
    # Check if models exist
    print("\n📁 Checking model files...")
    models_found = 0
    for model_key, config in backtester.model_configs.items():
        if config.file_path.exists():
            size_gb = config.file_path.stat().st_size / (1024**3)
            print(f"   ✅ {config.name}: {size_gb:.1f}GB")
            models_found += 1
        else:
            print(f"   ❌ {config.name}: Missing")
    
    print(f"\n📊 Found {models_found}/5 models")
    
    if models_found < 2:
        print("❌ Need at least 2 models to test ensemble")
        return
    
    # Quick load test (first 2 models)
    print("\n🔧 Testing model loading...")
    test_models = ["mistral", "neural_chat"] if models_found >= 2 else list(backtester.model_configs.keys())[:models_found]
    
    loaded_count = 0
    for model_key in test_models[:2]:  # Test first 2 for speed
        config = backtester.model_configs[model_key]
        if not config.file_path.exists():
            continue
            
        print(f"   Loading {config.name}...")
        try:
            from llama_cpp import Llama
            start_time = time.time()
            
            model = Llama(
                model_path=str(config.file_path),
                n_ctx=512,  # Smaller context for speed
                n_threads=4,
                verbose=False
            )
            
            load_time = time.time() - start_time
            print(f"   ✅ {config.name} loaded in {load_time:.1f}s")
            
            # Quick prediction test
            test_prompt = """Analyze this game: Team A vs Team B, spread -3.5, total 47.
Return JSON: {"make_bet": true/false, "confidence": 50}"""
            
            response = model(test_prompt, max_tokens=50, temperature=0.1)
            print(f"   🧠 Response: {response['choices'][0]['text'][:100]}...")
            
            loaded_count += 1
            del model  # Free memory
            
        except Exception as e:
            print(f"   ❌ Failed to load {config.name}: {e}")
    
    print(f"\n🎉 Successfully tested {loaded_count}/2 models")
    
    if loaded_count > 0:
        print("\n✅ MODELS READY FOR BACKTESTING!")
        print("Next steps:")
        print("1. Run full backtest: python gguf_ensemble_backtest.py")
        print("2. Or use Colab GPU: python colab_gpu_backtest.py")
    else:
        print("\n❌ Models need troubleshooting before backtesting")

async def quick_backtest_sample():
    """Run a tiny backtest with 10 games"""
    print("\n🧪 QUICK BACKTEST SAMPLE")
    print("=" * 40)
    
    backtester = GGUFEnsembleBacktester()
    
    # Override to create tiny dataset
    original_method = backtester._generate_mock_season_data
    backtester._generate_mock_season_data = lambda season, count: original_method(season, 10)  # Only 10 games
    
    try:
        # Run mini backtest
        results = await backtester.run_comprehensive_backtest(seasons=["2023"])
        
        print(f"\n🏆 SAMPLE BACKTEST RESULTS:")
        print(f"   Ensemble ROI: {results.ensemble_result.roi_percent:+.1f}%")
        print(f"   Win Rate: {results.ensemble_result.win_rate:.1%}")
        print(f"   Models Tested: {len([r for r in results.individual_results.values() if r])}")
        
        print("\n✅ ENSEMBLE PIPELINE WORKING!")
        print("Ready for full 200 GPU hour Colab run!")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔥 Starting GGUF Model Tests...")
    
    # Run quick tests
    asyncio.run(quick_model_test())
    
    # Ask if user wants to run sample backtest
    response = input("\n🤔 Run quick 10-game backtest sample? (y/N): ")
    if response.lower().startswith('y'):
        asyncio.run(quick_backtest_sample())
        
    print("\n🎉 Testing complete!")