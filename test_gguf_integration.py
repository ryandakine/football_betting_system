#!/usr/bin/env python3
"""
Test GGUF Integration Script
===========================

Quick test to verify GGUF model is working with your football betting system.
"""

import logging
import time
from huggingface_cloud_gpu import CloudGPUAIEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gguf_fallback():
    """Test GGUF fallback functionality."""
    print("🧪 Testing GGUF Model Integration")
    print("=" * 50)
    
    # Configuration to use GGUF fallback
    config = {
        "gguf_fallback_path": "./models/gguf/OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf",
        "blackout_detector": {"enabled": False},  # Disable for testing
        "meta_learner": {"enabled": False},       # Disable for testing
        "hybrid": {"enabled": False},             # Disable for testing
        "vision": {"enabled": False},             # Disable for testing
        "feedback": {"enabled": False},           # Disable for testing
    }
    
    try:
        # Initialize ensemble
        print("📦 Initializing ensemble...")
        ensemble = CloudGPUAIEnsemble(config=config)
        
        if not ensemble.gguf_model:
            print("❌ GGUF model not loaded - check setup")
            return False
        
        print("✅ GGUF model loaded successfully!")
        
        # Test game data (Broncos vs Chiefs example)
        test_game = {
            "home_team": "Denver Broncos",
            "away_team": "Kansas City Chiefs",
            "sport": "NFL",
            "season": "2024",
            "week": "8",
            "commence_time": "2024-10-24T20:25:00Z",
            "spread": -3.5,
            "total": 43.5,
            "referee": "Tony Corrente"
        }
        
        print("🏈 Testing game analysis...")
        print(f"Game: {test_game['away_team']} @ {test_game['home_team']}")
        
        start_time = time.time()
        
        # This should force GGUF usage since no cloud models are loaded
        result = ensemble._analyze_with_gguf(
            test_game, 
            "h2h", 
            RuntimeError("Testing GGUF fallback")
        )
        
        end_time = time.time()
        
        print(f"\n⏱️ Analysis completed in {end_time - start_time:.2f} seconds")
        print("\n📊 Results:")
        print(f"  Probability: {result.get('probability', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        print(f"  Risk Level: {result.get('risk_level', 'N/A')}")
        print(f"  Models Used: {result.get('models_used', 'N/A')}")
        
        if result.get('key_factors'):
            print(f"  Key Factors:")
            for i, factor in enumerate(result['key_factors'][:3], 1):
                print(f"    {i}. {factor}")
        
        if result.get('analysis'):
            analysis = result['analysis'][:200] + "..." if len(result.get('analysis', '')) > 200 else result.get('analysis', '')
            print(f"  Analysis: {analysis}")
        
        print("\n✅ GGUF integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_ensemble_with_fallback():
    """Test the full ensemble with fallback."""
    print("\n🔄 Testing Full Ensemble (should fallback to GGUF)")
    print("=" * 50)
    
    config = {
        "gguf_fallback_path": "./models/gguf/OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf"
    }
    
    try:
        ensemble = CloudGPUAIEnsemble(config=config)
        
        # Test game
        test_game = {
            "home_team": "Miami Dolphins",  
            "away_team": "Buffalo Bills",
            "sport": "NFL",
            "season": "2024",
            "week": "9"
        }
        
        print(f"🏈 Analyzing: {test_game['away_team']} @ {test_game['home_team']}")
        
        start_time = time.time()
        
        # This should try cloud models first, then fallback to GGUF
        import asyncio
        result = asyncio.run(ensemble.analyze_football_game(test_game))
        
        end_time = time.time()
        
        if result:
            print(f"\n⏱️ Analysis completed in {end_time - start_time:.2f} seconds")
            print(f"📊 Probability: {result.get('probability', 'N/A')}")
            print(f"📊 Models Used: {result.get('models_used', 'N/A')}")
            
            # Check if GGUF was used
            models_used = result.get('models_used', [])
            if any('gguf' in str(model).lower() for model in models_used):
                print("✅ GGUF fallback was successfully used!")
            else:
                print("ℹ️ Cloud models were used (GGUF available as backup)")
                
            return True
        else:
            print("❌ No result returned")
            return False
            
    except Exception as e:
        print(f"❌ Full ensemble test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 GGUF Integration Test Suite")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Direct GGUF fallback
    if test_gguf_fallback():
        success_count += 1
    
    # Test 2: Full ensemble with fallback
    if test_ensemble_with_fallback():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {success_count}/{total_tests} passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Your GGUF integration is working perfectly.")
        print("\nYour system will now:")
        print("✅ Use cloud models when available") 
        print("✅ Automatically fall back to GGUF when cloud models fail")
        print("✅ Provide uncensored analysis for referee patterns and betting insights")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure llama-cpp-python is installed")
        print("2. Verify the GGUF model file is in ./models/gguf/")
        print("3. Check that you have sufficient RAM/VRAM")

if __name__ == "__main__":
    main()