#!/usr/bin/env python3
"""
Test Practical GGUF Ensemble
============================

Test the memory-efficient ensemble of smaller GGUF models.
"""

import time
import json
from practical_gguf_ensemble import PracticalGGUFEnsemble

def test_single_model():
    """Test single model prediction."""
    print("🧪 Testing Single Model Prediction")
    print("=" * 50)
    
    try:
        ensemble = PracticalGGUFEnsemble()
        
        test_game = {
            "home_team": "Denver Broncos",
            "away_team": "Kansas City Chiefs", 
            "season": "2024",
            "week": "8",
            "spread": -3.5,
            "total": 43.5,
            "referee": "Tony Corrente"
        }
        
        print(f"🏈 Game: {test_game['away_team']} @ {test_game['home_team']}")
        print("📊 Getting single model prediction...")
        
        start_time = time.time()
        result = ensemble.get_prediction(test_game)
        end_time = time.time()
        
        if result:
            print(f"\n⏱️ Prediction time: {end_time - start_time:.2f} seconds")
            print(f"📈 Probability: {result['probability']:.3f}")
            print(f"🎯 Confidence: {result['confidence']:.3f}")
            print(f"⚠️ Risk Level: {result['risk_level']}")
            print(f"💡 Key Factors:")
            for i, factor in enumerate(result.get('key_factors', [])[:3], 1):
                print(f"   {i}. {factor}")
            print(f"📝 Analysis: {result['analysis'][:150]}...")
            print("✅ Single model test passed!")
            return True
        else:
            print("❌ No prediction returned")
            return False
            
    except Exception as e:
        print(f"❌ Single model test failed: {e}")
        return False

def test_ensemble_prediction():
    """Test ensemble prediction with multiple models."""
    print("\n🔄 Testing Ensemble Prediction")
    print("=" * 50)
    
    try:
        ensemble = PracticalGGUFEnsemble()
        
        test_game = {
            "home_team": "Miami Dolphins",
            "away_team": "Buffalo Bills",
            "season": "2024", 
            "week": "9",
            "spread": -6.0,
            "total": 48.5
        }
        
        print(f"🏈 Game: {test_game['away_team']} @ {test_game['home_team']}")
        print("📊 Getting ensemble prediction (3 models)...")
        
        start_time = time.time()
        result = ensemble.get_ensemble_prediction(test_game, num_models=3)
        end_time = time.time()
        
        if result:
            print(f"\n⏱️ Ensemble time: {end_time - start_time:.2f} seconds")
            print(f"📈 Final Probability: {result['probability']:.3f}")
            print(f"🎯 Final Confidence: {result['confidence']:.3f}")
            print(f"🤖 Models Used: {', '.join(result['models_used'])}")
            print(f"🎭 Ensemble Size: {result['ensemble_size']}")
            print(f"💡 Key Factors:")
            for i, factor in enumerate(result.get('key_factors', [])[:4], 1):
                print(f"   {i}. {factor}")
            print(f"📝 Analysis: {result['analysis'][:200]}...")
            print("✅ Ensemble test passed!")
            return True
        else:
            print("❌ No ensemble prediction returned")
            return False
            
    except Exception as e:
        print(f"❌ Ensemble test failed: {e}")
        return False

def test_model_management():
    """Test model loading/unloading management."""
    print("\n⚙️ Testing Model Management")
    print("=" * 50)
    
    try:
        ensemble = PracticalGGUFEnsemble()
        
        # Check initial status
        status = ensemble.get_model_status()
        print(f"📦 Available models: {status['available_models']}")
        print(f"🔄 Max concurrent: {status['max_concurrent']}")
        print(f"💾 Currently loaded: {len(status['loaded_models'])}")
        
        # Load models by making predictions
        test_games = [
            {"home_team": "Cowboys", "away_team": "Giants", "season": "2024"},
            {"home_team": "Packers", "away_team": "Bears", "season": "2024"},
        ]
        
        for game in test_games:
            print(f"\n🏈 Analyzing {game['away_team']} @ {game['home_team']}...")
            result = ensemble.get_prediction(game)
            if result:
                print(f"   ✅ Got prediction ({result['probability']:.3f})")
            else:
                print("   ❌ Failed")
        
        # Check final status
        final_status = ensemble.get_model_status()
        print(f"\n📊 Final Status:")
        print(f"   💾 Loaded models: {final_status['loaded_models']}")
        
        for model_name, stats in final_status['model_stats'].items():
            if stats['loaded']:
                print(f"   🤖 {model_name}: {stats['specialty'][:30]}...")
        
        print("✅ Model management test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model management test failed: {e}")
        return False

def test_performance_tracking():
    """Test performance tracking and updates."""
    print("\n📈 Testing Performance Tracking")
    print("=" * 50)
    
    try:
        ensemble = PracticalGGUFEnsemble()
        
        # Make a prediction to get a model loaded
        test_game = {
            "home_team": "Ravens", "away_team": "Steelers",
            "season": "2024", "week": "10"
        }
        
        result = ensemble.get_prediction(test_game)
        if not result:
            print("❌ Could not get initial prediction")
            return False
        
        # Simulate performance updates
        status = ensemble.get_model_status()
        loaded_models = status['loaded_models']
        
        if loaded_models:
            model_name = loaded_models[0]
            print(f"📊 Updating performance for: {model_name}")
            
            # Simulate some correct and incorrect predictions
            ensemble.update_model_performance(model_name, True, 0.8)
            ensemble.update_model_performance(model_name, True, 0.7)
            ensemble.update_model_performance(model_name, False, 0.6)
            ensemble.update_model_performance(model_name, True, 0.9)
            
            # Check updated stats
            final_status = ensemble.get_model_status()
            model_stats = final_status['model_stats'][model_name]
            
            print(f"   Games analyzed: {model_stats['games_analyzed']}")
            print(f"   Accuracy: {model_stats['accuracy']:.3f}")
            print(f"   Avg confidence: {model_stats['avg_confidence']:.3f}")
            print(f"   Avg response time: {model_stats['avg_response_time']:.2f}s")
            
            print("✅ Performance tracking test passed!")
            return True
        else:
            print("❌ No models loaded for performance testing")
            return False
            
    except Exception as e:
        print(f"❌ Performance tracking test failed: {e}")
        return False

def benchmark_speed():
    """Benchmark inference speed."""
    print("\n⚡ Speed Benchmark")
    print("=" * 50)
    
    try:
        ensemble = PracticalGGUFEnsemble()
        
        test_games = [
            {"home_team": "Jets", "away_team": "Patriots", "season": "2024"},
            {"home_team": "Eagles", "away_team": "Commanders", "season": "2024"},
            {"home_team": "49ers", "away_team": "Seahawks", "season": "2024"},
        ]
        
        times = []
        
        for i, game in enumerate(test_games, 1):
            print(f"🏈 Benchmark {i}/3: {game['away_team']} @ {game['home_team']}")
            
            start_time = time.time()
            result = ensemble.get_prediction(game)
            end_time = time.time()
            
            if result:
                response_time = end_time - start_time
                times.append(response_time)
                print(f"   ⏱️ {response_time:.2f}s - Probability: {result['probability']:.3f}")
            else:
                print(f"   ❌ Failed")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\n📊 Speed Results:")
            print(f"   Average: {avg_time:.2f}s")
            print(f"   Fastest: {min_time:.2f}s")
            print(f"   Slowest: {max_time:.2f}s")
            
            if avg_time < 10.0:
                print("✅ Good performance (< 10s average)")
            elif avg_time < 30.0:
                print("⚠️ Acceptable performance (10-30s)")
            else:
                print("❌ Slow performance (> 30s)")
            
            return True
        else:
            print("❌ No successful predictions for benchmark")
            return False
            
    except Exception as e:
        print(f"❌ Speed benchmark failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Practical GGUF Ensemble Test Suite")
    print("=" * 60)
    
    tests = [
        ("Single Model", test_single_model),
        ("Ensemble Prediction", test_ensemble_prediction),
        ("Model Management", test_model_management),
        ("Performance Tracking", test_performance_tracking),
        ("Speed Benchmark", benchmark_speed)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # Spacing between tests
        except KeyboardInterrupt:
            print("\n⏹️ Tests interrupted by user")
            break
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"🏁 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Your ensemble is working perfectly.")
        print("\nNext steps:")
        print("1. Integrate with your main betting system")
        print("2. Set up automated model performance tracking")
        print("3. Consider adding more specialized models")
    elif passed > 0:
        print("⚠️ Some tests passed. Check the failures above.")
    else:
        print("❌ All tests failed. Check your setup:")
        print("1. Run: python setup_practical_gguf_ensemble.py")
        print("2. Make sure llama-cpp-python is installed")
        print("3. Verify you have sufficient RAM")

if __name__ == "__main__":
    main()