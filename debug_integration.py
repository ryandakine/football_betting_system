#!/usr/bin/env python3
"""
Comprehensive Debug Script for Football Betting System Integration
================================================================

Tests the integration between:
1. Legacy Model Components:
   - EnhancedNFLWithSocialAnalysis
   - EnhancedGPUAnalyzer
   - FootballProductionBettingSystem
   - AdvancedNFLAnalysis
   - GPUNFLWeekendAnalyzer

2. TaskMaster Real-Time Components:
   - MultiProviderWebSocketManager
   - EventQueue
   - StreamProcessor
   - BehavioralIntelligenceEngine
   - MarketIntelligenceSystem
   - PortfolioManagementSystem
   - SelfImprovingLoop

This script will identify import errors, missing dependencies, and integration issues.
"""

import sys
import traceback
from datetime import datetime
import importlib

def test_import(module_name, class_name=None, friendly_name=None):
    """Test if a module or class can be imported successfully"""
    if friendly_name is None:
        friendly_name = module_name if class_name is None else f"{module_name}.{class_name}"
    
    try:
        module = importlib.import_module(module_name)
        if class_name:
            getattr(module, class_name)
        print(f"✅ {friendly_name} - Import successful")
        return True
    except ImportError as e:
        print(f"❌ {friendly_name} - Import failed: {e}")
        return False
    except AttributeError as e:
        print(f"❌ {friendly_name} - Class not found: {e}")
        return False
    except Exception as e:
        print(f"❌ {friendly_name} - Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_legacy_systems():
    """Test all legacy model components"""
    print("\n🔍 Testing Legacy Model Components")
    print("=" * 50)
    
    legacy_components = [
        ("enhanced_nfl_with_social", "EnhancedNFLWithSocialAnalysis", "EnhancedNFLWithSocialAnalysis"),
        ("enhanced_gpu_nfl_analyzer", "EnhancedGPUAnalyzer", "EnhancedGPUAnalyzer"),
        ("football_production_main", "FootballProductionBettingSystem", "FootballProductionBettingSystem"),
        ("advanced_nfl_analysis", "AdvancedNFLAnalysis", "AdvancedNFLAnalysis"),
        ("gpu_nfl_weekend_analyzer", None, "GPUNFLWeekendAnalyzer (module)"),
    ]
    
    success_count = 0
    for module, class_name, friendly_name in legacy_components:
        if test_import(module, class_name, friendly_name):
            success_count += 1
    
    print(f"\nLegacy Systems: {success_count}/{len(legacy_components)} working")
    return success_count == len(legacy_components)

def test_taskmaster_systems():
    """Test all TaskMaster real-time components"""
    print("\n🔍 Testing TaskMaster Real-Time Components")
    print("=" * 50)
    
    taskmaster_components = [
        ("realtime_websocket_client", "MultiProviderWebSocketManager", "MultiProviderWebSocketManager"),
        ("event_driven_message_queue", "EventQueue", "EventQueue"),
        ("stream_processing_engine", "StreamProcessor", "StreamProcessor"),
        ("behavioral_intelligence_engine", "BehavioralIntelligenceEngine", "BehavioralIntelligenceEngine"),
        ("market_intelligence_system", "MarketIntelligenceSystem", "MarketIntelligenceSystem"),
        ("portfolio_management_system", "PortfolioManagementSystem", "PortfolioManagementSystem"),
        ("self_improving_loop", "SelfImprovingLoop", "SelfImprovingLoop"),
    ]
    
    success_count = 0
    for module, class_name, friendly_name in taskmaster_components:
        if test_import(module, class_name, friendly_name):
            success_count += 1
    
    print(f"\nTaskMaster Systems: {success_count}/{len(taskmaster_components)} working")
    return success_count == len(taskmaster_components)

def test_unified_system():
    """Test the unified system integration"""
    print("\n🔍 Testing Unified System Integration")
    print("=" * 50)
    
    try:
        from unified_nfl_intelligence_system import UnifiedNFLIntelligenceSystem
        print("✅ UnifiedNFLIntelligenceSystem - Import successful")
        
        # Test instantiation (this will reveal runtime errors)
        try:
            system = UnifiedNFLIntelligenceSystem(bankroll=1000.0)
            print("✅ UnifiedNFLIntelligenceSystem - Instantiation successful")
            return True
        except Exception as e:
            print(f"❌ UnifiedNFLIntelligenceSystem - Instantiation failed: {e}")
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"❌ UnifiedNFLIntelligenceSystem - Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ UnifiedNFLIntelligenceSystem - Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Test critical dependencies"""
    print("\n🔍 Testing Critical Dependencies")
    print("=" * 50)
    
    dependencies = [
        ("torch", None, "PyTorch"),
        ("numpy", None, "NumPy"),
        ("pandas", None, "Pandas"),
        ("asyncio", None, "AsyncIO"),
        ("json", None, "JSON"),
        ("logging", None, "Logging"),
        ("datetime", None, "DateTime"),
        ("typing", None, "Typing"),
    ]
    
    success_count = 0
    for module, class_name, friendly_name in dependencies:
        if test_import(module, class_name, friendly_name):
            success_count += 1
    
    print(f"\nDependencies: {success_count}/{len(dependencies)} working")
    return success_count == len(dependencies)

def generate_report(legacy_ok, taskmaster_ok, unified_ok, deps_ok):
    """Generate a comprehensive report of the debug results"""
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE DEBUG REPORT")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Debug Time: {timestamp}")
    
    print(f"\n📋 System Status:")
    print(f"   Legacy Model Components: {'✅ OK' if legacy_ok else '❌ ISSUES'}")
    print(f"   TaskMaster Components: {'✅ OK' if taskmaster_ok else '❌ ISSUES'}")
    print(f"   Unified System: {'✅ OK' if unified_ok else '❌ ISSUES'}")
    print(f"   Dependencies: {'✅ OK' if deps_ok else '❌ ISSUES'}")
    
    overall_status = all([legacy_ok, taskmaster_ok, unified_ok, deps_ok])
    print(f"\n🎯 Overall Status: {'✅ READY' if overall_status else '❌ NEEDS FIXES'}")
    
    if not overall_status:
        print("\n🔧 Recommended Actions:")
        if not legacy_ok:
            print("   • Fix legacy model component imports")
        if not taskmaster_ok:
            print("   • Fix TaskMaster component imports")
        if not unified_ok:
            print("   • Resolve unified system integration issues")
        if not deps_ok:
            print("   • Install missing dependencies")
    
    return overall_status

def main():
    """Main debug function"""
    print("🚀 Football Betting System Integration Debug")
    print("=" * 60)
    
    # Test all components
    legacy_ok = test_legacy_systems()
    taskmaster_ok = test_taskmaster_systems()
    unified_ok = test_unified_system()
    deps_ok = test_dependencies()
    
    # Generate report
    overall_ok = generate_report(legacy_ok, taskmaster_ok, unified_ok, deps_ok)
    
    # Exit with appropriate code
    sys.exit(0 if overall_ok else 1)

if __name__ == "__main__":
    main()
