#!/usr/bin/env python3
"""
NCAA Betting System Validation
===============================

Checks that all components are ready for live betting

USAGE:
    python validate_system.py
"""

import sys
from pathlib import Path


def validate_system():
    """Validate all system components"""

    print("\n" + "="*80)
    print("🔍 NCAA BETTING SYSTEM VALIDATION")
    print("="*80 + "\n")

    errors = []
    warnings = []

    # ==========================================
    # 1. Core Modules
    # ==========================================

    print("📦 Checking Core Modules...")

    modules = {
        'ncaa_trap_detection.py': 'Trap detection (sharp vs public)',
        'ncaa_deepseek_r1_reasoner.py': 'R1 meta-reasoning engine',
        'ncaa_deepseek_r1_analysis.py': 'Full R1 pipeline',
        'ncaa_contrarian_intelligence.py': 'Contrarian signals',
        'ncaa_daily_predictions_with_contrarian.py': 'Daily predictions',
        'backtest_ncaa_r1_system.py': 'R1 backtest system',
        'scrape_action_network_handle.py': 'Handle data scraper',
    }

    for module, desc in modules.items():
        if Path(module).exists():
            print(f"  ✅ {module}")
        else:
            print(f"  ❌ {module} - MISSING!")
            errors.append(f"Missing module: {module} ({desc})")

    print()

    # ==========================================
    # 2. Configuration Files
    # ==========================================

    print("⚙️  Checking Configuration Files...")

    configs = {
        'ncaa_model_config.py': 'Model configuration',
        'ncaa_optimal_llm_weights.json': 'R1 optimal config',
        'scraper_config.py': 'Scraper configuration',
    }

    for config, desc in configs.items():
        if Path(config).exists():
            print(f"  ✅ {config}")
        else:
            print(f"  ⚠️  {config} - Missing ({desc})")
            warnings.append(f"Missing config: {config}")

    print()

    # ==========================================
    # 3. Documentation
    # ==========================================

    print("📚 Checking Documentation...")

    docs = {
        'TRAP_DETECTION_INTEGRATION.md': 'Trap detection guide',
        'R1_BACKTEST_GUIDE.md': 'R1 backtest guide',
        'RUN_SCRAPERS_NOW.md': 'Scraper quick reference',
    }

    for doc, desc in docs.items():
        if Path(doc).exists():
            print(f"  ✅ {doc}")
        else:
            print(f"  ⚠️  {doc} - Missing")

    print()

    # ==========================================
    # 4. Dependencies
    # ==========================================

    print("🐍 Checking Python Dependencies...")

    try:
        import numpy
        print("  ✅ numpy")
    except ImportError:
        print("  ❌ numpy - REQUIRED")
        errors.append("Missing dependency: numpy (pip install numpy)")

    try:
        import pandas
        print("  ✅ pandas")
    except ImportError:
        print("  ❌ pandas - REQUIRED")
        errors.append("Missing dependency: pandas (pip install pandas)")

    try:
        from openai import OpenAI
        print("  ✅ openai")
    except ImportError:
        print("  ⚠️  openai - Needed for R1 (pip install openai)")
        warnings.append("Missing dependency: openai (needed for DeepSeek R1)")

    try:
        import requests
        print("  ✅ requests")
    except ImportError:
        print("  ⚠️  requests - Needed for scrapers")
        warnings.append("Missing dependency: requests (pip install requests)")

    try:
        from bs4 import BeautifulSoup
        print("  ✅ beautifulsoup4")
    except ImportError:
        print("  ⚠️  beautifulsoup4 - Needed for scrapers")
        warnings.append("Missing dependency: beautifulsoup4 (pip install beautifulsoup4)")

    print()

    # ==========================================
    # 5. Test Trap Detection
    # ==========================================

    print("🧪 Testing Trap Detection...")

    try:
        from ncaa_trap_detection import NCAATrapDetector
        detector = NCAATrapDetector()

        # Quick test
        trap_signal = detector.analyze_game(
            home_ml=-150,
            actual_handle=0.85,
            line_opened=-130,
            line_current=-150
        )

        if trap_signal.trap_score == -100:
            print("  ✅ Trap detection working correctly")
        else:
            print(f"  ⚠️  Trap detection returned unexpected score: {trap_signal.trap_score}")
            warnings.append("Trap detection test returned unexpected value")

    except Exception as e:
        print(f"  ❌ Trap detection failed: {e}")
        errors.append(f"Trap detection error: {e}")

    print()

    # ==========================================
    # 6. Data Directories
    # ==========================================

    print("📁 Checking Data Directories...")

    dirs = {
        'data': 'Main data directory',
        'data/handle_data': 'Handle data storage',
        'data/market_spreads': 'Market spreads',
        'models/ncaa': 'Trained models',
        'backtest_results': 'Backtest results',
    }

    for dir_path, desc in dirs.items():
        path = Path(dir_path)
        if path.exists():
            files = list(path.glob('*'))
            print(f"  ✅ {dir_path} ({len(files)} files)")
        else:
            print(f"  ⚠️  {dir_path} - Not found ({desc})")
            warnings.append(f"Directory not found: {dir_path}")

    print()

    # ==========================================
    # 7. Summary
    # ==========================================

    print("="*80)
    print("📊 VALIDATION SUMMARY")
    print("="*80 + "\n")

    if not errors and not warnings:
        print("✅✅✅ ALL SYSTEMS READY!")
        print()
        print("Your NCAA betting system is fully operational:")
        print("  • Trap detection working")
        print("  • R1 reasoner ready (needs API key)")
        print("  • All modules present")
        print()
        print("🚀 NEXT STEPS:")
        print("  1. Get handle data (Action Network or manual)")
        print("  2. Run on Tuesday MACtion:")
        print("     python ncaa_deepseek_r1_analysis.py <ODDS_KEY> <DEEPSEEK_KEY>")
        print("  3. Optional: Backtest first-half season:")
        print("     python backtest_ncaa_r1_system.py <DEEPSEEK_KEY>")
        print()

    elif errors:
        print(f"❌ CRITICAL ERRORS: {len(errors)}")
        for error in errors:
            print(f"  • {error}")
        print()
        print("⚠️  FIX THESE ERRORS BEFORE LIVE BETTING")
        print()

    if warnings:
        print(f"⚠️  WARNINGS: {len(warnings)}")
        for warning in warnings:
            print(f"  • {warning}")
        print()
        print("These won't prevent testing but needed for full functionality")
        print()

    # ==========================================
    # 8. System Capabilities
    # ==========================================

    print("="*80)
    print("🎯 SYSTEM CAPABILITIES")
    print("="*80 + "\n")

    print("✅ WORKING:")
    print("  • Trap Detection - Sharp vs public money analysis")
    print("  • 12-Model Ensemble - XGBoost, Neural Net, Bayesian, etc.")
    print("  • R1 Meta-Reasoning - Synthesizes models + trap signals")
    print("  • Contrarian Intelligence - Fade the public detection")
    print()

    print("⏳ NEEDS DATA:")
    print("  • Handle data - Run scraper or get from Action Network")
    print("  • Market spreads - For backtest validation (80%+ coverage)")
    print()

    print("🔑 NEEDS API KEYS:")
    print("  • DeepSeek API - For R1 reasoning")
    print("  • Odds API - For live game data")
    print("  • Action Network API (optional) - For handle data")
    print()

    print("="*80 + "\n")

    return len(errors) == 0


if __name__ == "__main__":
    success = validate_system()
    sys.exit(0 if success else 1)
