#!/usr/bin/env python3
"""Test AI providers"""

import sys
sys.path.append('.')

try:
    from football_master_gui import UnifiedAIProvider
    print("✅ UnifiedAIProvider import successful")

    # Test with empty API keys to see free providers
    api_keys = {}
    provider = UnifiedAIProvider(api_keys)

    status = provider.get_provider_status()

    print("\n🤖 AI Provider Status:")
    for name, info in status.items():
        status_icon = "✅" if info['status'] == 'active' else "❌"
        print(f"{status_icon} {info['name']} ({name}): {info['status']}")

    active_count = sum(1 for info in status.values() if info['status'] == 'active')
    print(f"\n🎯 Total Active Providers: {active_count}/8")

    # Test consensus analysis
    print("\n🧠 Testing consensus analysis...")
    game_data = {
        'home_team': 'Kansas City Chiefs',
        'away_team': 'Buffalo Bills',
        'sport': 'NFL'
    }

    consensus = provider.get_consensus_analysis(game_data)
    print(f"Consensus result: {len(consensus.get('individual_analyses', []))} analyses")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
