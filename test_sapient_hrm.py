#!/usr/bin/env python3
"""
Test script for Sapient HRM adapter integration
"""

try:
    from hrm_sapient_adapter import SapientHRMAdapter
    print('✅ Sapient HRM adapter import successful!')

    # Test instantiation
    adapter = SapientHRMAdapter()
    print('✅ Sapient HRM adapter instantiated!')

    # Test model info
    info = adapter.get_model_info()
    print('📊 Model:', info['model_name'])
    print('🏗️  Architecture:', info['architecture'])
    print('📏 Parameters:', info['parameters'])
    print('⚡ Status:', info['status'])
    print('🎯 Capabilities:', len(info['capabilities']), 'features')

    # Test mock analysis
    test_game = {
        'home_team': 'Kansas City Chiefs',
        'away_team': 'Buffalo Bills',
        'home_ml_odds': 1.80,
        'away_ml_odds': 2.10,
        'edge_detected': 2.5,
        'weather': {'temperature_f': 45},
        'injuries': {'home': [], 'away': []}
    }

    result = adapter.analyze_game(test_game)
    print('')
    print('🏈 Test Analysis:')
    print('Prediction:', result['prediction'])
    print('Confidence:', '.1%')
    print('Expected Value:', '.1%')
    print('Provider:', result['provider'])

    print('')
    print('🎉 Sapient HRM adapter ready!')

except Exception as e:
    print('❌ Error:', str(e))
    import traceback
    traceback.print_exc()
