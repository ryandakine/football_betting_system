#!/usr/bin/env python3
"""
Test GGUF ensemble on tonight's Monday Night Football game
"""

from practical_gguf_ensemble import PracticalGGUFEnsemble
import time

def main():
    print('🏈 MONDAY NIGHT FOOTBALL ANALYSIS')
    print('='*50)
    print('Ravens @ Buccaneers - Week 7, 2024')
    print('Spread: TB -3.0 | Total: 51.5 | Ref: Ron Torbert')
    print('='*50)

    # Initialize ensemble
    try:
        ensemble = PracticalGGUFEnsemble()
        print(f'📦 Ensemble loaded: {len(ensemble.model_configs)} models available')
    except Exception as e:
        print(f'❌ Failed to load ensemble: {e}')
        return

    # Tonight's MNF game
    mnf_game = {
        'home_team': 'Tampa Bay Buccaneers',
        'away_team': 'Baltimore Ravens',
        'season': '2024',
        'week': '7',
        'spread': -3.0,
        'total': 51.5,
        'referee': 'Ron Torbert',
        'commence_time': '2024-10-21T20:15:00Z'
    }

    print('🤖 Getting AI analysis from GGUF ensemble...')
    start_time = time.time()

    try:
        # Get prediction from best available model
        result = ensemble.get_prediction(mnf_game)
        
        end_time = time.time()
        
        if result:
            print(f'\n⏱️  Analysis completed in {end_time - start_time:.1f} seconds')
            print(f'\n🎯 PREDICTION RESULTS:')
            print(f'   Probability (TB wins): {result["probability"]:.3f}')
            print(f'   Confidence: {result["confidence"]:.3f}') 
            print(f'   Risk Level: {result["risk_level"]}')
            print(f'   Model Used: {result.get("model_specialty", "Unknown")}')
            
            if result.get('key_factors'):
                print(f'\n💡 KEY FACTORS:')
                for i, factor in enumerate(result['key_factors'][:5], 1):
                    print(f'   {i}. {factor}')
            
            if result.get('analysis'):
                print(f'\n📝 ANALYSIS:')
                print(f'   {result["analysis"]}')
                
            if result.get('recommendation'):
                print(f'\n🎲 BETTING RECOMMENDATION:')
                print(f'   {result["recommendation"]}')
                
            print('\n✅ GGUF ensemble working perfectly for live games!')
            
            # Show which model was actually used
            print(f'\n🔧 Technical Details:')
            print(f'   Response time: {result.get("response_time", 0):.2f}s')
            print(f'   Model specialty: {result.get("model_specialty", "N/A")}')
            
        else:
            print('❌ No prediction returned')
            
    except KeyboardInterrupt:
        print('\n⚠️ Analysis interrupted by user')
    except Exception as e:
        print(f'❌ Error during analysis: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()