#!/usr/bin/env python3
"""
NFL Referee Analysis Backtesting Runner

This script runs comprehensive backtests of the Bayesian referee analysis system
to validate betting strategies based on referee patterns and anomalies.
"""

import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtesting.referee_backtest import RefereeBettingBacktest
from data_loaders.historical_data_loader import NFLDataLoader

def run_comprehensive_backtest():
    """Run comprehensive backtests across multiple seasons and strategies"""
    
    print("🏈 NFL Referee Analysis Backtesting System")
    print("=" * 50)
    
    # Initialize components
    loader = NFLDataLoader()
    backtest = RefereeBettingBacktest(initial_bankroll=10000)
    
    # Load historical data
    print("Loading historical data...")
    historical_data = loader.get_historical_data_for_backtest('2020-01-01', '2023-12-31')
    print(f"Loaded {len(historical_data)} games")
    
    if not historical_data:
        print("⚠️  No historical data found. Please populate the database first.")
        return
    
    # Run backtests for different strategies
    strategies = ['conservative', 'aggressive', 'high_confidence']
    results = []
    
    for strategy in strategies:
        print(f"\n📊 Running {strategy} strategy backtest...")
        result = backtest.run_backtest(historical_data, strategy)
        results.append(result)
        
        # Print quick summary
        print(f"   Total Bets: {result.total_bets}")
        print(f"   Win Rate: {result.win_rate:.1%}")
        print(f"   ROI: {result.roi:.1%}")
        print(f"   Profit: ${result.total_profit:,.2f}")
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    report = backtest.generate_report(results)
    print(report)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'referee_backtest_results_{timestamp}.json'
    
    detailed_results = []
    for result in results:
        detailed_results.append({
            'strategy': result.strategy_name,
            'period': f"{result.period_start} to {result.period_end}",
            'total_bets': result.total_bets,
            'winning_bets': result.winning_bets,
            'win_rate': round(result.win_rate, 4),
            'total_profit': round(result.total_profit, 2),
            'roi': round(result.roi, 4),
            'avg_bet_size': round(result.avg_bet_size, 2),
            'max_drawdown': round(result.max_drawdown, 4),
            'sharpe_ratio': round(result.sharpe_ratio, 2)
        })
    
    with open(results_file, 'w') as f:
        json.dump({
            'backtest_date': timestamp,
            'initial_bankroll': backtest.initial_bankroll,
            'data_period': f"{historical_data[0]['date']} to {historical_data[-1]['date']}" if historical_data else "No data",
            'total_games': len(historical_data),
            'strategies': detailed_results
        }, f, indent=2)
    
    print(f"💾 Detailed results saved to: {results_file}")
    
    # Performance summary
    print("\n📈 PERFORMANCE SUMMARY")
    print("-" * 30)
    
    best_roi = max(results, key=lambda x: x.roi)
    best_sharpe = max(results, key=lambda x: x.sharpe_ratio)
    most_bets = max(results, key=lambda x: x.total_bets)
    
    print(f"🏆 Best ROI: {best_roi.strategy_name} ({best_roi.roi:.1%})")
    print(f"📊 Best Sharpe: {best_sharpe.strategy_name} ({best_sharpe.sharpe_ratio:.2f})")
    print(f"🎯 Most Active: {most_bets.strategy_name} ({most_bets.total_bets} bets)")
    
    # Risk analysis
    print(f"\n⚠️  RISK ANALYSIS")
    print("-" * 20)
    for result in results:
        print(f"{result.strategy_name}: Max DD {result.max_drawdown:.1%}, Avg Bet ${result.avg_bet_size:,.0f}")

def run_seasonal_analysis():
    """Run season-by-season analysis to identify trends"""
    
    print("\n🗓️  SEASONAL ANALYSIS")
    print("=" * 30)
    
    loader = NFLDataLoader()
    backtest = RefereeBettingBacktest(initial_bankroll=10000)
    
    seasons = ['2020', '2021', '2022', '2023']
    strategy = 'conservative'  # Use conservative strategy for seasonal analysis
    
    seasonal_results = []
    
    for season in seasons:
        start_date = f"{season}-01-01"
        end_date = f"{season}-12-31"
        
        historical_data = loader.get_historical_data_for_backtest(start_date, end_date)
        
        if historical_data:
            result = backtest.run_backtest(historical_data, strategy)
            seasonal_results.append({
                'season': season,
                'games': len(historical_data),
                'bets': result.total_bets,
                'win_rate': result.win_rate,
                'roi': result.roi,
                'profit': result.total_profit
            })
            
            print(f"{season}: {result.total_bets} bets, {result.win_rate:.1%} WR, {result.roi:.1%} ROI")
    
    # Save seasonal analysis
    with open('seasonal_analysis.json', 'w') as f:
        json.dump(seasonal_results, f, indent=2)

def run_crew_specific_analysis():
    """Analyze performance by specific referee crews"""
    
    print("\n👨‍⚖️ CREW-SPECIFIC ANALYSIS")
    print("=" * 35)
    
    # This would analyze which crews provide the most profitable opportunities
    # Implementation would require grouping historical data by crew_id
    # and running mini-backtests for each crew
    
    print("Crew analysis would go here...")
    print("(Requires crew-specific data grouping)")

if __name__ == "__main__":
    try:
        # Run comprehensive backtest
        run_comprehensive_backtest()
        
        # Optional additional analyses
        if len(sys.argv) > 1 and '--detailed' in sys.argv:
            run_seasonal_analysis()
            run_crew_specific_analysis()
        
        print("\n✅ Backtesting complete!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Backtesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during backtesting: {str(e)}")
        sys.exit(1)