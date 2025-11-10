#!/usr/bin/env python3
"""
HIGH-ROI Optimized Backtest with Selective Betting
Only bets on the best opportunities with high edge + high confidence
"""

import numpy as np
import pandas as pd
from pathlib import Path
from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator
import pickle

print("="*80)
print("🏈 NCAA 9-MODEL BACKTEST - HIGH-ROI OPTIMIZATION")
print("="*80)
print()

# HIGH-ROI Configuration
STARTING_BANKROLL = 10000
MIN_EDGE = 0.05  # 5% edge (vs 3% before) - more selective
MIN_CONFIDENCE = 0.70  # 70% confidence (vs 60% before) - higher quality
FRACTIONAL_KELLY = 0.25
MAX_BETS_PER_WEEK = 20  # Limit to best opportunities

# Initialize
engineer = NCAAFeatureEngineer("data/football/historical/ncaaf")
orchestrator = SuperIntelligenceOrchestrator("models/ncaa")

# Load trained models
print("📂 Loading trained models...")
for model_name in ['spread_ensemble', 'total_ensemble', 'moneyline_ensemble',
                    'first_half_spread', 'home_team_total', 'away_team_total',
                    'alt_spread', 'xgboost_super', 'neural_net_deep']:
    model_path = Path(f"models/ncaa/{model_name}.pkl")
    if model_path.exists():
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            orchestrator.models[model_name].model = model_data['model']
            orchestrator.models[model_name].scaler = model_data['scaler']
            orchestrator.models[model_name].is_trained = model_data['is_trained']
            print(f"  ✅ {model_data['name']}")

# Test on multiple seasons for robustness
test_seasons = [2023, 2024]
print(f"\n📊 Backtesting on seasons: {test_seasons}")
print(f"💰 Starting bankroll: ${STARTING_BANKROLL:,.2f}")
print(f"📈 Min edge: {MIN_EDGE:.1%} (SELECTIVE)")
print(f"🎯 Min confidence: {MIN_CONFIDENCE:.1%} (HIGH-QUALITY)")
print(f"🔢 Max bets per week: {MAX_BETS_PER_WEEK}")
print()

all_bets = []
bankroll = STARTING_BANKROLL

for season in test_seasons:
    print(f"\n{'='*80}")
    print(f"🏈 Season {season}")
    print('='*80)

    games = engineer.load_season_data(season)
    completed_games = [g for g in games if g.get('completed')]
    print(f"Loaded {len(completed_games)} completed games")

    # Group by week for weekly bet limits
    games_by_week = {}
    for game in completed_games:
        week = game.get('week', 0)
        if week not in games_by_week:
            games_by_week[week] = []
        games_by_week[week].append(game)

    season_bets = []

    for week in sorted(games_by_week.keys()):
        week_games = games_by_week[week]
        week_opportunities = []

        # Evaluate all games in the week
        for game in week_games:
            home_score = game.get('homePoints')
            away_score = game.get('awayPoints')

            if home_score is None or away_score is None:
                continue

            actual_margin = home_score - away_score

            # Engineer features
            features_dict = engineer.engineer_features(game, season)
            features = pd.DataFrame([features_dict])

            # Get consensus prediction
            spread_predictions = []
            for model_name in ['spread_ensemble', 'xgboost_super', 'neural_net_deep', 'alt_spread']:
                if orchestrator.models[model_name].is_trained:
                    try:
                        pred = orchestrator.models[model_name].predict(features)
                        spread_predictions.append(pred[0] if hasattr(pred, '__iter__') else pred)
                    except:
                        continue

            if not spread_predictions:
                continue

            predicted_spread = np.mean(spread_predictions)
            spread_std = np.std(spread_predictions)

            # Confidence based on model agreement
            raw_confidence = max(0.5, min(0.95, 1 - (spread_std / 15.0)))

            # Apply calibration (models are overconfident)
            confidence = raw_confidence * 0.90  # Calibration multiplier

            # Calculate edge
            market_spread = -3.0  # Simplified
            edge = abs(predicted_spread - market_spread) / 14.0

            # Filter by thresholds
            if edge < MIN_EDGE or confidence < MIN_CONFIDENCE:
                continue

            # Determine bet
            if predicted_spread > market_spread + 1:
                bet_team = game.get('homeTeam')
                bet_spread = market_spread
                covers = actual_margin > market_spread
            elif predicted_spread < market_spread - 1:
                bet_team = game.get('awayTeam')
                bet_spread = -market_spread
                covers = actual_margin < market_spread
            else:
                continue

            # Store as opportunity
            week_opportunities.append({
                'game': game,
                'bet_team': bet_team,
                'predicted_spread': predicted_spread,
                'market_spread': market_spread,
                'actual_margin': actual_margin,
                'edge': edge,
                'confidence': confidence,
                'covers': covers,
                'week': week
            })

        # Select top N opportunities from this week
        week_opportunities.sort(key=lambda x: x['edge'] * x['confidence'], reverse=True)
        top_opportunities = week_opportunities[:MAX_BETS_PER_WEEK]

        # Place bets on top opportunities
        for opp in top_opportunities:
            # Kelly sizing
            win_prob = opp['confidence']
            b = 0.909
            kelly = (win_prob * b - (1 - win_prob)) / b
            kelly = max(0, min(0.10, kelly))

            stake = STARTING_BANKROLL * kelly * FRACTIONAL_KELLY
            stake = max(100, min(stake, 500))

            # Result
            if opp['covers']:
                profit = stake * 0.909
            else:
                profit = -stake

            bankroll += profit

            bet = {
                'season': season,
                'week': opp['week'],
                'game': f"{opp['game'].get('awayTeam')} @ {opp['game'].get('homeTeam')}",
                'bet_team': opp['bet_team'],
                'predicted_spread': opp['predicted_spread'],
                'market_spread': opp['market_spread'],
                'actual_margin': opp['actual_margin'],
                'edge': opp['edge'],
                'confidence': opp['confidence'],
                'stake': stake,
                'profit': profit,
                'covers': opp['covers'],
                'bankroll': bankroll
            }

            season_bets.append(bet)
            all_bets.append(bet)

    # Season summary
    if season_bets:
        wins = sum(1 for b in season_bets if b['covers'])
        total = len(season_bets)
        total_staked = sum(b['stake'] for b in season_bets)
        total_profit = sum(b['profit'] for b in season_bets)

        print(f"\n📈 Season {season} Results:")
        print(f"  Bets Placed: {total} (filtered from {len(completed_games)} games)")
        print(f"  Selectivity: {total/len(completed_games):.1%} of games")
        print(f"  Wins: {wins}")
        print(f"  Win Rate: {wins/total:.1%}")
        print(f"  Total Staked: ${total_staked:,.2f}")
        print(f"  Total Profit: ${total_profit:+,.2f}")
        print(f"  ROI: {total_profit/total_staked:.2%}")
        print(f"  Ending Bankroll: ${bankroll:,.2f}")

# Overall summary
print(f"\n{'='*80}")
print("📊 OVERALL HIGH-ROI RESULTS")
print('='*80)

if all_bets:
    total_bets = len(all_bets)
    total_wins = sum(1 for b in all_bets if b['covers'])
    total_staked = sum(b['stake'] for b in all_bets)
    total_profit = sum(b['profit'] for b in all_bets)

    profits = [b['profit'] for b in all_bets]
    sharpe = (np.mean(profits) / np.std(profits)) if np.std(profits) > 0 else 0

    print(f"\n💰 Financial Metrics:")
    print(f"  Starting Bankroll: ${STARTING_BANKROLL:,.2f}")
    print(f"  Ending Bankroll: ${bankroll:,.2f}")
    print(f"  Total Profit: ${total_profit:+,.2f}")
    print(f"  Return: {((bankroll - STARTING_BANKROLL) / STARTING_BANKROLL):.2%}")

    print(f"\n🎯 Betting Metrics:")
    print(f"  Total Bets: {total_bets}")
    print(f"  Wins: {total_wins}")
    print(f"  Losses: {total_bets - total_wins}")
    print(f"  Win Rate: {total_wins/total_bets:.1%}")
    print(f"  Total Staked: ${total_staked:,.2f}")
    print(f"  ROI: {total_profit/total_staked:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")

    print(f"\n📊 Quality Metrics:")
    print(f"  Avg Edge: {np.mean([b['edge'] for b in all_bets]):.2%}")
    print(f"  Avg Confidence: {np.mean([b['confidence'] for b in all_bets]):.1%}")
    print(f"  Avg Stake: ${np.mean([b['stake'] for b in all_bets]):,.2f}")

    # Edge analysis
    print(f"\n📈 Edge Performance:")
    edge_bins = [(0.05, 0.07), (0.07, 0.10), (0.10, 1.0)]
    for low, high in edge_bins:
        bin_bets = [b for b in all_bets if low <= b['edge'] < high]
        if bin_bets:
            bin_wins = sum(1 for b in bin_bets if b['covers'])
            print(f"  {low:.0%}-{high:.0%} edge: {bin_wins}/{len(bin_bets)} ({bin_wins/len(bin_bets):.1%})")

else:
    print("\n❌ No bets placed - thresholds too high")

print(f"\n{'='*80}")
print("✅ HIGH-ROI BACKTEST COMPLETE")
print('='*80)
