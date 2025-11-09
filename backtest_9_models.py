#!/usr/bin/env python3
"""
Backtest the 9 Trained NCAA Models
Uses actual trained models to make predictions and evaluate performance
"""

import numpy as np
import pandas as pd
from pathlib import Path
from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator

print("="*80)
print("🏈 NCAA 9-MODEL BACKTEST")
print("="*80)
print()

# Configuration
STARTING_BANKROLL = 10000
MIN_EDGE = 0.03  # 3%
MIN_CONFIDENCE = 0.60  # 60%
FRACTIONAL_KELLY = 0.25  # Use 25% Kelly for safety

# Initialize
engineer = NCAAFeatureEngineer("data/football/historical/ncaaf")
orchestrator = SuperIntelligenceOrchestrator("models/ncaa")

# Load all trained models
import pickle

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

# Backtest parameters
test_seasons = [2024]  # Test on most recent season
print(f"\n📊 Backtesting on seasons: {test_seasons}")
print(f"💰 Starting bankroll: ${STARTING_BANKROLL:,.2f}")
print(f"📈 Min edge: {MIN_EDGE:.1%}")
print(f"🎯 Min confidence: {MIN_CONFIDENCE:.1%}")
print()

# Track results
all_bets = []
bankroll = STARTING_BANKROLL

for season in test_seasons:
    print(f"\n{'='*80}")
    print(f"🏈 Season {season}")
    print('='*80)

    # Load games
    games = engineer.load_season_data(season)
    completed_games = [g for g in games if g.get('completed')]
    print(f"Loaded {len(completed_games)} completed games")

    season_bets = []

    for game in completed_games:
        # Get actual result
        home_score = game.get('homePoints')
        away_score = game.get('awayPoints')

        if home_score is None or away_score is None:
            continue

        actual_margin = home_score - away_score

        # Engineer features
        features_dict = engineer.engineer_features(game, season)
        features = pd.DataFrame([features_dict])

        # Get consensus prediction from spread models
        spread_predictions = []

        for model_name in ['spread_ensemble', 'xgboost_super', 'neural_net_deep', 'alt_spread']:
            if orchestrator.models[model_name].is_trained:
                try:
                    pred = orchestrator.models[model_name].predict(features)
                    spread_predictions.append(pred[0] if hasattr(pred, '__iter__') else pred)
                except Exception as e:
                    continue

        if not spread_predictions:
            continue

        # Consensus prediction
        predicted_spread = np.mean(spread_predictions)
        spread_std = np.std(spread_predictions)

        # Confidence based on model agreement (lower std = higher confidence)
        confidence = max(0.5, min(0.95, 1 - (spread_std / 15.0)))

        # Calculate edge (simplified - using implied spread)
        # Assume market spread is -3 if home favored, check if we have edge
        market_spread = -3.0  # Placeholder (in production, use actual odds)
        edge = abs(predicted_spread - market_spread) / 14.0  # Normalize

        # Filter by minimum thresholds
        if edge < MIN_EDGE or confidence < MIN_CONFIDENCE:
            continue

        # Determine bet direction
        if predicted_spread > market_spread + 1:
            # Bet on home team
            bet_team = game.get('homeTeam')
            bet_spread = market_spread
            covers = actual_margin > market_spread
        elif predicted_spread < market_spread - 1:
            # Bet on away team
            bet_team = game.get('awayTeam')
            bet_spread = -market_spread
            covers = actual_margin < market_spread
        else:
            continue

        # Kelly Criterion bet sizing
        win_prob = confidence
        b = 0.909  # For -110 odds
        kelly = (win_prob * b - (1 - win_prob)) / b
        kelly = max(0, min(0.10, kelly))  # Cap at 10% of bankroll

        # Fractional Kelly for safety - use STARTING bankroll to prevent exponential growth
        stake = STARTING_BANKROLL * kelly * FRACTIONAL_KELLY
        stake = max(100, min(stake, 500))  # Between $100 and $500 (flat bet sizing)

        # Result
        if covers:
            profit = stake * 0.909  # Win $90.91 per $100 at -110
        else:
            profit = -stake

        bankroll += profit

        bet = {
            'season': season,
            'week': game.get('week'),
            'game': f"{game.get('awayTeam')} @ {game.get('homeTeam')}",
            'bet_team': bet_team,
            'predicted_spread': predicted_spread,
            'market_spread': market_spread,
            'actual_margin': actual_margin,
            'edge': edge,
            'confidence': confidence,
            'stake': stake,
            'profit': profit,
            'covers': covers,
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
        print(f"  Bets Placed: {total}")
        print(f"  Wins: {wins}")
        print(f"  Win Rate: {wins/total:.1%}")
        print(f"  Total Staked: ${total_staked:,.2f}")
        print(f"  Total Profit: ${total_profit:+,.2f}")
        print(f"  ROI: {total_profit/total_staked:.2%}")
        print(f"  Ending Bankroll: ${bankroll:,.2f}")

# Overall summary
print(f"\n{'='*80}")
print("📊 OVERALL BACKTEST RESULTS")
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

    print(f"\n📉 Risk Metrics:")
    print(f"  Avg Stake: ${np.mean([b['stake'] for b in all_bets]):,.2f}")
    print(f"  Max Stake: ${max(b['stake'] for b in all_bets):,.2f}")
    print(f"  Avg Profit per Bet: ${np.mean(profits):+,.2f}")
    print(f"  Max Win: ${max(profits):+,.2f}")
    print(f"  Max Loss: ${min(profits):+,.2f}")

    # Confidence analysis
    print(f"\n🎲 Confidence Analysis:")
    conf_bins = [0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(conf_bins)-1):
        bin_bets = [b for b in all_bets if conf_bins[i] <= b['confidence'] < conf_bins[i+1]]
        if bin_bets:
            bin_wins = sum(1 for b in bin_bets if b['covers'])
            print(f"  {conf_bins[i]:.0%}-{conf_bins[i+1]:.0%}: {bin_wins}/{len(bin_bets)} ({bin_wins/len(bin_bets):.1%})")

else:
    print("\n❌ No bets placed with current thresholds")
    print("Try lowering MIN_EDGE or MIN_CONFIDENCE")

print(f"\n{'='*80}")
print("✅ BACKTEST COMPLETE")
print('='*80)
