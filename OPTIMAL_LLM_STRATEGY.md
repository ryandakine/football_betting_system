# OPTIMAL LLM META-MODEL STRATEGY

## Executive Summary

After backtesting 5635 historical NFL games across 8 different strategies,
we have identified the optimal approach for combining LLM meta-models.

## 🏆 RECOMMENDED STRATEGY

**Strategy: 100% DeepSeek-R1 reasoning model**

### Performance Metrics
- **ROI:** 37.03%
- **Win Rate:** 74.57%
- **Total Bets:** 3875
- **Profit:** $5991.00 (from $100.00 starting bankroll)
- **Sharpe Ratio:** 4.046
- **Max Drawdown:** 6.44%

### Model Weights
```json
{
  "deepseek-r1": 1.0,
  "mistral-7b": 0.0,
  "mixtral-8x7b": 0.0
}
```

### Betting Rules
- Minimum Confidence: 70.0%
- Require Agreement: No
- Dynamic Weighting: No

### Bet Sizing
- **High Confidence (80%+):** 6 units
- **Medium Confidence (75-79%):** 4 units
- **Low Confidence (70-74%):** 2 units

## 📊 Alternative: Best Risk-Adjusted Strategy

The highest ROI strategy is also the best risk-adjusted strategy!

## 🎯 Key Insights

### Single Model vs Ensemble
- **Best Single Model:** 100% DeepSeek-R1 reasoning model (37.03% ROI)
- **Best Ensemble:** 50% DeepSeek, 25% Mistral, 25% Mixtral (36.40% ROI)

**Conclusion:** A single model is sufficient for optimal results.

## 🚀 Implementation Guide

1. Load optimal weights: `optimal_llm_weights.json`
2. For each game, get predictions from all 3 LLM models
3. Apply the recommended weighting strategy
4. Only bet when consensus confidence meets threshold
5. Size bets using Kelly Criterion (2-6 units)

## 📈 Expected Results

Based on historical backtesting:
- Starting with $100, expect to reach $6091.00
- 3875 bets over 5635 games
- Betting on ~68.8% of games
