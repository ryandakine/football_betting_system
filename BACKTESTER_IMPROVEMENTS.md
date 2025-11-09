# NCAA Backtester - Identified Issues & Improvements

## 🐛 Critical Bugs

### 1. **Edge Calculation Bug** (Line 73)
```python
# CURRENT (WRONG):
edge = abs(win_prob - 0.5) - (market_prob - 0.5)
```

**Problem**: This formula doesn't make sense mathematically. It calculates edge as the absolute distance from 50% minus market efficiency, which is not how betting edge works.

**Fix**: Edge should be:
```python
# If win_prob > 0.5 (favoring home):
edge = win_prob - market_prob  # Your edge over the market
```

### 2. **Kelly Criterion Bug** (Line 183)
```python
# CURRENT (WRONG):
kelly_fraction = edge / 0.5  # Simplified Kelly
```

**Problem**: This is not Kelly Criterion. Kelly is: `(edge * probability) / odds`

**Fix**:
```python
# Proper Kelly for -110 odds (52.4% break-even):
win_prob = prediction['win_prob']
kelly_fraction = (win_prob * 1.909 - 1) / 0.909  # For -110 odds
kelly_fraction = max(0, kelly_fraction)  # Never bet negative edge
```

### 3. **Hardcoded -110 Odds** (Line 192)
```python
# CURRENT:
profit = stake * 0.909 if won else -stake  # Always assumes -110
```

**Problem**: Real games have varying odds. This oversimplifies the backtest.

**Fix**: Should use actual market odds from The Odds API or at minimum vary by predicted margin.

---

## ⚠️ Missing Features

### 4. **No Team Name Normalization**
SP+ might use "USC" while games data uses "Southern California"

**Fix**: Add team name mapping:
```python
TEAM_NAME_MAP = {
    'USC': 'Southern California',
    'Miami': 'Miami (FL)',
    'UCF': 'Central Florida',
    # ... etc
}
```

### 5. **No Conference Breakdown**
NCAA betting varies HUGELY by conference (SEC vs MAC)

**Fix**: Track and display results by conference:
```python
by_conference = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit': 0})
```

### 6. **No Bet Direction Tracking**
Should track: Home vs Away, Favorite vs Underdog

**Fix**:
```python
bet_type = 'home' if predicted_winner == 1 else 'away'
is_favorite = abs(predicted_margin) > 0
```

### 7. **No Statistical Significance Testing**
Current results could be random luck

**Fix**: Add t-test or bootstrap confidence intervals:
```python
from scipy import stats
t_stat, p_value = stats.ttest_1samp(profits, 0)
print(f"P-value: {p_value:.4f} (significant if < 0.05)")
```

### 8. **No Walk-Forward Validation**
Uses future data to predict past (look-ahead bias)

**Fix**: Only use data available at time of bet:
```python
# For 2023 games, only use 2022 and earlier SP+ ratings
# Simulate real-time betting conditions
```

### 9. **Missing Division by Zero Protection**
Line 293: `total_bets/games_with_data*100` could fail

**Fix**:
```python
pct = (total_bets/games_with_data*100) if games_with_data > 0 else 0
```

---

## 📊 Data Structure Assumptions

### 10. **Unclear Game Data Fields**
Code looks for multiple field names without knowing which exists:
- `home_points` vs `home_score`
- `winner` field might not exist
- `completed` field might not exist

**Fix**: First inspect actual data structure, then adapt code.

---

## 🎯 Performance Concerns

### 11. **Inefficient SP+ Lookup**
Creates ratings dict inside loop

**Fix**: Load all SP+ ratings once at start

### 12. **No Caching**
Re-reads JSON files if run multiple times

**Fix**: Add caching or use pickle for faster loading

---

## 💡 Enhancement Opportunities

### 13. **Add More Features**
- **Weather impact**: Rain/wind affects college more than NFL
- **Travel distance**: West coast teams traveling east
- **Rest days**: Teams on short rest
- **Rivalry games**: Historical matchups
- **Home field advantage by team**: Not all are 3 points

### 14. **Model Improvements**
- **Ensemble**: Combine SP+ with other metrics
- **Machine Learning**: Train on features to predict outcomes
- **Line shopping**: Account for best available odds
- **Closing line value**: Compare to market close

### 15. **Better Metrics**
- **Expected Value**: Calculate EV per bet
- **Luck-adjusted results**: Separate skill from variance
- **Confidence intervals**: Show range of expected outcomes
- **Monte Carlo simulation**: 10,000 trials to see distribution

---

## 🚀 Quick Wins (Immediate Fixes)

1. **Fix edge calculation** (5 min)
2. **Fix Kelly criterion** (5 min)
3. **Add conference breakdown** (10 min)
4. **Add team name normalization** (15 min)
5. **Add statistical significance test** (10 min)

Total: ~45 minutes to fix critical issues

---

## 📈 Priority Order

**Critical (Fix Now):**
1. Edge calculation bug
2. Kelly criterion bug
3. Team name normalization

**High Priority:**
4. Conference analysis
5. Bet direction tracking
6. Statistical significance

**Medium Priority:**
7. Walk-forward validation
8. Variable odds
9. Performance optimizations

**Nice to Have:**
10. Weather data
11. Advanced ML features
12. Monte Carlo simulation
