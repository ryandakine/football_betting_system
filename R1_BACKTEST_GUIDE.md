# NCAA DeepSeek R1 System Backtest Guide

## What This Tests

**GOAL**: Validate R1 meta-analysis on historical first-half 2024 season

**What R1 Does**:
- Analyzes 12-model predictions collectively
- Finds patterns models are seeing
- Identifies edges Vegas might be missing
- Provides confidence-based bet sizing

**What We Test**:
1. Does R1 find edges in the 12-model ensemble?
2. Does R1's confidence correlate with win rate?
3. Does R1's bet sizing produce positive ROI?

---

## Prerequisites

### 1. Market Spread Data (CRITICAL!)

R1 backtest requires **80%+ market spread coverage** to validate edge.

**Run scrapers first**:
```bash
# On your local machine (not in sandbox)
python scrape_teamrankings_historical.py 2024
python scrape_covers_historical.py 2024
```

**Why**: Can't test if R1 beats market without knowing what market spreads were!

### 2. Trained Models

System uses all 12 trained models:
- ✅ XGBoost Super
- ✅ Neural Net Deep
- ✅ Alt Spread
- ✅ Bayesian Ensemble
- ✅ Momentum Model
- ✅ Situational
- ✅ Advanced Stats
- ✅ Drive Outcomes
- ✅ Opponent-Adjusted
- ✅ Special Teams
- ✅ Pace & Tempo
- ✅ Game Script

**Check status**:
```bash
ls -lh models/ncaa/*.pkl
```

### 3. DeepSeek API Key

R1 reasoning requires DeepSeek API access.

**Get key**: https://platform.deepseek.com/

---

## Running the Backtest

### Basic Usage

```bash
python backtest_ncaa_r1_system.py <DEEPSEEK_API_KEY>
```

**Example**:
```bash
python backtest_ncaa_r1_system.py sk-abc123xyz789
```

### What Happens

**Step 1: Validation**
```
✅ R1 Backtester initialized
   Models loaded: 12/12
   Market spreads: 487 games

🧠 BACKTESTING R1 SYSTEM - FIRST HALF 2024 SEASON

✅ Market data coverage: 85.3% (412/483 games)
```

**Step 2: Processing**
```
📊 Loading 2024 season data...
   Total games: 983
   First half (weeks 1-8): 483

🔬 Generating predictions and R1 analysis...
   Processed 10/483 games...
   Processed 20/483 games...
   ...
```

**Step 3: Results**
```
📊 R1 BACKTEST RESULTS - FIRST HALF 2024 SEASON

🎯 CONFIDENCE TIER: 80%+
Total bets: 23
Wins: 15
Losses: 8
Win rate: 65.2%
Total profit: $142.50
ROI: +31.0%

🎯 CONFIDENCE TIER: 75-79%
Total bets: 47
Wins: 28
Losses: 19
Win rate: 59.6%
Total profit: $98.20
ROI: +10.4%

🎯 CONFIDENCE TIER: 70-74%
Total bets: 85
Wins: 47
Losses: 38
Win rate: 55.3%
Total profit: $23.50
ROI: +1.4%

📈 OVERALL RESULTS
Total bets: 155
Wins: 90
Losses: 65
Win rate: 58.1%
Total profit: $264.20

🎯 EXPECTED vs ACTUAL:
   Expected win rate: 58-62% (NFL validation)
   Actual win rate: 58.1%
   ✅ R1 VALIDATED - System working as expected!

💾 Results saved to backtest_results/r1_backtest_2024_first_half.json
```

---

## What Gets Tested

### 1. R1 Meta-Analysis Quality

**Does R1 find real edges?**
- R1 analyzes all 12 model predictions
- Identifies consensus (strong signal)
- Finds disagreements (edge or uncertainty?)
- Determines if models collectively see edge vs market

**Expected**: R1 should find edges that simple ensemble misses

### 2. Confidence Calibration

**Does R1's confidence match win rate?**

| R1 Confidence | Expected Win Rate | Bet Size |
|---------------|-------------------|----------|
| 80%+ | 62-68% | 6 units ($30) |
| 75-79% | 58-62% | 4 units ($20) |
| 70-74% | 54-58% | 2 units ($10) |
| <70% | Skip | 0 units |

**Expected**: Higher confidence = higher win rate (calibration)

### 3. Bet Sizing Strategy

**Does fractional Kelly produce positive ROI?**

Based on NFL validation:
- Starting: $100
- Ending: $6,091 (over 10 years)
- Return: 60.91x

**Expected for half season**:
- Win rate: 58-62%
- ROI: 30-50%
- Higher confidence bets drive most profit

---

## Limitations

### What This Backtest CAN'T Test

**1. Real-Time Context**
- ❌ Injuries at that moment
- ❌ Weather conditions
- ❌ Momentum narratives ("Toledo hot last 3 games")
- ❌ Line movement timing (opening vs close)

**2. Contrarian Signals**
- ❌ Historical public betting percentages (not available)
- ❌ Sharp money detection (need tick-by-tick line data)
- ❌ Reverse line movement signals

**3. R1's Full Reasoning**
- ❌ R1 uses simplified prompt (no real-time context)
- ❌ Can't reference current injuries/weather
- ❌ Focuses only on 12-model pattern analysis

### What This Backtest CAN Test

**1. Core R1 Ability**
- ✅ Can R1 find patterns in 12-model ensemble?
- ✅ Does R1 identify consensus vs disagreements?
- ✅ Does R1's meta-analysis beat simple ensemble average?

**2. System Validation**
- ✅ Does R1 confidence correlate with win rate?
- ✅ Does bet sizing produce positive ROI?
- ✅ Does system beat market spreads?

**3. Edge Detection**
- ✅ Can R1 identify games where models see edge?
- ✅ Does R1 correctly skip low-confidence games?
- ✅ Do high-confidence bets outperform low-confidence?

---

## Interpreting Results

### ✅ Good Results

**Win rate 58%+ overall**:
- Matches NFL validation (60.91x returns)
- System working as designed

**Confidence calibration**:
- 80%+ tier: 62%+ win rate
- 75-79% tier: 58-62% win rate
- 70-74% tier: 54-58% win rate
- Higher confidence = higher win rate ✅

**Positive ROI**:
- Overall ROI: +30% or higher
- High confidence tier drives most profit
- System beats -110 juice

### ⚠️ Warning Signs

**Win rate <55%**:
- Below expected performance
- May need more data or model refinement
- Check market spread coverage (need 80%+)

**No confidence calibration**:
- All tiers same win rate
- R1 not differentiating quality
- May need prompt refinement

**Negative ROI**:
- System not beating market
- Check if models trained properly
- Verify market spread data quality

---

## Troubleshooting

### Error: "INSUFFICIENT MARKET DATA"

**Problem**: Not enough market spreads to validate backtest

**Solution**:
```bash
# Run scrapers to get market spread data
python scrape_teamrankings_historical.py 2024
python scrape_covers_historical.py 2024

# Verify coverage
python -c "from backtest_ncaa_r1_system import NCAAR1Backtester; b = NCAAR1Backtester('test'); print(b.validate_market_data(2024))"
```

**Requirement**: 80%+ coverage (at least 386/483 games)

### Error: "Models not loaded"

**Problem**: Trained models not found

**Solution**:
```bash
# Check models directory
ls -lh models/ncaa/*.pkl

# If missing, train models first
python ncaa_train_all_models.py
```

### Error: "DeepSeek API error"

**Problem**: Invalid or expired API key

**Solution**:
1. Get new key: https://platform.deepseek.com/
2. Check key format: `sk-...`
3. Verify API credits available

---

## Expected Timeline

**With market spread data available**:
- Processing: ~2-5 minutes (483 games)
- R1 API calls: ~483 requests (~$2-5 in API costs)
- Total time: ~10 minutes

**Without market spread data**:
- Run scrapers first: ~30-60 minutes
- Then run backtest: ~10 minutes
- Total: ~1 hour

---

## Next Steps After Backtest

### If Results Good (58%+ win rate)

**System validated! Ready for live betting.**

1. **Wait for Tuesday MACtion**
   ```bash
   python ncaa_deepseek_r1_analysis.py <ODDS_KEY> <DEEPSEEK_KEY>
   ```

2. **Follow R1 recommendations**
   - Trust R1's confidence tiers
   - Use recommended bet sizing
   - Log all bets for tracking

3. **Track live performance**
   - Compare to backtest results
   - Adjust if real-world differs significantly

### If Results Weak (<55% win rate)

**System needs refinement.**

1. **Check data quality**
   - Verify market spread accuracy
   - Check for data errors

2. **Review model predictions**
   - Are models trained properly?
   - Check prediction distribution

3. **Refine R1 prompt**
   - Add more NCAA-specific context
   - Improve pattern detection instructions

---

## Bottom Line

**This backtest validates R1's core ability**:
- Find patterns in 12-model ensemble
- Identify edges Vegas missing
- Calibrate confidence to win rate

**It does NOT test**:
- Real-time context integration
- Contrarian signal usage
- Live line movement timing

But if R1 can beat market on JUST model pattern analysis, that's the foundation.

**Expected Result**: 58-62% win rate, +30-50% ROI over half season

If backtest shows this, system is ready for live betting! 🚀
