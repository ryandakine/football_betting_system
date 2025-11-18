# 📊 Production Data System

**Complete system for capturing, storing, and using real betting odds**

## Overview

This system solves the critical problem: **You can't backtest without historical betting lines.**

**What it does:**
1. ✅ Captures daily NFL/NCAA odds from The Odds API
2. ✅ Stores historical odds in `data/historical_odds.json`
3. ✅ Provides manual entry for historical odds
4. ✅ Scrapes historical odds from Pro Football Reference (template)
5. ✅ Validates data quality (no synthetic data)
6. ✅ Integrates with your v2 betting system

---

## Quick Start

### 1. Set Up Daily Odds Capture (5 minutes)

```bash
# Install (one-time setup)
bash setup_daily_odds_capture.sh

# This will:
# - Check dependencies
# - Verify API key
# - Create cronjob (runs daily at noon)
# - Test the system
```

**That's it!** Odds will now be captured automatically every day.

---

### 2. Add Historical Odds (Choose One Method)

#### Method A: Manual Entry (Quick Testing)

```bash
# Interactive manual entry
python3 scrape_historical_odds.py --manual

# Example:
Game key: 2024-09-05_KC_at_BAL
Spread (home): 2.5
Total: 46.5
Home ML: +110
Away ML: -130
```

#### Method B: CSV Bulk Import (Recommended)

Create `historical_odds.csv`:
```csv
date,away_team,home_team,spread,total,home_ml,away_ml,source
2024-09-05,KC,BAL,2.5,46.5,+110,-130,closing_line
2024-09-08,PHI,GB,-2.5,49.5,-140,+120,closing_line
2024-09-08,LAR,DET,3.5,52.5,+155,-185,closing_line
```

Import:
```bash
python3 scrape_historical_odds.py --csv historical_odds.csv
```

#### Method C: Web Scraping (Advanced)

```bash
# Scrape Pro Football Reference (template provided)
python3 scrape_historical_odds.py --year 2024 --week 1

# Note: May require HTML parsing adjustments
```

---

## Daily Workflow (During NFL Season)

### Morning (Before Games)
```bash
# Automatic (cronjob runs at noon)
# Manual: python3 capture_daily_odds.py
```

### Evening (Run Predictions)
```bash
# NFL
python3 run_nfl_12model_deepseek_v2.py --week 11

# NCAA
python3 run_ncaa_12model_deepseek_v2.py --week 12
```

### Compare to Market
```bash
# Find value bets (where your model disagrees with market)
python3 find_value_bets.py

# Example output:
# Chiefs -7.5 (your model: 75% confidence, market implies: 60%)
# → VALUE BET (15% edge)
```

### After Games
```bash
# Update results
python3 update_game_results.py

# Rebuild dataset
python3 production_data_builder.py  # If you create this

# Validate
python3 validate_production_data.py  # If you create this
```

---

## Files Created

| File | Purpose | Status |
|------|---------|--------|
| `capture_daily_odds.py` | Daily odds capture | ✅ Ready |
| `scrape_historical_odds.py` | Historical odds tools | ✅ Ready |
| `setup_daily_odds_capture.sh` | Cronjob setup | ✅ Ready |
| `data/historical_odds.json` | Odds database | 📝 Needs data |

---

## Data Format

### historical_odds.json Structure

```json
{
  "2024-09-05_KC_at_BAL": {
    "spread": 2.5,
    "total": 46.5,
    "home_ml": +110,
    "away_ml": -130,
    "bookmaker": "DraftKings",
    "captured_at": "2024-09-05T11:30:00",
    "source": "odds_api"
  },
  "2024-09-08_PHI_at_GB": {
    "spread": -2.5,
    "total": 49.5,
    "home_ml": -140,
    "away_ml": +120,
    "source": "manual"
  }
}
```

### Game Key Format

```
{date}_{away_team}_at_{home_team}

Examples:
- 2024-09-05_KC_at_BAL
- 2024-11-14_BUF_at_KC
- 2025-01-12_SF_at_PHI
```

---

## Integrating with Your Betting System

### 1. Use Historical Odds in Predictions

```python
from pathlib import Path
import json

# Load historical odds
with open('data/historical_odds.json') as f:
    odds_db = json.load(f)

# Get odds for a game
game_key = "2024-11-17_BUF_at_KC"
if game_key in odds_db:
    market_spread = odds_db[game_key]['spread']
    market_total = odds_db[game_key]['total']

    # Compare to your model's prediction
    your_spread = -9.5
    your_total = 48.0

    # Find edge
    spread_edge = abs(your_spread - market_spread)
    total_edge = abs(your_total - market_total)

    print(f"Spread edge: {spread_edge} points")
    print(f"Total edge: {total_edge} points")
```

### 2. Backtest with Real Odds

```python
# Load predictions
predictions = load_predictions('data/predictions/nfl_prediction_log.json')

# Match with historical odds
for pred in predictions:
    game_key = create_game_key(pred)

    if game_key in odds_db:
        # Use real odds for backtest
        market_odds = odds_db[game_key]

        # Calculate if you would have bet
        if pred['confidence'] > 0.65 and has_edge(pred, market_odds):
            # Simulate bet
            result = calculate_bet_outcome(pred, market_odds)
            track_results(result)
```

---

## Troubleshooting

### "No API key found"
```bash
# Set environment variable
export ODDS_API_KEY='your_key_here'

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export ODDS_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### "HTTP 403 Forbidden"
- API key invalid or expired
- Get new key from: https://the-odds-api.com/
- Free tier: 500 requests/month

### "No odds captured"
- Check if games are scheduled
- NFL season: September - February
- NCAA season: September - December

### "Cronjob not running"
```bash
# Check cronjob exists
crontab -l

# Check logs
tail -f logs/odds_capture.log

# Run manually to test
python3 capture_daily_odds.py
```

---

## Cost Analysis

### The Odds API (Free Tier)
- **500 requests/month free**
- NFL season: ~17 weeks × 14 games = 238 games
- NCAA season: ~15 weeks × 50 games = 750 games
- Daily capture: 1 request/day × 30 days = 30 requests/month

**Verdict**: Free tier is sufficient for daily capture!

### Paid Tier ($50/month)
- 10,000 requests/month
- Live odds updates every 5 minutes
- Historical odds API access
- Multiple sports

**Recommendation**: Start with free tier, upgrade only if needed.

---

## Data Sources

### Primary: The Odds API
- **Pros**: Real-time, multiple bookmakers, JSON format
- **Cons**: Limited free tier, no deep historical
- **Use for**: Daily capture, live odds

### Secondary: Pro Football Reference
- **Pros**: Free, comprehensive historical data
- **Cons**: Requires scraping, HTML parsing
- **Use for**: Historical odds, backtesting

### Tertiary: Manual Entry
- **Pros**: 100% accurate, no scraping needed
- **Cons**: Time-consuming
- **Use for**: Quick testing, critical games

---

## Best Practices

### 1. Capture Timing
- **Best**: 30-60 minutes before kickoff (closing lines)
- **Good**: Morning of game day
- **Acceptable**: Any time before game starts

### 2. Data Validation
- Always use closing lines for backtests (most efficient prices)
- Cross-reference odds from multiple bookmakers
- Document source for each game

### 3. Storage
- Keep raw API responses (for debugging)
- Version control your odds database
- Back up database weekly

### 4. Usage
- Never backtest on synthetic/estimated odds
- Account for vig (-110 standard = 52.38% breakeven)
- Test multiple Kelly fractions (0.25, 0.5, 1.0)

---

## Future Enhancements

### Phase 2: Live Odds
- Track line movements throughout the day
- Alert when lines move significantly
- Find arbitrage opportunities

### Phase 3: Multi-Bookmaker
- Compare odds across 5+ bookmakers
- Line shopping automation
- Best price finder

### Phase 4: Historical Archive
- Complete NFL odds database (2015-2025)
- NCAA odds archive
- Playoff/bowl game odds

---

## Example Use Cases

### Use Case 1: Find Value Bets

```python
# Your model says 70% confidence
model_confidence = 0.70

# Market odds: -110 (implies 52.38%)
market_implied = 0.5238

# Edge calculation
edge = model_confidence - market_implied
# → 0.70 - 0.5238 = 0.1762 (17.62% edge!)

if edge > 0.10:
    print("VALUE BET!")
```

### Use Case 2: Validate Your Model

```python
# Load your predictions vs actual odds
comparisons = []

for game in season_games:
    your_prediction = get_prediction(game)
    market_line = get_historical_odds(game)

    # How often did you beat the market?
    if abs(your_prediction - actual_result) < abs(market_line - actual_result):
        comparisons.append(1)  # You were closer
    else:
        comparisons.append(0)  # Market was closer

beat_market_rate = sum(comparisons) / len(comparisons)
print(f"Beat market {beat_market_rate*100}% of the time")
```

### Use Case 3: Track Line Movements

```python
# Capture odds multiple times per day
morning_odds = capture_odds(time='9am')
noon_odds = capture_odds(time='12pm')
closing_odds = capture_odds(time='pre-kickoff')

# Analyze sharp money
if closing_odds['spread'] != morning_odds['spread']:
    movement = closing_odds['spread'] - morning_odds['spread']
    print(f"Line moved {movement} points")

    if abs(movement) > 2:
        print("SHARP MONEY DETECTED!")
```

---

## Support

**Issues?**
1. Check API quota: https://the-odds-api.com/account/
2. Verify cron logs: `tail -f logs/odds_capture.log`
3. Test manually: `python3 capture_daily_odds.py`

**Questions?**
- The Odds API docs: https://the-odds-api.com/liveapi/guides/v4/
- Cron syntax: https://crontab.guru/

---

## Status

✅ **Daily capture system**: Ready
✅ **Manual entry tools**: Ready
✅ **Cronjob setup**: Ready
📝 **Historical database**: Needs population (start capturing now!)

**Start capturing odds today to build your historical database!** 🏈
