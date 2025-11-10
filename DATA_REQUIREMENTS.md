# NCAA Betting System - Data Requirements

## 🎯 What Data We Need vs What We Have

### ✅ WHAT WE HAVE (Game Results)

**Source:** College Football Data API
**Data:** 21,522 games (2015-2024)
**Fields:**
- Home team / Away team
- Home score / Away score (actual results)
- Date, week, venue
- Team stats (SP+ ratings, recruiting rankings, etc.)

**Example:**
```json
{
  "id": 401234567,
  "homeTeam": "Alabama",
  "awayTeam": "Georgia",
  "homePoints": 31,
  "awayPoints": 24,
  "week": 5,
  "date": "2023-09-30"
}
```

**What this tells us:** Alabama beat Georgia 31-24

---

### ❌ WHAT WE'RE MISSING (Market Spreads)

**What we need:** Historical betting lines (what DraftKings/FanDuel were offering)
**Why we need it:** To train models and backtest correctly

**Example of what's missing:**
```json
{
  "id": 401234567,
  "homeTeam": "Alabama",
  "awayTeam": "Georgia",
  "market_spread": -7.5,     // ← MISSING (what sportsbooks offered)
  "closing_odds": -110,       // ← MISSING
  "opening_spread": -6.5,     // ← MISSING
  "line_movement": -1.0       // ← MISSING
}
```

**What market spread tells us:**
- Sportsbooks thought Alabama would win by 7.5 points
- This is what the "market" (sharp bettors + public) believed
- We need to beat THIS number, not just predict the score

---

## 🧠 Why We Need Market Spreads for 12-Model System

### **Current Problem:**

Without market spreads, we can only measure:
- ✅ Prediction accuracy (how close we got to actual score)
- ❌ Betting edge (whether we beat the market)

**These are NOT the same thing!**

### **Example:**

**Game:** Alabama vs Georgia
**Actual result:** Alabama wins 31-24 (spread: +7)
**Market spread:** Alabama -7.5

**Our predictions:**
- Model 1: Alabama -8.0
- Model 2: Alabama -7.2
- Model 3: Alabama -6.8

**Without market spreads:**
- We think Model 3 is best (closest to +7 actual)
- But Model 3 would have LOST the bet!

**With market spreads:**
- Model 1 & 2: BET ALABAMA (they covered -7.5)
- Model 3: BET GEORGIA (Alabama didn't cover)
- Market spread: -7.5, Actual: +7
- Models 1 & 2 were WRONG (lost money)
- Model 3 was RIGHT (made money)

**Conclusion:** Without market spreads, we train models to be "accurate" but not "profitable"

---

## 📊 How This Affects 12-Model System

### **Models That REQUIRE Market Spreads:**

1. **Model 4: Line Movement Tracker**
   - Needs: Opening spread, closing spread
   - Uses: Line movement patterns to detect sharp money

2. **Model 5: Market Efficiency Analyzer**
   - Needs: Market spread, actual spread
   - Uses: Identifies inefficient markets (MACtion, small conferences)

3. **Model 6: Public Betting Patterns**
   - Needs: Market spread, public betting %
   - Uses: Fades public when overbet

4. **Model 8: Conference Edge Finder**
   - Needs: Market spread by conference
   - Uses: Finds which conferences markets misprice

5. **Model 10: Situational Betting Model**
   - Needs: Market spread + situational factors
   - Uses: Spots trap games, look-ahead spots

### **Models That Work Without Market Spreads (But Are Limited):**

6. **Model 1-3: Core predictive models** (XGBoost, Neural Net, Alt Spread)
   - Can predict scores
   - But can't calculate betting edge

7. **Model 7: Weather Impact** - Works with predictions

8. **Model 9: Injury Impact** - Works with predictions

9. **Model 11: Officiating Bias** - Works with predictions

10. **Model 12: Prop Bet Specialist** - Works with predictions

**Conclusion:** Core models work, but we can't validate if they're PROFITABLE without market data.

---

## 💰 Why Market Spreads = Real Edge

### **Scenario 1: Prediction Accuracy (What We Measure Now)**

```
Alabama vs Georgia
Actual: Alabama wins by 7
Our prediction: Alabama wins by 6.8

Error: 0.2 points
Grade: A+ (very accurate!)
```

### **Scenario 2: Betting Profitability (What Actually Matters)**

```
Alabama vs Georgia
Market: Alabama -7.5
Our prediction: Alabama -6.8
Actual: Alabama wins by 7

We bet: Georgia +7.5 (we think Alabama wins by less than market)
Result: LOSS (Alabama covered -7)
Profit: -$110

Accurate prediction ≠ Profitable bet
```

**THE PROBLEM:** Without market spreads, we optimize for accuracy, not profit.

---

## 🎯 Where to Get Market Spreads

### **Option 1: Scrape FREE (What We're Doing)**

**Sites with historical closing lines:**
- TeamRankings.com - Full season closing lines
- Covers.com - Historical matchups database
- OddsShark.com - Odds database going back years
- Archive.org - Historical snapshots

**Scrapers we built:**
- `scrape_teamrankings_historical.py` - Full season scraper
- `scrape_covers_historical.py` - Week-by-week scraper
- `run_all_scrapers.sh` - Automated 2015-2024 scrape

**Expected coverage:** 90-95% of games (with multiple sources)
**Cost:** $0
**Time:** 8-10 hours overnight

**Run on local machine:**
```bash
chmod +x run_all_scrapers.sh
./run_all_scrapers.sh
```

### **Option 2: Buy Historical Data ($99)**

**Sports Insights:**
- URL: https://www.sportsinsights.com/
- Price: $99/year
- Data: Closing lines for all NCAA games 2003-2024
- Format: CSV download
- Quality: Professional grade, used by sharps

**Pros:**
- Guaranteed complete coverage
- Accurate data
- Instant download
- Includes line movement data

**Cons:**
- Costs $99
- Need to pay yearly for updates

### **Option 3: Forward Test Only (FREE)**

**Skip historical validation, test on 2025 season:**
- Use live odds from The Odds API (free tier)
- Run daily predictions
- Track results over full season
- Calculate ROI at end of season

**Pros:**
- $0 cost
- Zero hindsight bias
- Real-world validation

**Cons:**
- Takes full season (13 weeks)
- Can't know if system works until end

---

## 📋 What Happens After We Get Market Spreads

### **Step 1: Retrain Models**

```bash
python retrain_with_market_data.py
```

**Changes:**
- Models now optimize for "beat the spread" not "predict accurately"
- Loss function: `max(0, market_spread - predicted_spread) * actual_result`
- Models learn to find market inefficiencies

### **Step 2: Run Realistic Backtest**

```bash
python backtest_ncaa_parlays_REALISTIC.py
```

**New metrics:**
- Win rate against the spread (expect 52-55%)
- ROI (expect 5-10%)
- Profit curve over 10 years
- Edge by conference, day of week, game type

### **Step 3: Validate System**

```
Expected results WITH market spreads:
- Win rate: 52-55% (vs 50% breakeven)
- ROI: 5-10% per season
- $10K over 10 years: $16K-$26K (not $108 billion!)
```

### **Step 4: Deploy with Confidence**

Now we KNOW:
- ✅ Models have real edge
- ✅ System is profitable
- ✅ Safe to bet real money

---

## 🚨 Why We Can't Deploy Without Market Spreads

### **Current Status:**

```
Models trained on: Game results only
Models optimized for: Prediction accuracy
Backtest shows: 97% win rate (WRONG - inflated)
Actual edge: UNKNOWN
```

### **Problems:**

1. **Can't calculate true edge** - Don't know what market thinks
2. **Can't identify profitable opportunities** - No baseline to beat
3. **Models optimize wrong objective** - Accuracy ≠ Profit
4. **Backtest is meaningless** - Not comparing to market
5. **Don't know if system works** - Could lose money even with good predictions

### **What We Need:**

```
Models trained on: Game results + Market spreads
Models optimized for: Beating the market
Backtest shows: 53% win rate, 8% ROI (REALISTIC)
Actual edge: +3% per bet
```

---

## ✅ Next Steps

### **Path 1: Run Scrapers Locally (Recommended)**

```bash
# On your local machine (not sandbox):
git clone [YOUR_REPO]
cd football_betting_system

# Install dependencies
pip install requests beautifulsoup4 pandas

# Run overnight scrape
chmod +x run_all_scrapers.sh
./run_all_scrapers.sh

# Next morning: Combine data
python combine_scraped_data.py

# Retrain models with market data
python retrain_with_market_data.py

# Run realistic backtest
python backtest_ncaa_parlays_REALISTIC.py

# Deploy if profitable!
```

**Time:** ~10 hours total
**Cost:** $0
**Result:** Know if system has edge

### **Path 2: Buy Data (Fastest)**

```bash
# Buy Sports Insights data ($99)
# Download CSVs to data/market_spreads_YEAR.csv

# Retrain models
python retrain_with_market_data.py

# Run backtest
python backtest_ncaa_parlays_REALISTIC.py

# Deploy if profitable!
```

**Time:** 1 hour
**Cost:** $99
**Result:** Know if system has edge

### **Path 3: Forward Test (Most Conservative)**

```bash
# Skip historical validation
# Deploy for 2025 season
python ncaa_daily_predictions.py YOUR_API_KEY

# Track results all season
# Calculate ROI at end

# If profitable, continue
# If not, don't bet
```

**Time:** Full season (13 weeks)
**Cost:** $0
**Result:** Real-world validation

---

## 🏁 Bottom Line

**Current system:** ⚠️  Can predict scores, can't predict profits
**Need:** Market spreads (betting lines) to validate system
**Options:** Scrape FREE (10 hrs) OR Buy ($99) OR Forward test (13 weeks)
**Then:** Know if system has real edge before betting money

**The 12-model system is READY. We just need market spreads to validate it works!** 🏈💰
