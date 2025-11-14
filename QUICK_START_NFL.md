# 🏈 NFL WEEKEND - QUICK START (5 Minutes)

## **RIGHT NOW (Next 30 Minutes)**

### **Step 1: Get Crawlbase Token (5 mins)**

```bash
# 1. Go to: https://crawlbase.com/signup
# 2. Sign up for FREE account
# 3. Copy your API token from dashboard
# 4. Save it:

export CRAWLBASE_TOKEN='your_token_here'
```

### **Step 2: Test It Works (2 mins)**

```bash
cd /home/user/football_betting_system

# Install Crawlbase
pip install crawlbase

# Quick test
python3 -c "
from crawlbase import CrawlingAPI
api = CrawlingAPI({'token': 'YOUR_TOKEN_HERE'})
response = api.get('https://www.espn.com/nfl/scoreboard')
print('✅ Working!' if response['statusCode'] == 200 else '❌ Failed')
"
```

### **Step 3: Run Complete Workflow (3 mins)**

```bash
# All-in-one command
./nfl_weekend_quickstart.sh

# OR run manually:
python3 crawlbase_nfl_scraper.py          # Collect data
python3 unified_nfl_intelligence_system.py # Run predictions
python3 kelly_calculator.py --bankroll 20  # Calculate bet sizes
```

---

## **SUNDAY BETTING WORKFLOW**

### **Morning (8:00 AM)**

```bash
# Get fresh data
python3 crawlbase_nfl_scraper.py

# Run predictions
python3 unified_nfl_intelligence_system.py > sunday_picks.txt

# Filter for STRONG_BET
grep "STRONG_BET\|65%\|70%" sunday_picks.txt
```

### **Before Early Games (12:45 PM)**

```bash
# Calculate bet sizes for early games
python3 kelly_calculator.py --bankroll 20

# Expected: 2-3 bets @ $0.75-1.50 each
# Line shop: DraftKings, FanDuel, BetMGM
# Place bets by 12:55 PM
```

### **Before Late Games (3:50 PM)**

```bash
# Recalculate for late games
python3 kelly_calculator.py --bankroll 20

# Expected: 1-2 bets @ $0.75-1.50 each
# Place bets by 4:00 PM
```

### **Before SNF (7:45 PM)**

```bash
# Final picks for Sunday Night
python3 kelly_calculator.py --bankroll 20

# Expected: 0-1 bet @ $1.00-2.00
# Place bet by 8:15 PM
```

---

## **TODAY (Friday) - NO GAMES**

**What to do:**

1. ✅ Set up Crawlbase account
2. ✅ Install dependencies
3. ✅ Test scraper works
4. ✅ Review your betting system
5. ✅ Set alarms for Sunday

**Tomorrow (Saturday):**

- Run full analysis for Sunday games
- Identify top 3-5 STRONG_BET picks
- Calculate exact bet sizes
- Line shop across sportsbooks

**Sunday:**

- Execute your betting plan
- Track results
- Profit! 💰

---

## **FILES YOU NEED**

| File | Purpose |
|------|---------|
| `NFL_WEEKEND_BETTING_GUIDE.md` | Complete strategy guide |
| `crawlbase_nfl_scraper.py` | Automated data collection |
| `kelly_calculator.py` | Bet sizing calculator |
| `nfl_weekend_quickstart.sh` | One-command workflow |
| `unified_nfl_intelligence_system.py` | Main prediction engine |

---

## **EXPECTED RESULTS**

**Sunday:**
- 3-6 bets @ $0.75-2.00 each
- Total risk: $3-9 (from $20 NFL bankroll)
- Expected profit (65% win rate): $0.75-2.25
- Ending bankroll: $20.75-22.25

**Monthly (4 Sundays + MNF):**
- Expected profit: $5-10
- ROI: 25-50%
- Win rate: 65%+

---

## **BANKROLL MANAGEMENT**

```
Current total: $101
NFL allocation: $20
NCAA allocation: $80

NFL bet sizing:
- STRONG_BET (65%+): $0.75-2.00 per bet
- Max 3 bets per window (early/late/SNF)
- Never risk more than 10% per day ($2 max total risk)

Kelly Criterion:
- Using 0.25 fractional Kelly (conservative)
- Only bet when edge ≥ 2.5%
- Only bet when confidence ≥ 65%
```

---

## **TROUBLESHOOTING**

**Error: "CRAWLBASE_TOKEN not set"**
```bash
export CRAWLBASE_TOKEN='your_token_here'
# OR add to .env file
echo "CRAWLBASE_TOKEN=your_token_here" >> .env
```

**Error: "No module named crawlbase"**
```bash
pip install crawlbase
```

**Error: "unified_nfl_intelligence_system.py not found"**
```bash
# Use alternative:
python3 main.py
# OR
python3 production_main.py
```

---

## **SUPPORT**

**Read the guides:**
- `NFL_WEEKEND_BETTING_GUIDE.md` - Full weekend strategy
- `README.md` - System overview
- `NFL_LIVE_TRACKING_README.md` - Live tracking

**Test the tools:**
```bash
# Test Kelly calculator
python3 kelly_calculator.py --bankroll 20

# Test data scraper
python3 crawlbase_nfl_scraper.py --token YOUR_TOKEN

# Test quick start
./nfl_weekend_quickstart.sh --help
```

---

## **NEXT ACTION**

**Right now:**
1. Go to https://crawlbase.com/signup
2. Get your free API token
3. Run: `export CRAWLBASE_TOKEN='your_token'`
4. Test: `python3 crawlbase_nfl_scraper.py`

**Saturday morning:**
1. Run: `./nfl_weekend_quickstart.sh`
2. Review predictions
3. Plan your Sunday bets

**Sunday:**
1. Execute betting plan
2. Track results
3. Profit! 💰

---

**Questions? Check `NFL_WEEKEND_BETTING_GUIDE.md` for complete details.**

**Ready to win? Let's go! 🚀**
