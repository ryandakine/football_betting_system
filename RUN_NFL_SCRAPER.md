# 🏈 Quick Start Guide - NFL Odds Scraper

## When You Get Home for Lunch

### Option 1: One Command (Easiest!)

```bash
cd ~/football_betting_system
chmod +x run_nfl_predictions.sh
./run_nfl_predictions.sh
```

This will automatically:
1. ✅ Scrape live odds from multiple websites
2. ✅ Run all 4 AI models (Claude, GPT-4, Grok, Perplexity)
3. ✅ Generate betting recommendations
4. ✅ Save results to `data/nfl_analysis_results.json`

---

### Option 2: Step by Step

#### Step 1: Scrape Odds
```bash
cd ~/football_betting_system
python3 nfl_odds_scraper.py
```

This scrapes NFL odds from:
- ESPN
- OddsShark
- Covers.com
- Action Network
- Bovada

#### Step 2: Analyze with AI
```bash
python3 analyze_scraped_odds.py
```

This runs the scraped odds through your 4 AI models and generates predictions.

---

## What You'll Get

### Console Output
```
🏈 Game: Cincinnati Bengals @ Baltimore Ravens

🤖 AI Model Predictions:

  CLAUDE:
    Pick: Baltimore Ravens
    Confidence: 68.5%
    Reasoning: Ravens strong at home, Bengals defense struggling

  GPT-4:
    Pick: Baltimore Ravens
    Confidence: 71.2%
    Reasoning: Historical advantage and better run defense

  GROK:
    Pick: Baltimore Ravens
    Confidence: 65.0%
    Reasoning: Home field advantage key factor

  PERPLEXITY:
    Pick: Baltimore Ravens
    Confidence: 69.8%
    Reasoning: Recent form favors Ravens

🎯 CONSENSUS:
    Pick: Baltimore Ravens
    Confidence: 68.6%
    Agreement: 100%
```

### Saved Files
- `data/scraped_nfl_odds.json` - Raw scraped odds
- `data/nfl_analysis_results.json` - Complete AI analysis

---

## Troubleshooting

### If Scraping Fails
The scraper tries multiple sources. If all fail:
1. Check your internet connection
2. Try running again (websites sometimes have rate limits)
3. Manually provide odds (see Option 3 below)

### If AI Analysis Fails
Check that your API keys are working:
```bash
python3 -c "from tri_model_api_config import validate_api_configuration; print(validate_api_configuration())"
```

---

## Option 3: Manual Odds Input

If scraping doesn't work, create a file `manual_game.json`:

```json
{
  "home_team": "Baltimore Ravens",
  "away_team": "Cincinnati Bengals",
  "home_moneyline": -200,
  "away_moneyline": +170,
  "spread": -4.5,
  "total": 49.5
}
```

Then run:
```bash
python3 analyze_manual_game.py manual_game.json
```

---

## Need Help?

1. Make sure you're connected to home WiFi (not work network)
2. Check that .env file has all API keys
3. Run `python3 -m pip install -r requirements.txt` if you get import errors
4. Install scraping dependencies: `pip3 install beautifulsoup4 lxml`

---

**Estimated Time: 2-3 minutes total**

Good luck! 🍀
