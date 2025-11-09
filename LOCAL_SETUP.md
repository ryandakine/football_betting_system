# 🏈 Run NFL Betting System on Your Local Machine

## Quick Setup (5 minutes)

### 1. Clone the Repository

```bash
# Clone your repo
git clone https://github.com/ryandakine/football_betting_system.git
cd football_betting_system

# Or pull latest if already cloned
git pull origin main
```

### 2. Create .env File

Create a file called `.env` in the project root with your API keys:

```bash
# Copy this content to .env file
THE_ODDS_API_KEY=e84d496405014d166f5dce95094ea024
ODDS_API_KEY=e84d496405014d166f5dce95094ea024
OPENWEATHER_API_KEY=6dde025b7e7e2fc2227c51ac72acb719

# AI Models (get fresh keys from providers)
CLAUDE_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
PERPLEXITY_API_KEY=your_key_here
GROK_API_KEY=your_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Predictions!

```bash
# One command to get predictions
./run_nfl_predictions.sh
```

## That's It!

The system will:
- ✅ Fetch live NFL odds
- ✅ Run AI analysis
- ✅ Generate betting recommendations
- ✅ Save results to JSON

---

## Alternative: Manual Python Run

If you prefer to run step-by-step:

```bash
# Step 1: Scrape odds
python3 nfl_odds_scraper.py

# Step 2: Analyze with AI
python3 analyze_scraped_odds.py

# Or use Hugging Face models
python3 hf_token_analyzer.py
```

---

## Why This Works Locally But Not in Claude Code

**Claude Code Environment:**
- ❌ Network restrictions for security
- ❌ Can't access external APIs
- ❌ Blocks betting/gaming sites

**Your Local Machine:**
- ✅ Full network access
- ✅ Can call all APIs
- ✅ No restrictions

---

## Troubleshooting

### "No module named 'X'"
```bash
pip install -r requirements.txt
```

### "API key invalid"
- Get fresh keys from the provider websites
- Update .env file
- Make sure .env is in the project root

### "Permission denied: ./run_nfl_predictions.sh"
```bash
chmod +x run_nfl_predictions.sh
```

---

## 🎯 For Tonight's Game

Since the game is starting now, manually bet:

**RAIDERS +9.5** (1-2 units)

Then set this up for future games!

---

**Questions?** Check the RUN_NFL_SCRAPER.md file for more details.
