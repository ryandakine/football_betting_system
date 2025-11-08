# 🚀 NCAA Data Collection - Warp AI Quick Start

This guide shows you how to collect NCAA football data using Warp AI (or any environment with proper network access).

## ✅ Your API Key is Already Configured!

```bash
CFB_DATA_API_KEY=M9/VpZQNUSQfSUd6OtHZaTetz9/hH2zAqSqQcNLxLTheI43qhvEWwgpe+n6Rzg7G
```

This is stored in your `.env` file and ready to use.

---

## 🎯 Quick Start (3 Steps)

### Step 1: Run the Collector Script in Warp AI

```bash
python warp_ai_ncaa_collector.py
```

This will:
- ✅ Test your API key
- 📊 Collect NCAA game data
- 📈 Collect team statistics
- ⭐ Try to collect SP+ ratings (if you have Silver/Gold tier)
- 💾 Save everything to `data/football/historical/ncaaf/`

### Step 2: Choose Seasons to Collect

When prompted, enter the seasons you want:

```
Enter seasons (comma-separated, e.g., 2023,2024): 2023,2024
```

**Recommended options:**
- **Quick test**: `2024` (just current season)
- **Recent data**: `2023,2024` (recommended)
- **Full history**: `2015,2016,2017,2018,2019,2020,2021,2022,2023,2024`

### Step 3: Use the Data

Once collected, run:

```bash
# Backtest your system
python demo_ncaa_backtest.py

# Analyze today's games
python analyze_today_games.py

# Run live predictions
python run_ncaaf_today.py
```

---

## 🔍 Alternative: Test API Key Only

If you just want to verify your API key works:

```bash
python test_cfb_api_auth.py
```

This tests different authentication methods and shows which one works.

---

## 📊 What Data Gets Collected?

### Free Tier (All Users)
- ✅ **Game Results** - All FBS games (2015-2024)
- ✅ **Team Stats** - Season statistics for all teams
- ✅ **Rankings** - AP Poll, Coaches Poll, etc.
- ✅ **Basic Metrics** - Scores, records, conferences

### Silver/Gold Tier ($25/month - Optional)
- ⭐ **SP+ Ratings** - Advanced team rankings
- ⭐ **EPA Data** - Expected Points Added
- ⭐ **Win Probability** - In-game win probabilities

**Note:** Free tier is enough to build a profitable system! Upgrade only after you prove profitability.

---

## 📁 Output Files

After collection, you'll have:

```
data/football/historical/ncaaf/
├── ncaaf_2023_games.json        # All 2023 games
├── ncaaf_2023_stats.json        # 2023 team stats
├── ncaaf_2023_sp_ratings.json   # SP+ (if Silver/Gold)
├── ncaaf_2023_summary.json      # Collection summary
├── ncaaf_2024_games.json        # All 2024 games
├── ncaaf_2024_stats.json        # 2024 team stats
└── ...
```

---

## 🐛 Troubleshooting

### "❌ UNAUTHORIZED: Invalid API key"

**Solution:**
1. Go to https://collegefootballdata.com
2. Log in to your account
3. Verify your email (check spam folder)
4. Copy your API key from the dashboard
5. Update `.env` file with the new key

### "❌ FORBIDDEN: Access denied"

**Solution:**
- Your account needs email verification
- Go to https://collegefootballdata.com
- Check for verification email
- Click the verification link

### "❌ CONNECTION ERROR"

**Solution:**
- Check your internet connection
- Make sure you're in Warp AI (not restricted environment)
- Try again in a few minutes

### "⚠️ SP+ data requires Silver/Gold tier"

**Solution:**
- This is normal if you have free tier
- Free tier still gets all game data and basic stats
- SP+ is optional - upgrade only if system is profitable

---

## 💡 Pro Tips

1. **Start Small** - Collect just 2023-2024 first (fast, tests system)
2. **Verify Data** - Check the summary.json files to confirm collection
3. **Backtest First** - Always backtest before live betting
4. **Rate Limiting** - The script respects API rate limits automatically

---

## 📈 Expected Results

### Free Tier
- **Games per season**: ~800 FBS games
- **Collection time**: 2-3 minutes per season
- **API calls**: ~15 calls per season

### What You Can Build
- ✅ Win probability models
- ✅ Spread prediction models
- ✅ Over/Under models
- ✅ Conference-based strategies
- ✅ Full backtesting (2015-2024)

---

## 🎯 Next Steps After Collection

1. **Verify Data**
   ```bash
   ls -lh data/football/historical/ncaaf/
   ```

2. **Run Backtest**
   ```bash
   python demo_ncaa_backtest.py
   ```

3. **Check ROI**
   - If ROI > 5%: You have a viable system!
   - If ROI > 15%: Consider upgrading to Silver tier
   - If ROI < 0%: Tune your models first

4. **Go Live** (only after positive backtest)
   ```bash
   python run_ncaaf_today.py
   ```

---

## 📞 Support

- **API Issues**: https://collegefootballdata.com
- **System Issues**: Check the demo scripts and documentation
- **API Docs**: https://api.collegefootballdata.com/api/docs

---

## ⚠️ Important Notes

- ✅ Your API key is already in `.env` - no need to edit it
- ✅ Free tier is sufficient for most users
- ✅ Always backtest before live betting
- ⚠️ Never share your API key publicly
- ⚠️ Respect API rate limits (script handles this)

---

**Ready to collect data? Run this in Warp AI:**

```bash
python warp_ai_ncaa_collector.py
```

Good luck! 🏈💰
