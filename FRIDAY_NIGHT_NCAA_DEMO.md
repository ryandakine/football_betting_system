# Friday Night NCAA Predictions - Week 11

## 🏈 System Status

**Date:** Friday, November 14, 2025
**Week:** 11 (NCAA Football)
**Status:** ⚠️ API Key Issue (403 Forbidden)

---

## ⚠️ Current Issue: API Access

The Odds API key is returning `403 Forbidden`. This means:

1. **API Key Invalid/Expired** - Key may need renewal
2. **No Credits Remaining** - Free tier exhausted
3. **API Endpoint Changed** - Odds API updated their system

### Solution:

**Get a new API key:**
```bash
# Visit: https://the-odds-api.com/
# Sign up for free tier (500 requests/month)
# Set environment variable:
export ODDS_API_KEY="your_new_key_here"
```

---

## 🎯 How to Run Friday Night Predictions (When API Works)

### **Option 1: Simple Runner**
```bash
python3 run_friday_simple.py
```

### **Option 2: Full World Models**
```bash
python3 run_friday_night_ncaa.py
```

### **Option 3: Manual with Existing Tools**
```bash
# Fetch games
python ncaa_daily_predictions.py YOUR_API_KEY

# Run 12-model predictions
python ncaa_live_predictions_2025.py

# Run DeepSeek R1 meta-analysis
python ncaa_deepseek_r1_reasoner.py

# Apply world model boosts
python predict_ncaa_world_models.py
```

---

## 📊 What You'd Get (Example Output)

```
🏈 FRIDAY NIGHT NCAA - WEEK 11
================================================================================
Date: Friday, November 14, 2025

📊 Fetching Friday NCAA games...
✅ Found 12 games

================================================================================
GAME 1: Northern Illinois @ Toledo
================================================================================
Spread: Toledo -7.5
Total: 54.5
Kickoff: 7:00 PM ET

📊 12-Model Consensus:
   Base Confidence: 73.2%
   Model Agreement: 87.5%

⚡ World Model Boosts:
   After calibration: 76.9% (+3.7%)
   After interaction: 79.1% (+2.2%)
      → 3 interactions active
   After causal: 81.3% (+2.2%)

   TOTAL BOOST: +8.1%

🎯 RECOMMENDATION: STRONG BET
   Confidence: 81.3%
   Suggested Stake: 5% bankroll

📌 Signals:
   ✓ STRONG MODEL ALIGNMENT
   ✓ SIGNIFICANT CAUSAL FACTORS
   ✓ Weather favorable for UNDER
```

---

## 🔧 Typical Friday Night NCAA Schedule

### **MACtion (Mid-American Conference)**
- **Games:** 5-10 Friday night games
- **Kickoffs:** 7:00 PM - 8:30 PM ET
- **Teams:** Toledo, Bowling Green, Northern Illinois, Western Michigan, Ball State, etc.
- **Why Important:** Softer lines, less public betting

### **Conference USA**
- **Games:** 2-4 Friday games
- **Kickoffs:** 7:30 PM - 9:00 PM ET
- **Teams:** Liberty, UTEP, FIU, Louisiana Tech, etc.

### **Occasional Power 5**
- **Games:** 1-2 Friday night games
- **Conferences:** ACC, Big 12, Pac-12
- **Higher profile, sharper lines**

---

## 🎯 What Makes Friday Night Different

### **Advantages:**
1. **Less Sharp Action** - Books focus on Saturday slate
2. **Lower Betting Limits** - Less professional money
3. **Market Inefficiency** - Models have more edge
4. **Focused Analysis** - Only 5-10 games to analyze deeply

### **World Models Opportunity:**
- **Interaction Learning** - Friday games help train for Saturday
- **Causal Discovery** - Weather patterns, rest days, travel
- **Low Stakes Testing** - Validate system before big Saturday slate

---

## 📅 Next Steps

### **Immediate (Friday Night):**
1. ✅ Get new Odds API key
2. ✅ Run `run_friday_simple.py`
3. ✅ Analyze 5-10 Friday games
4. ✅ Place recommended bets

### **After Games (Saturday Morning):**
1. ✅ Update results in system
2. ✅ World models learn from Friday
3. ✅ Improved predictions for Saturday

### **Saturday (Main Slate):**
1. ✅ 50+ games to analyze
2. ✅ World models now trained on Friday data
3. ✅ Interaction & causal boosts active
4. ✅ Maximum edge opportunity

---

## 🚀 System Ready When You Are

**Files Created:**
- ✅ `run_friday_night_ncaa.py` - Full world models runner
- ✅ `run_friday_simple.py` - Simple API fetcher
- ✅ `predict_ncaa_world_models.py` - World model predictor
- ✅ `college_football_system/WORLD_MODELS_GUIDE.md` - Complete guide

**Status:**
- ✅ World models integrated
- ✅ Dual-reader R1 (Claude + DeepSeek)
- ✅ 12-model ensemble ready
- ✅ Causal discovery configured
- ⚠️ Need valid API key to fetch games

**Once you have a working API key, you're ready to roll!** 🏈
