---
name: tuesday-maction-analysis
description: Complete Tuesday MACtion betting analysis workflow. Triggers on "tuesday", "maction", "mac conference", or "analyze tuesday game". This is a PERSISTENT workflow that survives across sessions.
---

# Tuesday MACtion Analysis Workflow

## PRINCIPLE: Persistent Workflow (Not Re-Teaching)
Investment → System: Don't re-explain workflow each session. Skill persists it.

## WHY TUESDAY MACTION?
- **Softest lines of the week** - Books don't focus on it
- **Usually 1 game** - Easy to analyze deeply
- **Low public betting volume** - Less line movement
- **Perfect validation opportunity** - Test system with minimal risk

## WORKFLOW (Follow Exactly)

### Step 1: Fetch Tuesday Games
```bash
python ncaa_daily_predictions.py 0c405bc90c59a6a83d77bf1907da0299
```

**Expected output:**
- List of Tuesday games (usually 1-2 MAC conference games)
- Market spreads from DraftKings/FanDuel
- Game times (usually 7-8 PM ET)

**If no games:**
- Check day of week (MACtion only happens Tue/Wed during season)
- Season runs Sep-Nov (not available in off-season)

### Step 2: Run 12-Model Predictions
```bash
python ncaa_live_predictions_2025.py
```

**What it does:**
- Loads all 12 trained models
- Engineers features for each game
- Ensemble prediction with confidence
- Calculates edge vs market spread

**Required output:**
- Predicted spread (e.g., "Toledo -3.5")
- Confidence (70%+ required for bet)
- Edge value (3%+ required for bet)
- Model agreement score

### Step 2.5: Run Contrarian Intelligence (AUTOMATIC)
```bash
# This runs automatically in ncaa_deepseek_r1_analysis.py
# Or run standalone:
python ncaa_contrarian_intelligence.py <API_KEY>
```

**What it does:**
- Fetches line movement (opening vs current)
- Estimates public betting percentages
- Detects sharp money (reverse line movement)
- Generates contrarian strength (0-5 stars)

### Step 2.75: DeepSeek R1 Meta-Analysis (THE EDGE FINDER!)
```bash
# This runs automatically in ncaa_deepseek_r1_analysis.py
python ncaa_deepseek_r1_analysis.py <ODDS_KEY> <DEEPSEEK_KEY>
```

**What R1 does:**
- **Analyzes all 12 model predictions collectively**
- Finds PATTERNS models are seeing
- Identifies CONSENSUS (strong signal)
- Detects DISAGREEMENTS (why? edge or uncertainty?)
- Determines what VEGAS IS MISSING
- Provides detailed REASONING

**R1 is the meta-layer:**
- 12 models each see different patterns
- R1 synthesizes what they're all seeing
- R1 finds edges Vegas doesn't price in
- R1 explains WHY to bet (or not bet)

**Based on NFL system:**
- 60.91x returns ($100 → $6,091 over 10 years)
- R1 meta-analysis beats any single model
- Simple (R1 alone) beats complex (combinations)

**Contrarian signals detected:**
- 🚨 **5 stars**: Extreme public overload + sharp fade
- ⭐⭐⭐⭐ **4 stars**: Very strong contrarian signal
- ⭐⭐⭐ **3 stars**: Strong signal - consider fading
- ⭐⭐ **2 stars**: Weak signal - be aware
- ⭐ **1 star**: Minimal signal

**NCAA-specific patterns:**
- Home teams get 55-65% public money (home bias)
- Big name schools (Alabama, Ohio State) get 5-10% extra public
- MACtion games especially vulnerable to public overreaction
- Public threshold: 65%+ (lower than NFL's 70%)

**Alert examples:**
```
🚨 STRONG CONTRARIAN SIGNAL
   • Public 78% on home team (too heavy!)
   • Line moved from -3.5 to -2.5 (toward away)
   • Sharp money detected on away team

💡 Recommendation: FADE HOME - Take AWAY team
```

**Integration:**
- Automatically runs after model predictions
- Alerts if strength ≥ 3 stars
- Influences final bet decision

### Step 3: Validate Against Thresholds

**AUTOMATIC VALIDATION** (hook enforces):
- ✅ Confidence ≥ 70%
- ✅ Edge ≥ 3%
- ✅ Model agreement ≥ 80%
- ✅ Stake within limits ($20-$500)

**Manual checks:**
- Recent team form (last 3 games)
- Key injuries or weather
- Conference matchup history
- Line movement (opening vs current)

### Step 4: Calculate Bet Size

**Fractional Kelly (25%):**
```
Edge = |Predicted - Market| / 7.0
Kelly% = (Confidence - 0.5) / 0.5
Bet = Bankroll × Kelly% × 0.25
```

**Example:**
- Bankroll: $10,000
- Confidence: 75%
- Edge: 5%
- Kelly%: 50%
- Bet: $10,000 × 0.50 × 0.25 = **$1,250** → Cap at $500 max

**Confidence-based bet sizing:**
- 70-75%: $100-200 (1-2 units)
- 75-80%: $200-300 (2-3 units)
- 80-85%: $300-400 (3-4 units)
- 85%+: $400-500 (4-5 units, capped)

### Step 5: Generate Betting Report

**Create summary:**
```
🏈 TUESDAY MACTION PICK

Game: Toledo @ Bowling Green
Time: 7:00 PM ET (Tuesday)
Network: ESPN2

PREDICTION:
- Model Consensus: Toledo -4.5
- Market Spread: Toledo -3.0
- Edge: 1.5 points (21% value)

CONFIDENCE:
- Overall: 76%
- Model Agreement: 11/12 models agree
- Historical: Toledo 8-2 ATS in MACtion

🎯 CONTRARIAN INTELLIGENCE:
- Strength: ⭐⭐⭐ (3/5)
- Public: 68% on Toledo (heavy)
- Line Movement: -3.5 → -3.0 (toward BG)
- Recommendation: CONSIDER FADING - Take BG +3.0
- Signals Detected:
  • Public heavy on Toledo home favorite
  • MACtion game - public often overreacts
  • No sharp money detected (neutral)

⚠️  CONTRARIAN ALERT: Public might be overvaluing Toledo!

🧠 DEEPSEEK R1 META-ANALYSIS:
- R1 Pick: Toledo -3.0
- R1 Confidence: 76%
- R1 Bet Size: 4 units ($20 from $100 bankroll)

📊 PATTERNS R1 DETECTED:
• 11/12 models agree Toledo covers (strong consensus)
• Momentum model highest confidence (Toledo hot streak)
• Pace/tempo model dissents (favors underdog slightly)
• XGBoost sees offensive mismatch market missing
• Advanced stats show Toledo EPA edge significant

🎯 WHAT VEGAS IS MISSING:
"Market pricing Toledo as -3.0, but advanced metrics suggest they should be -4.5 to -5.0.
Public loading Toledo creates value ON Toledo (not against). Contrarian signal misleading here -
this is a case where 68% public is RIGHT. Toledo's offensive efficiency vs BG's defensive weakness
is underpriced. MACtion context: Books don't spend resources analyzing these games deeply.
Edge: 1.5-2 points of value on Toledo."

🤔 R1 REASONING:
"Analyzing 12-model ensemble reveals strong pattern: offensive efficiency models (XGBoost,
Advanced Stats, Drive Outcomes) all project Toledo -4.5 or higher. Only pace/tempo and special
teams models show neutral/underdog lean (minor factors). Contrarian intelligence shows public
heavy on Toledo BUT this appears to be PUBLIC BEING RIGHT scenario - not fade spot. Toledo's
last 3 games show 35+ PPG while BG allowing 30+ PPG. Market at -3.0 undervalues this matchup
significantly. Recommendation: Take Toledo -3.0 despite public load."

BET RECOMMENDATION (R1 FINAL):
- Side: **Toledo -3.0**
- Stake: **4 units** (75-79% confidence tier)
- Amount: $20 (from $100 bankroll)
- Risk: Medium
- Expected Value: +$15 per bet
- Edge: Market missing 1.5-2 points of value

RATIONALE:
- 11/12 models agree (strong consensus)
- R1 identifies offensive mismatch Vegas underpricing
- Contrarian signal present BUT R1 analysis shows public correct here
- MACtion edge: Books don't analyze these games deeply
- DECISION: Trust R1 meta-analysis over simple contrarian rule
```

### Step 6: Place Bet (If Validated)

**Only proceed if:**
- ✅ All thresholds met
- ✅ Hook validation passed
- ✅ Manual checks look good
- ✅ User approved

**Use bet placement skill:**
- Execute skill: `place-ncaa-bet`
- Log bet to tracking system
- Update bankroll state

### Step 7: Track & Learn

**Post-game (Wednesday morning):**
```bash
python track_bet_results.py
```

**Analyze:**
- Did model prediction beat actual spread?
- Was edge estimate accurate?
- Model performance breakdown
- Update calibration if needed

## ERROR HANDLING

**If predictions fail:**
1. Check if models trained: `ls -la models/ncaa/*.pkl`
2. Check data available: `python -c "from ncaa_models.feature_engineering import NCAAFeatureEngineer; e = NCAAFeatureEngineer(); print(len(e.load_season_data(2024)))"`
3. Fallback: Use manual analysis from odds comparison sites

**If no Tuesday games:**
- Check college football schedule (MACtion Oct-Nov only)
- Analyze Wednesday MAC games instead
- Wait for Saturday main slate

**If confidence < 70%:**
- DO NOT BET (system blocks this anyway via hook)
- Analyze why (model disagreement? insufficient edge?)
- Use as learning opportunity, not betting opportunity

## PERSISTENCE

This workflow persists across ALL future sessions.

**Next Tuesday:**
1. You mention "Tuesday game"
2. I automatically invoke this skill
3. Follow exact workflow above
4. No re-teaching required

**Investment:** 1 hour to create skill
**Operational cost:** 0 hours forever

## SUCCESS CRITERIA

**Short-term (First 5 bets):**
- 3+ wins out of 5 (60%+ win rate)
- Positive profit (even 1 unit = success)
- System follows workflow exactly

**Long-term (Full season):**
- 58-60% win rate (elite level)
- 5-10% ROI
- Model confidence calibrated correctly

## NOTES

- MACtion ends mid-November
- After MACtion, apply same workflow to Saturday games
- Workflow is day-agnostic, optimized for low-volume days
- Hook validation ensures you can't deviate from thresholds
