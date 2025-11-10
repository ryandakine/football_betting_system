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

BET RECOMMENDATION:
- Side: Toledo -3.0 (buying better line)
- Stake: $250 (2.5 units)
- Risk: Medium
- Expected Value: +$19 per bet

RATIONALE:
- Toledo elite offense vs BG weak defense
- Market undervaluing Toledo road dominance
- Sharp money moving toward Toledo
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
