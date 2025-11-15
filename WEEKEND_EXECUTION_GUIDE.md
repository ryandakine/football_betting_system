# 🎬 WEEKEND EXECUTION GUIDE - WEEK 11

**Goal**: Validate system with real money at small stakes

**Weekend**: Friday 8PM - Sunday 6PM

**Expected**: 5-8 bets, 55-60% win rate, $2-5 profit

---

## 📅 FRIDAY (Tonight, 8:00 PM)

### Step 1: Generate NCAA Predictions (8:00 PM)
```bash
cd ~/code/football_betting_system

# Option A: If you have ncaa_live_week_11_plus.py:
python3 ncaa_live_week_11_plus.py $ODDS_API_KEY

# Option B: Generate from predictions you already have:
python3 betting_plan_generator.py 11 --sport NCAA
```

**What you're looking for:**
- All Week 11 matchups with confidence scores
- Odds from sportsbooks
- Logs to `data/predictions/prediction_log.json`

### Step 2: Apply Edge Detector (8:15 PM)
```bash
python3 bet_selector_unified.py ncaa
```

**What you'll see:**
- Games organized by tier
- Expected win rates per tier
- Recommended bet sizes
- Edge factors explained

**Example output:**
```
TIER 1 MEGA EDGE: [None expected Week 11]
TIER 2 SUPER EDGE: [None expected Week 11]
TIER 3 STRONG EDGE: 1-2 games (Big Ten/Mountain West homes)
TIER 4 MODERATE EDGE: 2-3 games
TIER 5 SELECTIVE: Remaining games
```

### Step 3: Video Scout Top Candidates (8:30 PM - 10:00 PM)
**Watch video editions for:**
- TIER 3 games (your best bets)
- Top TIER 4 games if there are 3+ TIER 3s

**Look for:**
- Team motivation (energy level, focus)
- Injury status (key players available?)
- Defensive soft spots
- Team chemistry

**Document in notes:**
```
Game: Ohio State vs Michigan
Video Notes: 
- Michigan showing focused energy (well-motivated)
- Ohio State defense has gap issues vs spread
- Key WR available (confirmed)
- Betting: Michigan slightly undervalued at +3
Confidence boost: +2%
```

### Step 4: Cross-Check With Week 11 Plan (10:00 PM)
```bash
cat WEEK_11_BETTING_PLAN.md
```

**Reality check:**
- "Are my top picks matching the expected TIER distribution?"
- "Do the video scouting notes align with tier predictions?"
- "Any surprises I need to account for?"

### Step 5: Finalize Saturday Bets (10:30 PM)
**Select 3-5 games for Saturday:**

**Decision rules:**
```
✅ TIER 3 games → AUTO-BET (if confidence ≥ 60%)
✅ TIER 4 games → BET if video scouting confirms
❌ TIER 5 games → SKIP (too low edge)
❌ Any game < 55% confidence → SKIP
```

**Sizing per tier:**
```
TIER 3: 1.5% bankroll = $1.50 per game
TIER 4: 1.2% bankroll = $1.20 per game
TIER 5: 1.0% bankroll = $1.00 per game
```

**Saturday bet slip example:**
```
1. Ohio State vs Michigan (-3) ✅ TIER 4 → $1.20
2. Penn State vs Rutgers (-21) ✅ TIER 3 → $1.50
3. Wisconsin vs Minnesota (-7) ✅ TIER 4 → $1.20

Total risk: $3.90
Expected profit (60% WR): $0.78
```

**Document everything in a file:**
```
~/.betting_system/saturday_bets_week11.txt
```

---

## 🏈 SATURDAY (Game Day)

### 10:00 AM - Final Checks
```bash
# Pull latest lines
python3 ncaa_live_week_11_plus.py --lines-only

# Check for weather alerts
# Check for last-minute injuries
# Verify game times haven't changed
```

**Look for changes:**
- Lines moved more than 0.5 points? (might indicate sharp action)
- New injuries reported? (update confidence)
- Weather became bad? (affects running game)

### 11:30 AM, 3:00 PM, 6:30 PM
**Place bets for each time slot:**

**Kickoff window: 11:30 AM games**
- Place bets 30 min before kickoff
- Watch for steam moves in final 30 min
- Record: bet slip, odds, time

**Kickoff window: 3:00 PM games**
- Place bets around 2:30 PM
- Repeat monitoring

**Kickoff window: 6:30 PM games**
- Place bets around 6:00 PM
- Repeat monitoring

**During games:**
- Track results in real-time
- Update `prediction_log.json` with outcomes
- Document any surprises

**End of day:**
- Calculate Saturday win rate
- Update bankroll
- Document learnings

---

## 🏈 SUNDAY (NFL Day)

### 10:00 AM - Generate NFL Predictions
```bash
# Read narrative tracker for context
python3 nfl_narrative_tracker.py --week 11

# Generate NFL predictions
python3 bet_selector_unified.py nfl
```

**Week 11 NFL Reality:**
- Home win rate: 50% (basically coin flip)
- Away bias emerging
- Late-season desperation increases variance
- Expect 2-3 quality bets (not 5-8)

**Tier distribution expected:**
```
TIER 1-2: Very rare (maybe 0-1)
TIER 3: 1-2 games (strong division homes)
TIER 4: 1-2 games
TIER 5: Rest
```

### 10:30 AM - Narrative Context

**Read narratives for each top candidate:**
```
Game: Chiefs @ Raiders
Narrative Type: DIVISION_LEADER (KC) vs REBUILDING (LV)
Narrative Arc: Chiefs clinching division, looking ahead?
Betting Edge: FADE (trap game potential)

Game: Bills @ Dolphins
Narrative Type: Both DIVISION_LEADER contenders
Narrative Arc: Playoff positioning crucial
Betting Edge: BET (maximum motivation)
```

### 11:00 AM - Final NFL Selections
**Select 2-3 games for Sunday:**

**Decision rules (same as Saturday):**
```
✅ TIER 3 + Positive narrative → BET
✅ TIER 3 but TRAP narrative → SKIP
❌ TIER 4-5 → SKIP (too much variance, late season)
```

**NFL bet slip example:**
```
1. Bills vs Dolphins (spread TBD) ✅ TIER 3 → $1.50
2. [Maybe one more if it looks good]

Total risk: $1.50-3.00
Expected profit (55% WR): $0.15-0.55
```

### 12:30 PM - Place NFL Bets
- Monitor lines until 1:00 PM kickoff
- Place bets 30 min before
- Record everything

### During Games
- Track results
- Update log
- Document any surprises

### 6:00 PM - End of Weekend Summary

**Document:**
```
SATURDAY (NCAA):
- Bets: 4 games
- Results: 2-2 (50% - below target)
- Profit: -$0.40
- Notes: [what went wrong/right]

SUNDAY (NFL):
- Bets: 1 game
- Results: 1-0 (100%)
- Profit: +$1.50
- Notes: [what went wrong/right]

WEEKEND TOTAL:
- Bets: 5
- Results: 3-2 (60% - ON TARGET)
- Profit: +$1.10
- Bankroll: $102.10
- Confidence: VALIDATED ✅
```

---

## ✅ Success Criteria for Weekend

### **Minimum (Validates System):**
- Win rate: 55-60%
- Profit: $1-3
- No major surprises
- System worked as expected

### **Good (Exceeded Target):**
- Win rate: 60-65%
- Profit: $4-6
- Video scouting helped
- Confidence in decisions

### **Excellent (Run It Back):**
- Win rate: 65%+
- Profit: $7+
- Multiple TIER 3 hits
- Ready to scale up

### **Needs Work (Don't Scale):**
- Win rate: < 50%
- Losses mounting
- Confidence scores off
- Need to debug before continuing

---

## 📊 Post-Weekend Analysis (Monday)

### Update Prediction Log
```bash
# View results
python3 track_ncaa_performance.py  # If you have this
python3 update_ncaa_results.py     # Update actual results
```

### Validate Calibration
**Question**: Did 60% confidence predictions actually win ~60% of the time?

**If NO:**
- Confidence too high → Use 0.85× multiplier
- Confidence too low → Boost by 0.5-2%
- Tier system off → Adjust thresholds

**If YES:**
- System is working
- Continue with small stakes
- Plan for scaling

### Document Learnings
```
Week 11 Validation Report:

What Worked:
- Tier 3 games (Big Ten homes) hit [X]% 
- Video scouting confirmed trends
- Narrative context helped NFL bets

What Didn't:
- TIER 4 games underperformed
- One surprise injury knocked out prediction
- Late-season unpredictability higher than expected

Adjustments for Week 12:
- Increase minimum TIER threshold to TIER 3 only
- Add injury checking 2 hours before kickoff
- Watch for bowl eligibility effects
```

---

## 💡 Key Reminders

### **Don't:**
- ❌ Over-bet because you're confident
- ❌ Chase losses by increasing bet size
- ❌ Ignore tier system and "feel good" bets
- ❌ Skip the calibration check Monday

### **Do:**
- ✅ Stick to tier-based sizing
- ✅ Document every bet and outcome
- ✅ Trust the system (55-60% is good)
- ✅ Validate before scaling

---

## 🚀 After This Weekend

### **If System Validates (≥55% win rate):**

**Week 12 Plan:**
- Read WEEK_12_BETTING_PLAN.md (bowl eligibility dynamics)
- Increase sizing slightly (+20%) if confident
- Add basketball predictions when ready

**By Thanksgiving:**
- Bankroll: $110-120
- Ready to add second sport
- Portfolio getting more stable

### **If System Needs Work (<55% win rate):**

**Debug priorities:**
1. Confidence calibration (multiply by 0.85?)
2. Tier thresholds (too aggressive?)
3. Video scouting (missing something?)
4. Edge detection (bugged?)

**Don't:**
- Keep betting same stakes on broken system
- Add capital to validate
- Panic or abandon

**Do:**
- Fix one thing at a time
- Run small tests
- Validate before scaling

---

## 📱 Live Monitoring Checklist

### **During Games:**

**For each game:**
- [ ] Confirm kickoff time (no delays)
- [ ] Check live score updates
- [ ] Watch for blowouts (early sign of confidence)
- [ ] Note unusual events (injuries, turnovers)
- [ ] Update prediction log in real-time

**At end of each game:**
- [ ] Record final score
- [ ] Calculate win/loss
- [ ] Update running total
- [ ] Note if result matches confidence level

---

## 🎯 Final Checklist Before Betting

**Friday night (before bets):**
- [ ] Predictions generated
- [ ] Edge selector run
- [ ] Video scouting complete
- [ ] Week 11 plan read
- [ ] Top 3-5 games identified
- [ ] Tier assignments confirmed
- [ ] Bet sizes calculated
- [ ] Tracking file prepared

**Saturday morning:**
- [ ] Weather checked
- [ ] Injuries confirmed
- [ ] Lines verified
- [ ] No surprises emerged

**Sunday morning:**
- [ ] NFL predictions ready
- [ ] Narratives checked
- [ ] 2-3 games selected
- [ ] Ready to execute

---

**Status**: Ready to execute tonight at 8:00 PM

**Bankroll**: $101
**Target Risk**: $6-12
**Expected Profit**: $2-5
**Expected Bankroll After**: $103-106

**Go validate this system.** 🎯
