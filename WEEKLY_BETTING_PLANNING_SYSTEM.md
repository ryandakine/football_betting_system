# 📅 PROACTIVE WEEKLY BETTING PLANNING SYSTEM

**Purpose**: Generate betting plans for upcoming weeks BEFORE games start, not scrambling day-of.

---

## 🎯 System Overview

Instead of reacting to week 11 on game day, we now prepare comprehensive betting plans for weeks 12-15 in advance. Each plan includes:

- **Expected patterns** for that specific week
- **Tier expectations** (how many TIER 1-5 games you'll see)
- **Conference-specific guidance** (which conferences have edges that week)
- **Week dynamics** (bowl eligibility, playoff implications, etc.)
- **Video scouting priorities** (which games to watch early)
- **Pre-week checklist** (everything to do before games start)

---

## 📊 Weekly Plans Generated

### ✅ Week 12: Bowl Eligibility Race
**File**: `WEEK_12_BETTING_PLAN.md`
- Home win rate: 52% (very slight edge)
- Tier focus: TIER 3-4 (Big Ten/Mountain West homes)
- Bets expected: 4-6 games
- Key dynamics: Teams fighting for 6-win bowl eligibility
- Video focus: Underdog motivation levels

### ✅ Week 13: Conference Title Races
**File**: `WEEK_13_BETTING_PLAN.md`
- Home win rate: 54% (still modest)
- Tier focus: TIER 3-4 (conference leaders)
- Bets expected: 4-6 games
- Key dynamics: Division races tightening, some teams eliminated
- Video focus: Rivalry dynamics, motivation levels

### ✅ Week 14: Conference Championship Week
**File**: `WEEK_14_BETTING_PLAN.md`
- Home win rate: 55% (improving slightly)
- Tier focus: TIER 3 on strong teams, TIER 5 for CCG prep rest
- Bets expected: 3-5 games
- Key dynamics: CCG matchups set, playoff implications critical
- Video focus: Top teams preparing for CCG

### ✅ Week 15: Final Regular Season
**File**: `WEEK_15_BETTING_PLAN.md`
- Home win rate: 56% (best late-season edge)
- Tier focus: TIER 3 on motivated teams
- Bets expected: 4-6 games
- Key dynamics: Playoff teams stay sharp, rest-of-field motivation varies
- Video focus: Transfer portal exodus, motivation assessment

---

## 🔄 Workflow: How to Use This System

### **Monday BEFORE the Week** (e.g., Monday before Week 12)

1. **Read the weekly plan**
   ```bash
   cat WEEK_12_BETTING_PLAN.md
   ```
   - Understand the dynamics (bowl eligibility, rankings, etc.)
   - See what tiers you should expect
   - Note key research areas

2. **Video scout the games**
   - Watch your system's video editions of Week 12 matchups
   - Identify soft spots, team chemistry, key player status
   - Document findings in prediction notes

3. **Gather information**
   - Check bowl eligibility tracker for all teams
   - Review playoff implications (if relevant)
   - Note key injuries
   - Identify rivalry games

### **Thursday BEFORE the Week**

4. **Run prediction model**
   ```bash
   python3 ncaa_live_week_12_plus.py $ODDS_API_KEY
   ```
   - Generates predictions for all Week 12 games
   - Logs to `data/predictions/prediction_log.json`

5. **Apply unified bet selector**
   ```bash
   python3 bet_selector_unified.py ncaa
   ```
   - Automatically assigns tiers based on discovered patterns
   - Shows expected win rates per tier
   - Recommends bet sizes

6. **Cross-reference with weekly plan**
   - Compare model predictions to expected tier distribution
   - Validate conference-specific patterns
   - Check for surprises

7. **Final vetting**
   - Watch top TIER 3-4 candidates on video
   - Confirm bowl eligibility/playoff status
   - Finalize bet list (don't modify day-of)

### **Friday BEFORE the Week**

8. **Prepare bet slip**
   - Finalize all wagers based on tier system
   - Set bet sizes per tier
   - Document in prediction log

9. **Set monitoring**
   - Enable line movement alerts
   - Note key numbers you're targeting
   - Prepare to catch steam moves

### **Saturday-Sunday of the Week**

10. **Execute & Monitor**
    - Place bets according to plan
    - Track line movements
    - Monitor in-game results

### **Monday AFTER the Week**

11. **Update results & analyze**
    ```bash
    python3 update_ncaa_results.py
    python3 track_ncaa_performance.py
    ```
    - Update actual results
    - Review win rate by tier
    - Document learnings

12. **Iterate for next week**
    - Note patterns that worked/didn't work
    - Adjust tier thresholds if needed
    - Prepare for next week's plan

---

## 📈 Pattern Expectations by Week

| Week | Phase | Home Win Rate | Tier Focus | Games Expected | Risk Level |
|------|-------|---|---|---|---|
| 1-2 | Early Season | 81%+ | TIER 1-2 | 8-10 | Low |
| 3-4 | Strong Home Field | 72%+ | TIER 2-3 | 8-10 | Low |
| 5-7 | Mid-Season | 65-68% | TIER 3-4 | 6-8 | Medium |
| 8-11 | Late Season | 48-57% | TIER 4-5 | 4-6 | Medium |
| **12** | **Bowl Eligibility** | **52%** | **TIER 3-4** | **4-6** | **Medium** |
| **13** | **Conference Race** | **54%** | **TIER 3-4** | **4-6** | **Medium** |
| **14** | **CCG Week** | **55%** | **TIER 3** | **3-5** | **Medium-High** |
| **15** | **Final Regular** | **56%** | **TIER 3** | **4-6** | **Medium** |

---

## 🎯 Key Differences: Proactive vs Reactive

### ❌ OLD REACTIVE WAY (e.g., Saturday Week 11)
- Friday night: "What games are there this week?"
- Saturday morning: Scramble to analyze
- Day-of research = rushed decisions
- No time to watch video editions
- Betting on gut instinct

### ✅ NEW PROACTIVE WAY (Week 12 prepared Monday)
- Monday: Already know what to expect
- Thursday: Model generates predictions
- Friday: Video confirmed + tier assigned
- Saturday: Execute pre-prepared plan
- Data-driven decisions with full context

**Result**: Better decisions, fewer surprises, more consistent ROI

---

## 📚 Supporting Materials

### Main Tools
- `betting_plan_generator.py` - Generates plans for any week
- `bet_selector_unified.py` - Applies tier system to predictions
- `ncaa_live_week_X_plus.py` - Generates live predictions

### Weekly Plans
- `WEEK_11_BETTING_PLAN.md` - Week 11 (current)
- `WEEK_12_BETTING_PLAN.md` - Next week (already prepared)
- `WEEK_13_BETTING_PLAN.md` - Week 13 (already prepared)
- `WEEK_14_BETTING_PLAN.md` - Week 14 (already prepared)
- `WEEK_15_BETTING_PLAN.md` - Week 15 (already prepared)

### Reference
- `COMPLETE_SYSTEM_INTEGRATION_GUIDE.md` - Tier system & patterns
- `NCAA_DEEP_EDGE_ANALYSIS.md` - Edge analysis foundation
- `SATURDAY_WEEK_11_BETTING_PLAN.md` - Week 11 specifics

---

## 💡 Video Scouting Advantage

Your system has video editions of upcoming games prepped. This is HUGE for weeks 12-15 because:

**Early Preparation** → **Video Scouting** → **Better Predictions**

### What to Look For in Videos (Weeks 12-15)

**Week 12**: Motivation levels for bowl eligibility desperate teams
- Are players engaged or mentally checked out?
- Is coaching staff pushing hard?
- Signs of team unity or discord?

**Week 13**: Conference title race implications
- How seriously are teams taking this?
- Any resting of starters?
- Team momentum indicators?

**Week 14**: CCG preparation effect
- Are starters being rested?
- Is focus already on next opponent?
- Physical vs mental fatigue?

**Week 15**: Playoff/bowl implications
- Motivation for teams in/out of contention
- Transfer portal departures visible?
- Team chemistry before bowl season?

---

## ✅ Pre-Week Checklist Template

Use this for EACH week:

### For Week 12:
- [ ] Read WEEK_12_BETTING_PLAN.md
- [ ] Watch video editions (3-4 key matchups)
- [ ] Check bowl eligibility tracker
- [ ] Note key injuries/player status
- [ ] Run prediction model Thursday
- [ ] Apply unified bet selector
- [ ] Cross-check with tier expectations
- [ ] Vet top candidates on video
- [ ] Finalize bet list by Friday
- [ ] Set monitoring alerts
- [ ] Execute Saturday
- [ ] Update results Sunday
- [ ] Analyze performance Monday

(Repeat for weeks 13, 14, 15)

---

## 🚀 Expected Outcomes (Weeks 12-15 Combined)

**Total Bets**: 15-23 games (4-6 per week)

**Expected Win Rate**: 58-63% average (realistic for late season)

**Expected ROI**: 5-10% per week (sustainable)

**Total Risk**: 6-12% of bankroll per week (conservative sizing)

**Video Prep Advantage**: +2-3% additional accuracy from early scouting

---

## 📞 Key Reminders

1. **Proactive ≠ Perfect**
   - Late-season games are inherently unpredictable
   - Realistic win rates are 55-65%, not 70%+
   - Smaller bets, higher variance expected

2. **Video scouting is your edge**
   - Day-of research = everyone's thinking the same
   - Pre-week video = unique insights
   - Motivation/fatigue visible on film, not in stats

3. **Consistency over heroics**
   - Skip games where edges are unclear
   - Stick to tier system (don't over-bet)
   - Track results by tier to validate system

4. **Preparation beats reaction**
   - Monday prep = Saturday confidence
   - Know what to expect before games start
   - Reduces emotional betting

---

## 🔄 Next Steps

1. **Read** `WEEK_12_BETTING_PLAN.md` for detailed analysis
2. **Prepare** video scouting list for Week 12 matchups
3. **Schedule** Monday as planning day (not Sunday!)
4. **Generate** predictions Thursday
5. **Apply** tier system Friday
6. **Execute** Saturday with confidence

---

**Status**: ✅ READY FOR PROACTIVE PLANNING

All weekly plans generated. System ready to execute for remainder of 2025 season.

Generated: 2025-11-15
