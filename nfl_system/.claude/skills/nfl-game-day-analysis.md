---
name: nfl-game-day-analysis
description: |
  Complete NFL game day analysis workflow with timing-sensitive validation.
  Ensures agent analyzes at optimal times and doesn't miss critical context.

triggers:
  - analyze nfl
  - nfl betting
  - sunday analysis
  - thursday night football
  - monday night football
---

# NFL Game Day Analysis Workflow

**WHY THIS WORKFLOW EXISTS:**
- NFL games have STRICT timing windows
- Missing injury reports = invalid analysis
- Weather changes rapidly for outdoor games
- Line movements are timing-sensitive
- Agent must analyze at OPTIMAL times, not arbitrary times

**CRITICAL: This workflow is TIME-AWARE**

---

## Phase 0: Pre-Flight Timing Validation

**OBJECTIVE: Ensure we're analyzing at the right time**

### Step 1: Verify Game Day and Time

```bash
# This runs automatically via hook
# Agent SEES the output and adjusts accordingly
```

**Decision Tree:**

IF Sunday AND time >= 1:00 PM ET:
- ❌ ABORT: Games already started
- Agent response: "Too late - games in progress"

IF Sunday AND time between 9:00 AM - 12:45 PM ET:
- ✅ OPTIMAL: Proceed with full analysis
- Check inactive lists (available 11:30 AM)

IF Sunday AND time < 9:00 AM ET:
- ⚠️  EARLY: Proceed but flag for re-run
- Injury reports not finalized
- Weather may change

IF Thursday AND time between 6:00 PM - 8:00 PM ET:
- ✅ OPTIMAL: TNF analysis window

IF Monday AND time between 5:00 PM - 8:00 PM ET:
- ✅ OPTIMAL: MNF analysis window

IF Tuesday/Wednesday:
- ℹ️  NOT GAME DAY: Line shopping mode
- Focus on early line values
- Less urgency on injury reports

### Step 2: Season Context

**Regular Season** (Sep-Dec):
- Standard analysis approach
- Full slate of games
- Conference implications matter

**Playoffs** (Jan-Feb):
- Tighter lines (sharper market)
- Higher variance (single elimination)
- Increase confidence thresholds by 3%
- Reduce parlay legs (max 2)

**Off-Season** (Apr-Aug):
- ❌ BLOCK analysis entirely
- No games available
- Wait until September

---

## Phase 1: Critical Information Gathering

**OBJECTIVE: Collect time-sensitive data BEFORE analysis**

### Step 1.1: Injury Report Status

**Sunday Morning Priority:**

```python
# Check if inactive lists are available
current_time = datetime.now()
if current_time.weekday() == 6:  # Sunday
    if current_time.hour >= 11 and current_time.minute >= 30:
        print("✅ Inactive lists available")
        # Proceed with full analysis
    else:
        print("⚠️ Inactive lists not yet published")
        print("   Published: 11:30 AM ET (90 min before kickoff)")
        # Proceed but flag for re-analysis
```

**Key Injury Impact Analysis:**

For each game:
1. Check both teams' injury reports
2. Focus on:
   - Starting QB (±7 points impact)
   - Star RB (±3 points impact)
   - Top WR (±2 points impact)
   - Key OL (affects total)
   - Defensive stars (affects total/spread)

**CRITICAL RULE:**
If star player (QB/RB1/WR1) is QUESTIONABLE:
- Reduce confidence by 10%
- Reduce stake by 50%
- Set alert for final status

### Step 1.2: Weather Data Collection

**For outdoor stadiums ONLY:**

Priority stadiums (high weather impact):
1. **Buffalo**: Snow/wind (check hourly)
2. **Green Bay**: Cold (<20°F affects ball)
3. **Chicago**: Wind (off Lake Michigan)
4. **Cleveland**: Lake effect snow
5. **Kansas City**: Cold/wind
6. **New England**: Cold/snow
7. **Denver**: Altitude + wind

**Weather Impact Thresholds:**

Wind:
- <10 MPH: No adjustment
- 10-15 MPH: -0.5 points to total
- 15-20 MPH: -1.5 points to total, favor running teams
- >20 MPH: -3 points to total, heavily favor defense

Precipitation:
- Light rain: -0.5 points to total
- Heavy rain: -2 points to total
- Snow: -3 points to total, favor running teams

Temperature:
- >40°F: No adjustment
- 20-40°F: -0.5 points to total
- <20°F: -1.5 points to total
- <0°F: -3 points to total, extreme unpredictability

### Step 1.3: Line Movement Analysis

**Check line movement from open to current:**

```python
# Track line movement
opening_line = get_opening_line(game_id)
current_line = get_current_line(game_id)
movement = current_line - opening_line

# Interpret movement
if abs(movement) >= 1.5:
    print(f"⚠️ Significant line movement: {movement:+.1f}")

    # Check which direction
    if movement > 0:  # Line moved toward favorite
        print("   Sharp money may be on favorite")
    else:
        print("   Sharp money may be on underdog")
```

**Sharp vs Public Detection:**

```python
# Compare line movement to betting percentages
public_percentage = get_public_betting(game_id)
line_movement = calculate_movement(game_id)

if public_percentage > 70 and line_movement < 0:
    print("🎯 SHARP OPPORTUNITY: Public on favorite, line moving to dog")
    # This is a fade-the-public opportunity

if public_percentage < 30 and line_movement > 0:
    print("🎯 SHARP OPPORTUNITY: Public on dog, line moving to favorite")
```

---

## Phase 2: AI-Powered Analysis with Context

**OBJECTIVE: Run analysis WITH all critical context**

### Step 2.1: Load Context into Analysis

```python
from nfl_ai_proof_config import get_config

config = get_config()
context = config.get_runtime_context()

# Context now includes:
# - Bankroll limits
# - Edge thresholds
# - Season adjustments
# - Timing validation
```

### Step 2.2: Run Multi-Layer Analysis

For each game:

1. **Base Statistical Analysis**
   - Team stats (offense/defense rankings)
   - Historical matchups
   - Home/away splits

2. **Injury Adjustment**
   - Apply injury impacts from Phase 1
   - Adjust point spreads
   - Adjust totals

3. **Weather Adjustment**
   - Apply weather impacts from Phase 1
   - Adjust totals
   - Adjust team strengths (pass vs run)

4. **Line Movement Insight**
   - Factor in sharp money detection
   - Adjust confidence based on movement

5. **AI Council Consensus**
   - 5 specialized agents analyze
   - Generate consensus recommendation
   - Calculate combined confidence

### Step 2.3: Validate Recommendations

```python
# For each recommended bet
for bet in recommendations:
    is_valid, reason = config.validate_bet(
        stake=bet['stake'],
        edge=bet['edge'],
        confidence=bet['confidence']
    )

    if not is_valid:
        print(f"❌ Bet rejected: {reason}")
        # Move to next bet
    else:
        print(f"✅ Bet validated: {bet['game']}")
        # Add to final recommendations
```

---

## Phase 3: Parlay Optimization with Correlation

**OBJECTIVE: Build parlays that account for NFL-specific correlations**

### Step 3.1: Filter Qualified Games

```python
qualified_games = [
    game for game in analyzed_games
    if game['edge'] >= 0.06  # NFL: 6% minimum (sharper lines)
    and game['confidence'] >= 0.62  # NFL: 62% minimum
]
```

### Step 3.2: Check Game Correlations

**CRITICAL: NFL games can be correlated**

Same-game correlations (avoid):
- Team total OVER + Team spread (correlated)
- Game total OVER + Underdog spread (correlated)

Slate-wide correlations (caution):
- Multiple road underdogs (slightly correlated)
- Multiple overs in bad weather (correlated)
- Division rivals in same week (correlated)

**Apply correlation penalties:**

```python
# 2-leg parlay
if games_are_correlated(game1, game2):
    correlation_penalty = 0.20  # 20% penalty
else:
    correlation_penalty = 0.10  # Standard 10%

combined_edge = (game1_edge + game2_edge) * (1 - correlation_penalty)
```

### Step 3.3: Kelly Sizing for Parlays

```python
# Parlay odds calculation
parlay_odds = 1.0
for game in parlay_legs:
    parlay_odds *= game['decimal_odds']

# Kelly criterion with safety factor
edge_fraction = combined_edge
kelly_fraction = edge_fraction / (parlay_odds - 1)
safe_kelly = kelly_fraction * 0.20  # 20% of Kelly for parlays

# Cap at 2% of bankroll for NFL parlays
max_stake = bankroll * 0.02
stake = min(safe_kelly * bankroll, max_stake)
```

---

## Phase 4: Timing-Based Execution Strategy

**OBJECTIVE: Place bets at optimal times**

### Sunday Strategy:

**11:30 AM - 12:00 PM ET** (After inactive lists):
- ✅ Place bets on early slate (1 PM games)
- ✅ All injury info incorporated
- ✅ Weather finalized
- ✅ Lines relatively stable

**3:00 PM - 3:45 PM ET** (Between slates):
- ✅ Place bets on late slate (4:25 PM games)
- ⚠️  Early slate already in progress
- ✅ Can adjust based on early results (if not in parlays)

**6:00 PM - 7:45 PM ET** (Before SNF):
- ✅ Place SNF bets
- ✅ Can see full day's results
- ⚠️  Lines may have moved significantly

### Thursday Strategy:

**6:00 PM - 8:00 PM ET** (Before TNF):
- ✅ Final injury reports available
- ✅ Optimal betting window
- ⚠️  After 8:15 PM: Game started (too late)

### Monday Strategy:

**5:00 PM - 8:00 PM ET** (Before MNF):
- ✅ Full weekend context available
- ✅ Final injury reports
- ✅ Weather finalized

---

## Phase 5: Post-Analysis Actions

### Step 5.1: Set Alerts

For games with:
- Edge ≥ 12% (holy grail): Immediate alert
- Questionable star player: Alert on final status
- Weather forecast changing: Alert on updates
- Line moving 2+ points: Alert on movement

### Step 5.2: Schedule Re-Analysis

IF current time is "EARLY" window:
- Schedule re-analysis for optimal window
- Example: Sunday 7 AM analysis → re-run at 11 AM

### Step 5.3: Track Performance

```python
# Log each bet for post-game analysis
bet_tracker.track_bet({
    'game_id': game_id,
    'sport_type': 'nfl',
    'bet_type': bet_type,
    'stake': stake,
    'edge': edge,
    'confidence': confidence,
    'analysis_time': datetime.now(),
    'timing_window': timing_window,  # OPTIMAL/GOOD/EARLY
})
```

---

## Error Handling: Time-Based Failures

### IF analyzed too late (Sunday after 1 PM):

```python
print("❌ ANALYSIS ABORTED: Games already started")
print("")
print("NEXT STEPS:")
print("1. Wait for next week")
print("2. Set calendar reminder for Sunday 10 AM")
print("3. Or: Enable automated Sunday morning analysis")
print("")
print("LESSON: NFL analysis must complete by 12:45 PM ET Sunday")
```

### IF missing injury reports:

```python
print("⚠️ WARNING: Inactive lists not available yet")
print("   Current time: [time]")
print("   Lists publish: 11:30 AM ET")
print("")
print("OPTIONS:")
print("1. Proceed with preliminary analysis (reduce confidence 10%)")
print("2. Wait for inactive lists, re-run analysis")
print("3. Skip games with questionable star players")
```

### IF weather forecast uncertain:

```python
print("⚠️ WARNING: Weather forecast uncertain for [stadium]")
print("   Current forecast: [conditions]")
print("   Game time: [time]")
print("")
print("RECOMMENDATION:")
print("1. Reduce total bet sizing by 50%")
print("2. Check forecast 2 hours before kickoff")
print("3. Be prepared to adjust or cancel bet")
```

---

## Success Criteria

✅ Analysis run at optimal time (GOOD or OPTIMAL window)
✅ All injury reports incorporated
✅ Weather data collected for outdoor games
✅ Line movements analyzed
✅ All bets validated against constraints
✅ Parlays account for correlations
✅ Performance tracking enabled

---

## Time-Based Rules (CANNOT BE OVERRIDDEN)

1. **Sunday early slate**: Analysis complete by 12:45 PM ET
2. **Inactive lists**: Must wait until 11:30 AM if star player questionable
3. **Weather**: Must check within 2 hours of outdoor game kickoff
4. **Playoff games**: Increase confidence threshold to 65%
5. **Parlay stake**: Maximum 2% of bankroll (NFL is high variance)
6. **Re-analysis**: If analyzed in EARLY window, MUST re-run before kickoff

---

## Agent Notes

- **This workflow is TIME-CRITICAL** - timing determines success
- **DO NOT skip timing validation** - it's structural, not optional
- **Injury reports are MANDATORY** - especially for Sunday analysis
- **Weather matters for outdoor games** - indoor games can skip
- **Line movement is a SIGNAL** - incorporate into confidence
- **NFL lines are SHARP** - require higher edges than college (6% vs 5%)

**If agent attempts to analyze at wrong time**: Hook will BLOCK execution
**If agent skips injury reports**: Confidence must decrease 10%
**If agent ignores weather**: Outdoor game bets must be reduced 50%

This workflow makes timing-sensitive NFL analysis structurally correct.
