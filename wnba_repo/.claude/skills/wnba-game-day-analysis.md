---
name: wnba-game-day-analysis
description: |
  Complete WNBA game day analysis workflow optimized for small slates (3-6 games).
  Ensures thorough analysis of each game and proper timing-sensitive validation.

triggers:
  - analyze wnba
  - wnba betting
  - wnba today
  - professional women's basketball
---

# WNBA Game Day Analysis Workflow

**WHY THIS WORKFLOW EXISTS:**
- WNBA has SMALL SLATES (3-6 games, not 10-15)
- Each game deserves thorough individual analysis
- Star player injuries have MASSIVE impact (smaller rosters)
- Rest/travel matters more (packed 40-game schedule in 100 days)
- Agent must analyze with WNBA-specific context

**CRITICAL: Quality over Quantity approach**

---

## Phase 0: Pre-Analysis Context

**OBJECTIVE: Understand WNBA-specific factors before analysis**

### WNBA vs Other Sports:

| Factor | WNBA | NFL/NCAA | Impact |
|--------|------|----------|--------|
| Games/day | 3-6 | 10-15 | More time per game |
| Teams | 12 total | 32/350+ | Know all teams well |
| Schedule | 40 games/100 days | Spaced out | Fatigue factor |
| Star impact | ±8-12 pts | ±3-5 pts | Injuries critical |
| Market | Smaller | Larger | Less liquidity |

**Strategy Implications:**
- Analyze EVERY game (small slate)
- Check ALL injury reports (bigger impact)
- Monitor rest/travel closely (packed schedule)
- Conservative parlay limits (3 legs max)
- Higher variance = tighter risk management

---

## Phase 1: Season Context Validation

**OBJECTIVE: Apply correct adjustments for season phase**

### Step 1.1: Determine Season Phase

```python
from datetime import datetime

current_month = datetime.now().month
current_day = datetime.now().day

if current_month < 5 or current_month > 10:
    print("❌ OFF-SEASON: No games available")
    exit()

elif current_month == 5 and current_day < 15:
    season_phase = "EARLY_SEASON"
    confidence_adjustment = 0.03  # +3%
    notes = "Teams finding rhythm, higher variance"

elif 6 <= current_month <= 8:
    season_phase = "MID_SEASON"
    confidence_adjustment = 0.00
    notes = "Peak performance, best analysis period"

elif (current_month == 9 and current_day >= 15) or current_month == 10:
    season_phase = "PLAYOFFS"
    confidence_adjustment = 0.05  # +5%
    parlay_limit = 2  # Max 2 legs in playoffs
    notes = "Single elimination, higher variance, stars play more"

else:
    season_phase = "REGULAR_SEASON"
    confidence_adjustment = 0.00
    notes = "Standard analysis"
```

### Step 1.2: Apply Season-Specific Rules

**Early Season (May 1-15):**
- Require 69% confidence (vs 64% standard)
- Teams still adjusting
- Less historical data
- Potential line value

**Mid-Season (June-August):**
- Standard 64% confidence
- Best analysis period
- Most consistent performance
- Sharpest lines

**Playoffs (Late Sep-Oct):**
- Require 69% confidence
- Max 2-leg parlays only
- Stars playing 35+ minutes
- Home court amplified
- Single elimination pressure

---

## Phase 2: Critical WNBA Data Collection

**OBJECTIVE: Gather all timing-sensitive data**

### Step 2.1: Injury Reports (CRITICAL FOR WNBA)

**Why injuries matter MORE in WNBA:**
- 12-player rosters (vs 53 in NFL)
- Star-dependent teams
- Less depth
- Impact: ±8-12 points (vs ±3-5 in NFL)

**Priority Check List:**

For EACH game, check top 3 players per team:
1. Team's leading scorer
2. Primary playmaker/point guard
3. Top rebounder/defender

**Status Impact:**

| Status | Action | Adjustment |
|--------|--------|------------|
| OUT (star) | Reduce confidence 15% | -8 pts to team total |
| QUESTIONABLE (star) | Reduce confidence 10% | Wait for final status |
| OUT (role player) | Reduce confidence 5% | -2 pts adjustment |
| QUESTIONABLE (role) | Monitor | Small adjustment |

**Example:**
```python
# Check star player status
if player_status == "OUT" and player_tier == "STAR":
    confidence_reduction = 0.15
    team_adjustment = -8  # points
    print(f"⚠️ Star OUT: Reduce confidence 15%, adjust spread/total")

elif player_status == "QUESTIONABLE" and player_tier == "STAR":
    confidence_reduction = 0.10
    print(f"⚠️ Star QUESTIONABLE: Monitor until game time")
    # Set alert for final status
```

### Step 2.2: Rest & Travel Analysis (WNBA-Specific)

**Check for each team:**

1. **Back-to-back games:**
   ```python
   if is_back_to_back:
       fatigue_adjustment = -3  # points
       confidence_reduction = 0.05
       print("⚠️ Back-to-back: Team on 2nd game, fatigue factor")
   ```

2. **Travel distance:**
   ```python
   if travel_distance > 2000:  # miles
       travel_adjustment = -2  # points
       print("⚠️ Cross-country travel in last 48 hours")
   ```

3. **Games in last 5 days:**
   ```python
   games_in_5_days = count_recent_games(team, days=5)
   if games_in_5_days >= 3:
       fatigue_adjustment = -2
       print("⚠️ 3 games in 5 days: High fatigue")
   ```

**Combined Fatigue Impact:**
```python
# Fresh team vs tired team
team_a_rest_days = 3  # days since last game
team_b_rest_days = 0  # playing back-to-back

if team_a_rest_days >= 2 and team_b_rest_days == 0:
    advantage = 4  # points for rested team
    print("🎯 Rest mismatch: 4-point advantage to rested team")
```

### Step 2.3: Home/Away Splits (Magnified in WNBA)

**WNBA home court advantage: ~4.5 points (vs ~2.5 in NBA)**

```python
# Check home/away performance
home_team_home_record = get_home_record(home_team)
away_team_road_record = get_road_record(away_team)

# Some teams are MUCH better at home
if home_team_home_win_pct > 0.70 and away_team_road_win_pct < 0.40:
    print("🏠 Strong home team advantage")
    home_boost = 2  # extra points beyond typical HCA
```

---

## Phase 3: Comprehensive Analysis (Per Game)

**OBJECTIVE: Thorough analysis of EACH game (small slate)**

### Step 3.1: Statistical Analysis

For each game:

1. **Team Statistics:**
   - Offensive rating
   - Defensive rating
   - Pace (possessions per game)
   - 3-point shooting %

2. **Head-to-Head:**
   - Last 3 meetings
   - Average point differential
   - Trend direction

3. **Recent Form:**
   - Last 5 games ATS record
   - Last 10 games total (over/under)

### Step 3.2: Matchup-Specific Factors

```python
# Example: Star player matchup
if aces_vs_liberty:
    # Wilson vs Stewart matchup
    # Both MVP-caliber centers
    # Slight edge to home team
    matchup_note = "Elite center matchup, slight home edge"
```

### Step 3.3: Apply All Adjustments

```python
# Start with base analysis
base_spread = -3.5
base_total = 165.0

# Apply injury adjustments
if star_player_out:
    adjusted_spread = base_spread + 8  # Massive adjustment
    adjusted_total = base_total - 8

# Apply rest adjustments
if back_to_back:
    adjusted_spread -= 3
    adjusted_total -= 2

# Apply home court
adjusted_spread -= 4.5  # WNBA home court advantage

# Final adjusted numbers
final_spread = adjusted_spread
final_total = adjusted_total
```

---

## Phase 4: Edge Calculation with WNBA Thresholds

**OBJECTIVE: Calculate edge with sport-specific minimums**

### WNBA-Specific Thresholds:

```python
# Higher thresholds than college (sharper lines)
MIN_EDGE = 0.06  # 6% (vs 5% for WCBB)
MIN_CONFIDENCE = 0.64  # 64% (vs 58% for WCBB)
MAX_EXPOSURE = 0.08  # 8% of bankroll (conservative)

# Playoff adjustments
if season_phase == "PLAYOFFS":
    MIN_CONFIDENCE += 0.05  # 69% in playoffs
    MAX_PARLAY_LEGS = 2
else:
    MAX_PARLAY_LEGS = 3
```

### Edge Calculation:

```python
from wnba_ai_proof_config import get_config

config = get_config()
context = config.get_runtime_context()

# Calculate edge for each bet opportunity
for bet in potential_bets:
    edge = calculate_edge(bet, adjustments)
    confidence = calculate_confidence(bet, context)

    # Validate against thresholds
    is_valid, reason = config.validate_bet(
        stake=calculate_stake(edge, confidence, bankroll),
        edge=edge,
        confidence=confidence
    )

    if is_valid:
        qualified_bets.append(bet)
    else:
        print(f"❌ Rejected: {reason}")
```

---

## Phase 5: Parlay Construction (Conservative Approach)

**OBJECTIVE: Build parlays suited for small WNBA slates**

### Step 5.1: Filter Qualified Games

```python
qualified_games = [
    game for game in analyzed_games
    if game['edge'] >= 0.06  # 6% minimum
    and game['confidence'] >= 0.64  # 64% minimum
]

print(f"Qualified games: {len(qualified_games)} out of {len(analyzed_games)}")
```

### Step 5.2: Check for Correlations

**WNBA-specific correlations:**

Same day correlations (higher in small league):
```python
# Check if teams recently played each other
if game1['teams'] overlap with game2['teams'] (within 5 days):
    correlation_penalty = 0.25  # 25% penalty
    print("⚠️ Teams overlap - high correlation")
```

### Step 5.3: Build Parlays (Max 2-3 Legs)

```python
# 2-leg parlays (preferred)
for combo in combinations(qualified_games, 2):
    parlay = build_parlay(combo, correlation_penalty=0.12)
    if parlay['expected_value'] > 0:
        parlays.append(parlay)

# 3-leg parlays (only if 4+ qualified games)
if len(qualified_games) >= 4:
    for combo in combinations(qualified_games, 3):
        parlay = build_parlay(combo, correlation_penalty=0.18)
        if parlay['expected_value'] > 0:
            parlays.append(parlay)

# NO 4+ leg parlays for WNBA (too risky for small slate)
```

### Step 5.4: Conservative Kelly Sizing

```python
# More conservative than other sports
kelly_fraction = edge / (parlay_odds - 1)
safe_kelly = kelly_fraction * 0.20  # 20% of Kelly (vs 25% for NFL)

# Hard cap at 2.5% of bankroll
max_stake = bankroll * 0.025
stake = min(safe_kelly * bankroll, max_stake)
```

---

## Phase 6: Timing-Based Execution

**OBJECTIVE: Place bets at optimal times**

### Weekday Strategy (Tue-Fri):

**3:00 PM - 6:00 PM ET** (Optimal window):
- ✅ Injury reports finalized
- ✅ Time to analyze before 7 PM games
- ✅ Lines relatively stable

**After 7:00 PM**:
- ⚠️ Games started - too late for pre-game bets
- Option: Live betting if applicable

### Weekend Strategy (Sat-Sun):

**10:00 AM - 12:00 PM ET** (Optimal window):
- ✅ Before afternoon games start
- ✅ Full slate analysis possible
- ✅ Injury reports available

**After 1:00 PM**:
- ⚠️ Some games may have started
- Focus on evening games only

---

## Phase 7: Post-Analysis Tracking

### Step 7.1: Log All Recommendations

```python
for bet in final_recommendations:
    bet_tracker.track_bet({
        'game_id': bet['game_id'],
        'sport_type': 'wnba',
        'bet_type': bet['type'],
        'stake': bet['stake'],
        'edge': bet['edge'],
        'confidence': bet['confidence'],
        'season_phase': season_phase,
        'timing_window': timing_window,
        'adjustments_applied': bet['adjustments']
    })
```

### Step 7.2: Set Alerts

For games with:
- Edge ≥ 12%: Immediate alert
- Star player questionable: Alert on final status
- Line moving 3+ points: Alert

---

## Error Handling: WNBA-Specific

### IF star player status unknown:

```python
if star_player_status == "QUESTIONABLE":
    print("⚠️ CRITICAL: Star player status unknown")
    print("OPTIONS:")
    print("1. Wait for final status (check 2 hours before game)")
    print("2. Reduce stake by 50% and proceed")
    print("3. Skip this game entirely")
    print("")
    print("RECOMMENDATION: Wait if time permits")
```

### IF small slate (< 3 games):

```python
if len(games_today) < 3:
    print("⚠️ Very small slate today ({len} games)")
    print("")
    print("ADJUSTMENTS:")
    print("- Analyze ALL games thoroughly")
    print("- No parlays if < 2 qualified games")
    print("- Consider larger single-game bets")
    print("- Quality over quantity")
```

### IF playoff game:

```python
if season_phase == "PLAYOFFS":
    print("🏆 PLAYOFF GAME DETECTED")
    print("")
    print("AUTOMATIC ADJUSTMENTS:")
    print("- Confidence threshold: 64% → 69%")
    print("- Max parlay legs: 3 → 2")
    print("- Home court: +4.5 → +6.0 points")
    print("- Star minutes: +5 per game")
```

---

## Success Criteria

✅ Analysis at optimal time (afternoon before games)
✅ ALL injury reports checked (small slate = no excuse)
✅ Rest/travel factors incorporated
✅ Season phase adjustments applied
✅ Conservative parlay approach (max 3 legs)
✅ All bets validated against thresholds

---

## WNBA-Specific Rules (CANNOT BE OVERRIDDEN)

1. **Minimum edge**: 6% (sharper lines than college)
2. **Minimum confidence**: 64% (69% in playoffs)
3. **Max parlay legs**: 3 regular season, 2 playoffs
4. **Max exposure**: 8% of bankroll per game
5. **Max parlay stake**: 2.5% of bankroll
6. **Star player OUT**: Mandatory 15% confidence reduction
7. **Back-to-back game**: Mandatory 5% confidence reduction

---

## Agent Notes

- **WNBA is NOT NBA** - different dynamics, smaller market
- **Small slate = thorough analysis** - no excuse for missing details
- **Injuries matter MORE** - smaller rosters, star-dependent
- **Rest matters MORE** - packed 40-game schedule
- **Conservative approach** - less liquidity, higher variance
- **Quality over quantity** - 2 great bets > 5 mediocre bets

**If agent tries to skip injury check**: BLOCK (small slate, no excuse)
**If agent uses 5+ leg parlay**: BLOCK (too risky for WNBA)
**If agent ignores rest/travel**: Reduce all confidence by 10%

This workflow makes WNBA analysis structurally correct for small slates.
