---
name: place-ncaa-bet
description: NCAA bet placement with automatic validation and tracking. Triggers on "place bet", "bet on", "make wager". Ensures every bet is logged correctly and validated before execution.
---

# NCAA Bet Placement Workflow

## PRINCIPLE: Error = System Gap (Not Agent Fault)
If bet placement fails, patch the system to make failure impossible.

## WHY THIS SKILL EXISTS

**Problem:** Without workflow:
- Agent forgets to log bets
- Bet sizing miscalculated
- No validation before execution
- Tracking inconsistent

**Solution:** Workflow enforces:
- Automatic validation (hook catches errors)
- Consistent logging format
- Bet tracking for ROI analysis
- Bankroll updates

## WORKFLOW (Execute in Order)

### Step 1: Validate Bet Parameters

**Required inputs from user:**
- Game (e.g., "Toledo @ Bowling Green")
- Side (e.g., "Toledo -3.0")
- Confidence (e.g., 76%)
- Stake (e.g., $250)
- Edge (e.g., 5%)

**Automatic validation (hook enforces):**
```bash
# Hook runs automatically - these checks happen before execution
✓ Confidence ≥ 70%
✓ Edge ≥ 3%
✓ Stake $20-$500
✓ Bankroll sufficient
```

**If validation fails:**
- Hook blocks execution with reason
- Display: "❌ BLOCKED: [reason]"
- DO NOT proceed until parameters adjusted

### Step 2: Calculate Expected Value

```python
# Expected Value calculation
implied_prob = confidence / 100
win_amount = stake * (100/110)  # -110 odds typical
lose_amount = stake

ev = (implied_prob * win_amount) - ((1 - implied_prob) * lose_amount)
ev_percent = (ev / stake) * 100
```

**Example:**
- Stake: $250
- Confidence: 76%
- Win amount: $227.27
- EV: (0.76 × $227) - (0.24 × $250) = **+$112.73**
- EV%: +45.1%

**Threshold:**
- EV% must be positive
- Higher is better (10%+ is excellent)

### Step 3: Log Bet to Tracking System

**Create bet record:**
```json
{
  "bet_id": "NCAA_2025_001",
  "timestamp": "2025-01-13T18:30:00Z",
  "game": "Toledo @ Bowling Green",
  "side": "Toledo -3.0",
  "confidence": 0.76,
  "edge": 0.05,
  "stake": 250.00,
  "expected_value": 112.73,
  "odds": -110,
  "sportsbook": "DraftKings",
  "status": "pending",
  "models_used": ["xgboost_super", "neural_net_deep", "alt_spread", ...],
  "model_agreement": 0.92,
  "bankroll_before": 10000.00,
  "day_of_week": "Tuesday",
  "is_maction": true
}
```

**Save to:**
```bash
echo "$BET_JSON" >> data/bets/ncaa_bets_2025.jsonl
```

**One bet per line (JSONL format)** for easy parsing.

### Step 4: Update Bankroll State

**Deduct stake from available bankroll:**
```python
# Before bet
available_bankroll = 10000.00

# After bet placed
pending_stake = 250.00
available_bankroll = 10000.00 - 250.00 = 9750.00

# Save state
with open('data/bankroll_state.json', 'w') as f:
    json.dump({
        'total': 10000.00,
        'available': 9750.00,
        'pending': 250.00,
        'profit_loss': 0.00,
        'last_updated': '2025-01-13T18:30:00Z'
    }, f)
```

**Why this matters:**
- Prevents over-betting (can't bet with locked funds)
- Tracks true available capital
- Accurate for next bet sizing

### Step 5: Generate Bet Confirmation

**Display to user:**
```
✅ BET PLACED

Game: Toledo @ Bowling Green
Side: Toledo -3.0
Stake: $250.00
Confidence: 76%
Edge: 5.0%
Expected Value: +$112.73 (45.1%)

TRACKING:
Bet ID: NCAA_2025_001
Logged: data/bets/ncaa_bets_2025.jsonl

BANKROLL:
Previous: $10,000.00
Available: $9,750.00
Pending: $250.00

Next steps:
- Game kicks off: Tuesday 7:00 PM ET
- Result tracking: Run after game completes
- Expected return: $477.27 if win
```

### Step 6: Set Result Tracking Reminder

**Create reminder file:**
```bash
echo "NCAA_2025_001|Toledo @ Bowling Green|2025-01-14T00:00:00Z" >> data/pending_results.txt
```

**Why:**
- Agent can check pending results next session
- Don't forget to close out bets
- Accurate P&L tracking

## POST-GAME WORKFLOW

**After game completes (next day):**

### Step 1: Fetch Game Result
```bash
python fetch_ncaa_results.py NCAA_2025_001
```

### Step 2: Determine Bet Outcome
```python
# Example: Toledo won 31-24 (by 7)
actual_spread = 7
bet_spread = 3.0
side = "Toledo"

if side == "Toledo":
    bet_won = actual_spread > bet_spread  # 7 > 3.0 = True
else:
    bet_won = actual_spread < bet_spread
```

### Step 3: Update Bet Record
```json
{
  "bet_id": "NCAA_2025_001",
  "status": "won",  // or "lost" or "push"
  "actual_spread": 7.0,
  "profit_loss": 227.27,  // +227.27 for win
  "settled_at": "2025-01-14T23:30:00Z"
}
```

### Step 4: Update Bankroll
```python
# Win: Return stake + profit
bankroll += (stake + profit)

# Loss: Already deducted, no change
# Push: Return stake only
```

### Step 5: Calculate Running Stats
```python
# Load all bets
total_bets = 10
total_wins = 6
win_rate = 6/10 = 60%

total_staked = 2500
total_returned = 2750
profit = 250
roi = 250/2500 = 10%
```

## ERROR PREVENTION

**Hook prevents these mistakes:**
- ❌ Betting without sufficient confidence
- ❌ Betting without sufficient edge
- ❌ Over-betting (exceeds $500)
- ❌ Under-betting (less than $20)
- ❌ Betting with insufficient bankroll

**Skill prevents these mistakes:**
- ❌ Forgetting to log bet
- ❌ Miscalculating EV
- ❌ Not updating bankroll
- ❌ Losing track of pending bets

**Result:** Agent structurally cannot make these errors.

## SELF-DOCUMENTING BET LOG

Every bet includes full context:

```jsonl
{"bet_id":"NCAA_2025_001","game":"Toledo @ BG","confidence":0.76,"stake":250,"ev":112.73,"won":true,"profit":227.27}
{"bet_id":"NCAA_2025_002","game":"Miami @ Ohio","confidence":0.71,"stake":150,"ev":45.50,"won":false,"profit":-150}
```

**Analysis becomes trivial:**
```bash
# Win rate
cat ncaa_bets_2025.jsonl | jq -s 'map(select(.status=="won")) | length'

# Total P&L
cat ncaa_bets_2025.jsonl | jq -s 'map(.profit_loss) | add'

# ROI
cat ncaa_bets_2025.jsonl | jq -s 'map(.profit_loss) | add as $profit | map(.stake) | add as $staked | ($profit / $staked * 100)'
```

## PERSISTENCE BENEFITS

**Session 1:**
- Create bet placement workflow
- First bet logged correctly

**Session 2 (next week):**
- Agent invokes skill automatically
- Exact same workflow
- No re-teaching

**Session 100:**
- Still exact same workflow
- Zero degradation
- All bets logged consistently

**Investment:** 1 hour to create
**Operational cost:** 0 hours forever
**Mistake prevention:** ∞ errors caught

## INTEGRATION WITH HOOKS

```
User: "Bet $250 on Toledo -3"
  ↓
Hook: Validate parameters automatically
  ↓ (if valid)
Skill: Execute bet placement workflow
  ↓
Hook: Validate execution
  ↓
Result: Bet logged, bankroll updated, confirmation shown
```

**Every step validated. Every step logged. Zero gaps.**

## SUCCESS METRICS

**Quality metrics:**
- 100% bet logging (no missed bets)
- 100% bankroll tracking accuracy
- 0 validation errors reach execution

**Performance metrics:**
- Win rate ≥ 52% (breakeven)
- Win rate ≥ 58% (elite, your target)
- ROI ≥ 5% (profitable)
- ROI ≥ 10% (excellent)

**System metrics:**
- 0 hook overrides (agent never fights system)
- 0 manual corrections needed
- 100% workflow adherence
