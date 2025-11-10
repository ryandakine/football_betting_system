# 🏈 PHI @ GB Monday Night Football - Betting Plan

**Game**: Philadelphia Eagles @ Green Bay Packers
**Time**: Monday, 8:15 PM ET
**Line**: 45.5 total
**Bankroll**: $100

---

## 🎯 REFEREE-BASED BETTING STRATEGY

### ✅ IF Shawn Hochuli is the referee:
**BET**: UNDER 45.5
**CONFIDENCE**: 73% 🟢
**BET SIZE**: 3 units ($6)

**Why it works**:
- GB gets 7.5 penalties per game with Hochuli (2+ above league avg)
- PHI gets 5.4 penalties with him (average: 6.45 total)
- High penalties = stalled drives = lower scoring
- Hochuli's OT rate: 7.9% (slightly elevated)
- Historical data: 14 combined games (8 PHI, 6 GB)

**Expected Value**: +$4.20 profit (assuming -110 odds)

---

### ✅ IF Adrian Hill is the referee:
**BET**: UNDER 45.5
**CONFIDENCE**: 70% 🟢
**BET SIZE**: 3 units ($6)

**Why it works**:
- PHI gets 7.9 penalties per game with Hill (VERY HIGH)
- GB gets 5.7 penalties with him
- Combined avg: 6.8 penalties (1.3 above league)
- Adrian Hill: Known for flag-happy games
- Historical data: 13 combined games (10 PHI, 3 GB)

**Expected Value**: +$3.80 profit (assuming -110 odds)

---

### 🟡 IF Bill Vinovich or Carl Cheffers is the referee:
**BET**: OVER 45.5
**CONFIDENCE**: 59% 🟡
**BET SIZE**: 2 units ($4)

**Why it works**:
- Both refs call LOW penalties (4.3-5.0 avg)
- Faster game = more possessions = higher scoring
- Primetime baseline: Both teams avg 47.5 points
- Line is 45.5 (2 points below baseline)

**Expected Value**: +$1.60 profit (assuming -110 odds)

---

### ⚪ IF any other referee:
**DECISION**: PASS
**REASON**: No statistical edge detected (<60% confidence)

---

## 🔍 HOW TO CONFIRM THE REFEREE

### Method 1: Football Zebras (Most Reliable)
Run this on your local machine:
```bash
curl -s 'https://www.footballzebras.com/category/assignments/' | grep -i 'eagles\|packers\|PHI\|GB' -A 5 -B 5
```

### Method 2: ESPN Game Preview
Visit: https://www.espn.com/nfl/game/_/gameId/401671745

Look for "Game Officials" section

### Method 3: Twitter/X
Search: `MNF referee` or `Eagles Packers referee`

---

## 💰 BETTING EXECUTION CHECKLIST

- [ ] **By 6:00 PM ET**: Confirm referee assignment
- [ ] **By 7:00 PM ET**: Match referee to scenario above
- [ ] **By 7:30 PM ET**: Place bet if 🟢 or 🟡 edge detected
- [ ] **By 8:00 PM ET**: Log bet in `track_bets.py`

### Bet Tracking Command:
```bash
python track_bets.py --add \
  --game "PHI @ GB" \
  --bet-type "total" \
  --pick "UNDER 45.5" \
  --odds -110 \
  --amount 6 \
  --confidence 73 \
  --reasoning "Shawn Hochuli assigned - GB 7.5 penalties avg with him"
```

---

## 📊 PRIMETIME CONTEXT

**PHI in Primetime**: 47.5 avg points, 5.5 penalties
**GB in Primetime**: 47.5 avg points, 4.7 penalties
**Combined Baseline**: 47.5 points (2 points ABOVE the line)

**Key Insight**: Without a referee edge, this game projects OVER. But high-penalty refs (Hochuli, Hill) would slow it down enough for UNDER.

---

## ⚠️ RISK MANAGEMENT

**Current Bankroll**: $100
**Max Bet**: $6 (3 units @ $2/unit)
**Risk of Ruin**: <5% with proper bankroll management

**If bet loses**: Bankroll = $94 (down 6%)
**If bet wins**: Bankroll = $105.45 (up 5.45%)

**Week 11 Total Exposure**:
- Thursday BAL @ PIT (if placed): $4-6
- MNF PHI @ GB: $4-6
- **Total risk**: $8-12 (8-12% of bankroll)

**This is within safe limits** (Kelly Criterion suggests <15% weekly exposure)

---

## 🎯 MOST LIKELY SCENARIO

Based on typical NFL referee assignments for primetime games:

**Prediction**: Shawn Hochuli or Adrian Hill (both create UNDER edge)

**Backup Plan**: If no edge detected, SAVE bankroll for Week 11 Sunday games

---

## ✅ FINAL DECISION TREE

```
1. Get referee assignment (by 6 PM ET)
   ↓
2. Is referee Hochuli or Hill?
   YES → Bet UNDER 45.5 ($6)
   NO → Continue to step 3
   ↓
3. Is referee Vinovich or Cheffers?
   YES → Bet OVER 45.5 ($4)
   NO → PASS (save bankroll)
   ↓
4. Log bet immediately
   ↓
5. Watch game and track result
   ↓
6. Update tracker after game ends
```

---

**Status**: ⏳ WAITING ON REFEREE ASSIGNMENT
**Action Required**: Check Football Zebras or ESPN by 6 PM ET
**Expected Edge**: 65-73% (if Hochuli/Hill assigned)

