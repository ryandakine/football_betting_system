# 🏈 Weekend NFL Predictions - Quick Start

## Get Picks for Sunday + Monday Night Football

### Option 1: One Command (Easiest)

```bash
./get_weekend_picks.sh
```

### Option 2: Python Script

```bash
python3 get_weekend_predictions.py
```

Both will:
- ✅ Call your AWS Lambda with GGUF models
- ✅ Analyze ALL weekend games (Sunday 1pm, 4pm, 8pm + Monday Night)
- ✅ Show high-confidence picks organized by time slot
- ✅ Save results to JSON

---

## Setup (First Time Only)

### Configure AWS Credentials

```bash
aws configure
```

Enter:
- AWS Access Key ID
- AWS Secret Access Key
- Region: `us-east-1`
- Output format: `json`

---

## When to Run

**Best time:** Saturday evening or Sunday morning before 1pm ET

This gives you:
- All Sunday games (early, late, night)
- Monday Night Football
- Time to place bets before kickoff

---

## Example Output

```
🏈 NFL WEEKEND PREDICTIONS
======================================================================

📅 SUNDAY EARLY (1:00 PM ET)
----------------------------------------------------------------------

1. Chiefs @ Bills
   🎯 Pick: Bills -3
   📊 Confidence: 78%
   📈 Spread: Bills -3 (-110)
   💎 Edge: 5.2%
   💡 Bills defense at home is dominant...

2. Cowboys @ Eagles
   🎯 Pick: Cowboys +4.5
   📊 Confidence: 72%
   📈 Spread: Cowboys +4.5 (-115)
   💎 Edge: 3.8%
   💡 Division game will be close...

📅 SUNDAY NIGHT FOOTBALL (8:20 PM ET)
----------------------------------------------------------------------

1. 49ers @ Packers
   🎯 Pick: Under 45.5
   📊 Confidence: 75%
   🎲 Total: 45.5
   💎 Edge: 4.1%
   💡 Cold weather game favors defense...

📅 MONDAY NIGHT FOOTBALL (8:15 PM ET)
----------------------------------------------------------------------

1. Dolphins @ Rams
   🎯 Pick: Rams -2.5
   📊 Confidence: 76%
   📈 Spread: Rams -2.5 (-108)
   💎 Edge: 4.5%
   💡 Rams at home after bye week...

======================================================================
Total Games: 14
High Confidence Picks: 8
======================================================================
```

---

## After You Get Picks

1. **Review Confidence Levels**
   - 75%+ = Strong bets (2-3 units)
   - 70-75% = Good bets (1-2 units)
   - 65-70% = Consider bets (0.5-1 unit)

2. **Check Edge**
   - 5%+ edge = Excellent value
   - 3-5% edge = Good value
   - 2-3% edge = Playable

3. **Organize Your Bets**
   - Early games (1pm ET)
   - Late games (4pm ET)
   - SNF (8:20pm ET)
   - MNF (8:15pm ET Monday)

4. **Place Bets**
   - Before 1pm ET Sunday
   - Shop lines at multiple books
   - Use recommended unit sizes

---

## Files Created

After running, check:
- `data/weekend_picks_YYYYMMDD.json` - Full results
- `data/lambda_predictions_*.json` - Raw Lambda response

---

## Troubleshooting

### "AWS credentials not configured"
```bash
aws configure
# Enter your credentials
```

### "Lambda function not found"
Check the function name in `call_lambda_predictions.py` and `get_weekend_predictions.py`

Default is: `nfl-live-predictions`

If yours is different, update the scripts.

### "No high-confidence picks"
Lambda returned results but no picks meet the confidence threshold.

Check `all_predictions` in the JSON output for lower confidence plays.

---

## Tips for Success

✅ **Run Saturday night** - Get picks early, shop for best lines
✅ **Check weather** - Cold/wind/rain affects totals
✅ **Follow unit sizing** - Don't overbet lower confidence picks
✅ **Track results** - Build a record over weeks
✅ **Line shop** - Compare odds across sportsbooks

---

## Quick Reference

**Saturday Night:** Run script, get all picks
**Sunday Morning:** Review picks, place early game bets
**Sunday 1pm:** Games start, enjoy!
**Sunday 4pm:** Late games
**Sunday 8:20pm:** SNF
**Monday 8:15pm:** MNF

---

Good luck this weekend! 🍀🏈
