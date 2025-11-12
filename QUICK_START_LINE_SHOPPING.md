# Quick Start: Strategic Line Shopping

## TL;DR - What You Need to Know

**Problem**: You can't fund 20 sportsbooks
**Solution**: Fund the right 3-5 books and capture 85-90% of max CLV
**Tools**: `strategic_line_shopping.py` + your existing workflow

---

## Step 1: Figure Out Your Budget

```bash
# See what you can do with your budget
python strategic_line_shopping.py --funding-plan YOUR_AMOUNT
```

### Budget Quick Reference

| Budget | Strategy | Accounts | Books | CLV Capture |
|--------|----------|----------|-------|-------------|
| **$30** | Bootstrap | 3 | DraftKings, FanDuel, BetMGM | ~47% |
| **$100** | Bootstrap+ | 3 | DK, FD, BetMGM (more capital each) | ~47% |
| **$125** | Recommended | 5 | + Caesars, BetRivers | ~85-90% |
| **$200** | Optimal | 7 | + PointsBet, bet365 | ~95% |

---

## Step 2: Open & Fund Your Accounts

### Most People Start Here (No Pinnacle/Circa Access):

1. **DraftKings** - $5 minimum
   - Sign up: [draftkings.com](https://www.draftkings.com)
   - Often has promos for new users

2. **FanDuel** - $10 minimum
   - Sign up: [fanduel.com](https://www.fanduel.com)
   - Major book, competitive lines

3. **BetMGM** - $10 minimum
   - Sign up: [betmgm.com](https://www.betmgm.com)
   - Often has best lines on NFL

**Total: $25 minimum to start**

### If You Can Access Sharp Books:

4. **Pinnacle** - $50 minimum (if available in your state)
   - Gold standard for line quality

5. **Circa Sports** - $50 minimum (Nevada, limited states)
   - Sharp Vegas book

---

## Step 3: Integrate Into Your Workflow

Your existing `auto_execute_bets.py` already uses line shopping!

```bash
# Run your normal workflow - it will shop lines automatically
python auto_execute_bets.py --auto
```

The system will:
1. Get DeepSeek-R1 pick
2. Apply contrarian filter
3. **Shop lines across all major books** ✅
4. Show you which book has the best line
5. Tell you where to place the bet

---

## Step 4: Manual Line Shopping (Optional)

For specific games:

```bash
# Check specific game
python line_shopper.py --game "PHI @ GB" --team GB

# See all recommendations
python strategic_line_shopping.py --recommend-accounts
```

---

## Real Example: $100 Budget

**Your situation:**
- $100 available
- Want to bet NFL
- Live in a state with DK/FD/BetMGM

**Recommended funding:**
```
DraftKings:  $30
FanDuel:     $35
BetMGM:      $35
────────────────
TOTAL:      $100
```

**Expected CLV**: ~47% of max (you'll get the best line 47% of the time)

**What this means:**
- Without line shopping: Average odds -110
- With line shopping: Average odds ~-107
- **Impact**: +2-3% ROI improvement

**Real dollars:**
- 50 bets @ $10 = $500 risked
- Without: $500 × 5% ROI = $25 profit
- With: $500 × 8% ROI = $40 profit
- **GAIN: +$15 per season** just from line shopping

---

## Pro Tips

### 1. **Don't Be Loyal to One Book**
Sharp bettors use whichever book has the best line. That's it.

### 2. **Keep Minimums in Each Account**
Don't lock up all your capital in one book. Spread it across your 3-5 accounts.

### 3. **Consolidate Winnings Weekly**
Withdraw from winning accounts, reload accounts that need funds.

### 4. **Check Lines BEFORE Placing Bet**
Lines change throughout the week. Check right before you bet.

### 5. **Track Your CLV**
Your existing `auto_execute_bets.py` does this automatically!

---

## Common Questions

**Q: Can I start with just DraftKings?**
A: Yes, but you're leaving 2-3% ROI on the table. Even adding FanDuel helps significantly.

**Q: What if I only have $20?**
A: Fund DraftKings ($5) and FanDuel ($10), keep $5 for a third book later.

**Q: Do I need to check lines every time?**
A: `auto_execute_bets.py` does this automatically. Manual checks optional.

**Q: How much better are sharp books (Pinnacle/Circa)?**
A: They have the best line 35% and 25% of the time respectively. Huge edge if you can access them.

**Q: Can I use promos?**
A: YES! Books often have "deposit match" or "bet insurance" promos. Stack these with line shopping for maximum value.

---

## Next Steps

1. **Choose your budget** ($30, $100, $200)
2. **Run**: `python strategic_line_shopping.py --funding-plan YOUR_BUDGET`
3. **Open the recommended accounts**
4. **Fund them according to the plan**
5. **Start betting** with `auto_execute_bets.py --auto`

---

## Questions?

Read the full guide: `LINE_SHOPPING_STRATEGY_GUIDE.md`

**Bottom line**: You don't need 20 accounts. You need the RIGHT 3-5 accounts. Start small, scale up as bankroll grows.

🎯
