# NFL Parlay Maker 🎯

**Smart parlay builder for NFL betting edges**

Automatically generates optimized parlays from betting edges detected by the referee intelligence system.

---

## Features

✅ **2-10 Leg Parlays** - Build parlays of any size
✅ **Smart Selection** - Avoids correlated bets (same game)
✅ **EV Optimization** - Ranks by expected value
✅ **Confidence Filtering** - Only use high-quality edges
✅ **Diversification** - Spreads bets across multiple games
✅ **Multiple Bet Types** - Spread, Total, ML, 1H, Team Totals

---

## Quick Start

### Basic Usage
```bash
# Generate parlays for Week 10
python nfl_parlay_maker.py --week 10

# Show top 20 parlays
python nfl_parlay_maker.py --week 10 --top 20

# Only 3-leg parlays
python nfl_parlay_maker.py --week 10 --min-legs 3 --max-legs 3
```

### Advanced Filtering
```bash
# High-confidence only (70%+)
python nfl_parlay_maker.py --week 10 --min-confidence 0.70

# MASSIVE edges only
python nfl_parlay_maker.py --week 10 --edge-sizes MASSIVE

# MASSIVE and LARGE edges
python nfl_parlay_maker.py --week 10 --edge-sizes MASSIVE LARGE

# Sort by confidence instead of EV
python nfl_parlay_maker.py --week 10 --sort-by confidence
```

### Save to File
```bash
python nfl_parlay_maker.py --week 10 --output reports/week10_parlays.txt
```

---

## How It Works

### 1. Edge Collection
Fetches betting edges from `auto_weekly_analyzer.py`:
- Referee intelligence patterns
- 640+ team-referee biases
- All bet types (Spread/Total/ML/1H/Team Totals)

### 2. Smart Filtering
Filters edges by:
- **Confidence**: Minimum probability threshold
- **Edge Size**: MASSIVE/LARGE/MEDIUM
- **Bet Type**: Choose specific markets

### 3. Parlay Generation
Generates all combinations:
- Avoids same-game correlations (by default)
- Diversifies across multiple games
- Builds 2-10 leg parlays

### 4. Ranking & Optimization
Ranks parlays by:
- **Expected Value (EV)** - Default, best for long-term profit
- **Confidence** - Highest probability of hitting
- **Payout** - Maximum potential return

---

## Example Output

```
🎯 NFL PARLAY MAKER - WEEK 10
================================================================================
Generated: 2025-11-10 15:30:00
Total Parlays Generated: 143

🏆 TOP 10 PARLAYS (Ranked by Expected Value)

1. 3-Leg Parlay (+595, 42% confidence)
   Expected Value: $1.92 per $1 wagered
   $100 bet → Win $595.00 (42.0% probability)

   Leg 1: KC -2.5
          SPREAD | 80% confidence | MASSIVE
          💡 Brad Rogers + KC = +14.6 margin bias

   Leg 2: OVER 42.0
          TOTAL | 75% confidence | LARGE
          💡 Carl Cheffers 8.6% OT rate, high-scoring games

   Leg 3: GB -3.0
          SPREAD | 70% confidence | MEDIUM
          💡 John Parry home favoritism pattern

--------------------------------------------------------------------------------

2. 4-Leg Parlay (+1200, 29% confidence)
   Expected Value: $2.48 per $1 wagered
   ...
```

---

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--week N` | NFL week number (required) | - |
| `--min-confidence X` | Minimum edge confidence (0.0-1.0) | 0.60 |
| `--min-legs N` | Minimum parlay legs | 2 |
| `--max-legs N` | Maximum parlay legs | 5 |
| `--top N` | Show top N parlays | 10 |
| `--sort-by {ev,confidence,payout}` | Ranking method | ev |
| `--edge-sizes MASSIVE LARGE` | Filter by edge sizes | all |
| `--allow-same-game` | Allow correlated bets | false |
| `--output FILE` | Save report to file | - |

---

## Understanding the Output

### Expected Value (EV)
```
Expected Value: $1.92 per $1 wagered
```
For every $1 you bet, you expect to profit $1.92 on average (long-term).

### Confidence vs Odds
```
42% confidence, +595 odds
```
- **42% confidence** = Our model's probability this parlay hits
- **+595 odds** = Sportsbook payout ($100 wins $595)
- **EV is positive** when our probability > implied probability from odds

### Parlay Math
3-leg parlay with -110 odds on each leg:
- Single leg: 1.909 decimal odds
- 3 legs: 1.909 × 1.909 × 1.909 = 6.95 decimal = +595 American
- All 3 must hit!

---

## Best Practices

### ✅ DO:
- Use **60%+ confidence** edges for parlays
- Diversify across **multiple games**
- Focus on **positive EV** parlays
- Keep parlays to **2-4 legs** for better hit rate
- Use **MASSIVE/LARGE** edges when available

### ❌ DON'T:
- Include bets from the **same game** (correlated)
- Build **10-leg parlays** (low probability)
- Chase **high payouts** over **positive EV**
- Bet more than you can **afford to lose**
- Ignore **edge quality** for parlay size

---

## Integration with Autonomous Agent

The parlay maker integrates with the autonomous betting agent:

```python
# In autonomous_betting_agent.py
# After getting game edges, generate parlays
parlay_result = subprocess.run(
    ['python', 'nfl_parlay_maker.py', '--week', '10', '--json'],
    capture_output=True
)
```

---

## Tips for Maximum Value

### 1. Wait for Referee Assignments
Referee assignments post **Thursday** before games. Run the system then for best edges.

### 2. Compare Against Market
If the parlay maker suggests KC -2.5 at 80% confidence, but the market moved to KC -3.5, the edge may have disappeared.

### 3. Bet Sizing
Even with positive EV, parlays are volatile. Use **Kelly Criterion** or **flat betting** (1-3% of bankroll).

### 4. Track Results
Log your parlays and track:
- Hit rate by parlay size
- Actual EV vs predicted EV
- Which edge types perform best

---

## Example Workflow

```bash
# Monday: Check week's schedule
python auto_weekly_analyzer.py --week 10

# Thursday: Refs announced, analyze edges
python auto_weekly_analyzer.py --week 10

# Friday: Generate parlays from edges
python nfl_parlay_maker.py --week 10 --min-confidence 0.70 --top 20

# Saturday: Review, select best parlays, place bets

# Sunday-Monday: Track results
```

---

## Troubleshooting

### "No edges found!"
**Solution**:
1. Run `auto_weekly_analyzer.py --week N` first
2. Check if referee assignments are posted (usually Thursday)
3. Lower `--min-confidence` threshold

### "Not enough edges to make parlays"
**Solution**:
1. Lower `--min-confidence` (try 0.55 or 0.50)
2. Include more edge sizes: `--edge-sizes MASSIVE LARGE MEDIUM`
3. Reduce `--min-legs` to 2

### "All parlays have negative EV"
**Solution**:
This is normal if edges are weak. Only bet parlays with **positive EV**.

---

## Disclaimer

⚠️ **Parlays are high-risk, high-reward bets**

- ALL legs must hit for the parlay to win
- One loss = entire parlay loses
- Even 80% confidence edges miss 20% of the time
- Recommended for entertainment, not income
- Only bet what you can afford to lose
- Gamble responsibly

**Past performance does not guarantee future results.**

---

## Future Enhancements

Planned features:
- **Live odds integration** - Fetch current sportsbook odds
- **Line shopping** - Compare across multiple books
- **Round robin** - Generate round robin bets
- **Hedge calculator** - Mid-parlay hedge opportunities
- **Result tracking** - Automated P&L tracking
- **Multi-sport** - Combine with NHL, NBA, etc.

---

## Support

For issues or questions:
1. Check this README
2. Review `auto_weekly_analyzer.py` output
3. Verify referee assignments are available
4. Open an issue with details

---

Built with ❤️ by the NFL Referee Intelligence System
