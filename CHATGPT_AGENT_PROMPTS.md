# ChatGPT Agent Mode Integration Prompts

**WHY THIS EXISTS:**
ChatGPT Agent Mode has web browsing and can collect data our system needs.
Use these prompts to collect referee data, verify backtests, and build intelligence.

**PHILOSOPHY: Investment → System**
- ChatGPT Agent Mode = Data Collection Agent (research)
- Your Betting System = Execution Agent (automation)
- Clean separation of concerns = robust system

---

## 🎯 PROMPT 1: Collect NFL Referee Intelligence

```
I need you to build a comprehensive NFL referee intelligence database for betting purposes.

TASK:
1. Go to footballzebras.com and other NFL referee tracking sites
2. For each active NFL referee, collect:
   - Full name
   - Average penalties called per game
   - Average penalty yards per game
   - Home vs away penalty bias
   - Impact on game pace (fast/slow)
   - Impact on scoring (over/under tendencies)
   - Any notable patterns

3. Historical data (2020-2024 seasons):
   - Games officiated
   - Average total points in their games
   - Over/under record
   - Penalty statistics by team

OUTPUT FORMAT:
Save as JSON file: referee_intelligence.json

{
  "referees": [
    {
      "name": "Shawn Hochuli",
      "avg_penalties_per_game": 15.2,
      "avg_penalty_yards_per_game": 127.5,
      "pace": "slow",
      "scoring_tendency": "under",
      "over_under_record": {
        "over": 45,
        "under": 67,
        "push": 3
      },
      "home_away_bias": {
        "home_penalties_avg": 7.8,
        "away_penalties_avg": 7.4
      },
      "betting_edge": "UNDER",
      "confidence_boost": 3,
      "notes": "Calls significantly more holding penalties than league average"
    }
  ]
}

FOCUS ON THESE REFEREES:
- Shawn Hochuli
- Adrian Hill
- Bill Vinovich
- Brad Allen
- Carl Cheffers
- Ron Torbert
- Clete Blakeman
- Craig Wrolstad
- John Hussey
- Land Clark
- Scott Novak
- Tra Blake

Be thorough - this data directly impacts betting ROI!
```

---

## 🎯 PROMPT 2: Verify Backtest Results

```
I need you to verify NFL game results for backtesting my betting model.

TASK:
I have a list of games and predicted outcomes. Please verify the actual results.

GAMES TO VERIFY:
[Paste your backtest games here]

Example format:
- Week 10, 2024: PHI @ GB - Predicted: GB -1.5 - Need: Final score

For each game, find:
1. Final score
2. Spread result (did favorite cover?)
3. Total result (over/under actual total)
4. Referee assigned to the game

OUTPUT FORMAT:
Save as JSON: backtest_verification.json

{
  "verified_games": [
    {
      "week": 10,
      "season": 2024,
      "game": "PHI @ GB",
      "final_score": {
        "away": 10,
        "home": 7
      },
      "spread": {
        "line": -1.5,
        "favorite": "GB",
        "result": "LOSS",
        "margin": -3
      },
      "total": {
        "line": 45.5,
        "actual": 17,
        "result": "UNDER"
      },
      "referee": "Shawn Hochuli"
    }
  ]
}

SOURCES TO USE:
- pro-football-reference.com
- ESPN.com
- NFL.com official stats
- footballzebras.com (for referee assignments)

Cross-reference at least 2 sources for accuracy!
```

---

## 🎯 PROMPT 3: Build Historical Referee Database

```
I need historical referee assignment data for NFL games (2020-2024 seasons).

TASK:
For each NFL season from 2020-2024, collect referee assignments for ALL games.

DATA NEEDED:
- Week number
- Game (AWAY @ HOME)
- Referee name
- Final score
- Total points
- Total penalties called
- Total penalty yards

OUTPUT FORMAT:
Save as JSON: referee_history_[season].json

{
  "season": 2024,
  "games": [
    {
      "week": 1,
      "game": "KC @ BAL",
      "referee": "Bill Vinovich",
      "final_score": {
        "away": 27,
        "home": 20
      },
      "total_points": 47,
      "penalties": {
        "total": 12,
        "home": 6,
        "away": 6,
        "total_yards": 98
      }
    }
  ]
}

SOURCES:
- footballzebras.com (referee assignments)
- pro-football-reference.com (game stats)
- nflpenalties.com (penalty data)

This will help identify which referees correlate with overs/unders!
```

---

## 🎯 PROMPT 4: Current Week Referee Assignments

```
I need this week's NFL referee assignments.

TASK:
Go to footballzebras.com and find the current week's referee assignments.

OUTPUT FORMAT:
Save as JSON: week_[X]_referees.json

{
  "week": 11,
  "season": 2024,
  "assignments": [
    {
      "game": "KC @ BUF",
      "referee": "Bill Vinovich",
      "kickoff": "2024-11-17T13:00:00",
      "betting_intelligence": {
        "edge": "OVER",
        "confidence": 2,
        "reasoning": "Vinovich has lowest flag rate in NFL, games trend faster pace"
      }
    }
  ]
}

ALSO CHECK:
- Any referee news (injuries, replacements)
- Crew assignments (some crews call more penalties)
- Playoff implications (refs may be tighter in important games)
```

---

## 🎯 PROMPT 5: Line Movement Analysis

```
I need to track line movement for this week's NFL games.

TASK:
For each game this week, track how the betting lines have moved over time.

DATA TO COLLECT:
- Opening line (Monday/Tuesday)
- Movement throughout week
- Current line (game day)
- Public betting percentages
- Sharp money indicators

OUTPUT FORMAT:
Save as JSON: week_[X]_line_movement.json

{
  "week": 11,
  "games": [
    {
      "game": "KC @ BUF",
      "spread": {
        "opening": -2.5,
        "current": -1.5,
        "movement": "+1.0 towards KC",
        "sharp_side": "BUF"
      },
      "total": {
        "opening": 47.5,
        "current": 45.5,
        "movement": "-2.0 towards UNDER"
      },
      "public_betting": {
        "spread_public_pct": 65,
        "total_public_pct": 58
      }
    }
  ]
}

SOURCES:
- ActionNetwork.com
- TheLines.com
- Covers.com

Sharp money moves are VERY valuable!
```

---

## 💾 HOW TO USE THE DATA

After ChatGPT Agent Mode collects the data:

1. **Save JSON files to `/data` directory**
   ```bash
   # ChatGPT gives you JSON → Save to:
   /home/user/football_betting_system/data/referee_intelligence.json
   /home/user/football_betting_system/data/backtest_verification.json
   /home/user/football_betting_system/data/referee_history_2024.json
   ```

2. **Import into referee_fetcher.py**
   ```python
   # referee_fetcher.py will automatically read from:
   # 1. data/referee_intelligence.json (expanded database)
   # 2. data/referee_cache.json (weekly assignments)
   ```

3. **Use for backtesting**
   ```bash
   python backtest_with_verified_data.py
   ```

---

## 🎯 WORKFLOW EXAMPLE

**Monday (Data Collection with ChatGPT):**
```
You: [Use Prompt 4 - Get current week referee assignments]
ChatGPT Agent Mode: *Browses web, collects data*
ChatGPT: Here's week_11_referees.json
You: *Save file to /data directory*
```

**Tuesday-Friday (System Works Automatically):**
```bash
# Your betting system uses the data ChatGPT collected
python auto_execute_bets.py --auto

# Outputs:
# ✅ Referee: Bill Vinovich (from ChatGPT's data)
# ✅ Betting edge: OVER +2% (from referee_intelligence.json)
```

**Monday After Games (Verification with ChatGPT):**
```
You: [Use Prompt 2 - Verify results]
ChatGPT Agent Mode: *Verifies all game results*
You: *Update backtest database*
```

---

## 🎯 SEPARATION OF CONCERNS

| Task | Agent | Why |
|------|-------|-----|
| **Data Collection** | ChatGPT Agent Mode | Can browse web, research, verify sources |
| **Data Storage** | Your system (JSON files) | Single source of truth |
| **Execution** | Your betting system | Fast, automated, reliable |
| **Verification** | ChatGPT Agent Mode | Cross-reference multiple sources |

---

## 🚀 OPERATIONAL EFFICIENCY

**Before (Manual):**
- 3 hours/week collecting referee data
- 2 hours/week verifying backtest results
- 1 hour/week tracking line movement
- **Total: 6 hours/week**

**After (ChatGPT + Your System):**
- 10 min/week (paste prompts to ChatGPT)
- 5 min/week (save JSON files)
- **Total: 15 minutes/week**

**Time saved: 5.75 hours/week × 17 weeks = 98 hours/season!**

---

## 💡 NEXT STEPS

1. Copy Prompt 1 → Paste into ChatGPT Agent Mode
2. Save the JSON output to `/data` directory
3. Your betting system will automatically use it
4. Repeat weekly with Prompt 4

**This is perfect "Investment → System" architecture!** 🎯
