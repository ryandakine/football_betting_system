# 🤖 NCAA Betting Agent System

**Autonomous agent for NCAA football betting analysis and predictions**

## 🎯 Overview

The NCAA Betting Agent is a multi-agent system that autonomously:
- 📊 Collects game data and SP+ ratings from College Football Data API
- 📈 Runs backtests to validate system performance
- 🎯 Generates betting picks for upcoming games
- 📉 Tracks results and monitors profitability

## 🏗️ Architecture

```
NCAA Betting Agent (Main Orchestrator)
├── Data Collector Agent     → Fetches games & SP+ ratings
├── Analysis Agent           → Runs backtests & evaluates performance
├── Prediction Agent         → Generates picks with edge calculations
└── Performance Tracker      → Monitors bet results & ROI
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `requests` - API calls
- `numpy` - Calculations
- `scipy` - Statistical tests
- `python-dotenv` - Environment variables

### 2. Configure API Keys

Copy the example config:
```bash
cp ncaa_agent_config.example.env .env
```

Edit `.env` and add your API keys:
```bash
CFB_DATA_API_KEY=M9/VpZQNUSQfSUd6OtHZaTetz9/hH2zAqSqQcNLxLTheI43qhvEWwgpe+n6Rzg7G
NCAA_BANKROLL=10000
NCAA_MIN_EDGE=0.03
NCAA_MIN_CONFIDENCE=0.60
```

### 3. Run the Agent

**Manual Mode (Interactive):**
```bash
python ncaa_agent.py --manual
```

This opens an interactive shell where you can:
- `update` - Update game data
- `backtest` - Run backtest analysis
- `picks` - Generate picks for upcoming games
- `results` - Check recent bet results
- `status` - Show agent status
- `run` - Run full daily cycle

**Automated Mode:**
```bash
python ncaa_agent.py
```

This runs the full daily cycle automatically.

### 4. Schedule Automation (Optional)

**Using cron (Linux/Mac):**
```bash
# Run every Tuesday, Wednesday, Saturday at 9 AM
0 9 * * 2,3,6 /path/to/run_ncaa_agent.sh

# Add to crontab:
crontab -e
```

**Using Windows Task Scheduler:**
- Create a task that runs `python ncaa_agent.py`
- Set schedule: Tuesday, Wednesday, Saturday at 9 AM

## 📋 Agent Commands (Manual Mode)

### `update` - Update Data
Fetches latest games and SP+ ratings for current season.

```
> update
📊 Checking for data updates...
Updating NCAA data...
✅ Collected 3595 games
```

### `backtest` - Run Backtest
Analyzes historical performance to validate system.

```
> backtest
📈 Running backtest...
Backtest Results:
  ROI: 12.45%
  Win Rate: 56.2%
  P-Value: 0.0123 (significant!)
```

### `picks` - Generate Picks
Creates picks for upcoming games within next 7 days.

```
> picks
🎯 Generating picks for upcoming games...

🏈 BETTING PICKS
1. Texas A&M @ Alabama
   Date: 2024-11-10
   Pick: home (Alabama)
   Edge: 8.5%
   Confidence: 72%
   Recommended Bet: $145.00
   Reasoning: Alabama has significant SP+ advantage (32.5 vs 18.2)
```

### `results` - Check Results
Tracks outcomes of recent bets.

```
> results
📊 Tracking recent results...
Recent Results (5 bets):
  Wins: 3
  Losses: 2
  Win Rate: 60.0%
  Profit: $156.30
```

### `status` - Agent Status
Shows current agent state.

```
> status
AGENT STATUS
Season: 2024
Bankroll: $10,156.30
Total Bets: 28
Total Wins: 16
Win Rate: 57.1%
Last Run: 2024-11-09 09:00:15
```

## 🎛️ Configuration Options

### Betting Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `NCAA_BANKROLL` | Starting bankroll | 10000 | Your actual bankroll |
| `NCAA_MIN_EDGE` | Minimum edge to bet | 0.03 (3%) | 0.03-0.05 |
| `NCAA_MIN_CONFIDENCE` | Minimum confidence | 0.60 (60%) | 0.60-0.65 |
| `NCAA_MAX_BET_PERCENT` | Max % of bankroll per bet | 0.15 (15%) | 0.10-0.20 |

### Agent Behavior

| Parameter | Description | Default |
|-----------|-------------|---------|
| `NCAA_AUTO_COLLECT` | Auto-collect data | true |
| `NCAA_AUTO_GENERATE_PICKS` | Auto-generate picks | true |
| `NCAA_AUTO_TRACK_RESULTS` | Auto-track results | true |
| `NCAA_RUN_DAYS` | Days to run agent | tue,wed,sat |

## 📊 Output Files

The agent creates the following files:

### Data Files
```
data/football/historical/ncaaf/
├── ncaaf_2024_games.json          # Game results
├── ncaaf_2024_sp_ratings.json     # SP+ ratings
└── ncaaf_2024_stats.json          # Team stats
```

### Agent State
```
data/agents/ncaa/results/
├── agent_state.json               # Agent state (bankroll, bets, etc.)
├── picks_20241109.json            # Generated picks
├── placed_bets.json               # Bets you've placed
├── bet_results.json               # Results of completed bets
└── backtest_20241109.json         # Backtest results
```

## 🔄 Daily Workflow

The agent follows this daily workflow:

1. **Data Update** (Weekly, Monday/Tuesday)
   - Fetches latest games
   - Updates SP+ ratings
   - Saves to data directory

2. **Backtest** (Weekly, Tuesday)
   - Validates system on current season
   - Checks statistical significance
   - Generates recommendations

3. **Generate Picks** (Daily, if games upcoming)
   - Analyzes games in next 7 days
   - Calculates edge and confidence
   - Recommends bet sizes

4. **Track Results** (Daily)
   - Checks completed games
   - Updates win/loss records
   - Calculates profit/loss

## 🎯 Recommendation System

The agent provides actionable recommendations:

### High ROI (>15%)
```
✅ System is highly profitable!
💰 Consider:
   - Start live betting with small units
   - Gradually increase stake sizes
   - Add weather and injury data
```

### Moderate ROI (5-15%)
```
⚠️ System shows promise
💡 Suggestions:
   - Collect more historical data
   - Focus on best-performing conferences
   - Test on paper for 2-3 weeks
```

### Low ROI (<5%)
```
❌ System needs improvement
🔧 Actions:
   - Increase min_confidence threshold
   - Review SP+ integration
   - DO NOT use for live betting yet
```

## 📈 Advanced Features

### Conference Analysis
The agent tracks performance by conference:
```
🏟️  Results by Conference:
   SEC              : 45 bets | WR: 62.2% | Profit: $1,234.50
   Big Ten          : 38 bets | WR: 55.3% | Profit: $   567.23
   Big 12           : 31 bets | WR: 48.4% | Profit: $  -123.45
```

### Statistical Significance
Every backtest includes a t-test:
```
📊 Statistical Significance:
   T-Statistic: 2.45
   P-Value: 0.0156
   ✅ Results are statistically significant (p < 0.05)
```

### Kelly Criterion Bet Sizing
Automatic optimal bet sizing:
```python
# Agent calculates:
kelly_fraction = (win_prob * odds - (1 - win_prob)) / odds
recommended_bet = bankroll * kelly_fraction * safety_factor
```

## 🔒 Safety Features

1. **Max Bet Limits**
   - Never bets more than configured % of bankroll
   - Caps individual bets at reasonable amounts
   - Prevents over-betting on single games

2. **Statistical Validation**
   - Requires p-value < 0.05 for significance
   - Tracks confidence intervals
   - Warns if results could be random

3. **Bankroll Protection**
   - Stops betting if bankroll drops too low
   - Reduces stake sizes after losses
   - Implements Kelly Criterion for optimal sizing

## 🐛 Troubleshooting

### "No API key found"
- Check that `.env` file exists
- Verify `CFB_DATA_API_KEY` is set
- Make sure key is valid at collegefootballdata.com

### "No games found"
- Check if it's NCAA season (Aug-Jan)
- Verify API key has access
- Run `update` command to fetch data

### "No SP+ ratings"
- SP+ requires Silver/Gold tier ($25/month)
- Free tier can still use basic stats
- Agent will warn if SP+ unavailable

### "Results not significant"
- Need more data (collect more seasons)
- Try adjusting min_edge and min_confidence
- Run longer backtests (2015-2024)

## 📚 Integration Examples

### Slack Notifications
```python
# Add to agent after generating picks
import requests

webhook = os.getenv('SLACK_WEBHOOK_URL')
if webhook and picks:
    message = f"🏈 {len(picks)} new NCAA picks generated!"
    requests.post(webhook, json={'text': message})
```

### Email Alerts
```python
# Email when system finds high-edge picks
for pick in picks:
    if pick['edge'] > 0.10:  # 10%+ edge
        send_email(f"High-value pick: {pick['predicted_winner']}")
```

### Discord Bot
```python
# Post picks to Discord channel
import discord

if picks:
    await discord_channel.send(f"New picks: {len(picks)}")
```

## 🎓 Best Practices

1. **Start with Paper Betting**
   - Track picks without real money
   - Validate for 20-30 bets
   - Only go live if profitable

2. **Use Proper Bankroll Management**
   - Never bet more than 20% on single game
   - Follow Kelly Criterion recommendations
   - Keep separate betting bankroll

3. **Monitor Performance**
   - Check `results` daily
   - Review backtests weekly
   - Adjust thresholds as needed

4. **Respect the Market**
   - Edges are small (3-8%)
   - Need volume for profitability
   - Variance is high in college football

5. **Continuous Improvement**
   - Collect more historical data
   - Add weather integration
   - Track closing line value

## 📞 Support

- **API Issues**: https://collegefootballdata.com
- **Agent Issues**: Check logs in `data/agents/ncaa/agent.log`
- **Questions**: Review NCAA_NEXT_STEPS.md

## 🎉 Example Session

```bash
$ python ncaa_agent.py --manual

NCAA BETTING AGENT - MANUAL MODE

Commands:
  1. update    - Update game data
  2. backtest  - Run backtest
  3. picks     - Generate picks
  4. results   - Check results
  5. status    - Show status
  6. run       - Run full cycle
  7. exit      - Exit

> status
AGENT STATUS
Season: 2024
Bankroll: $10,000.00
Total Bets: 0
Last Run: Never

> update
📊 Updating NCAA data...
✅ Collected 3,595 games

> backtest
📈 Running backtest...
ROI: 12.3%
Win Rate: 56.1%
P-Value: 0.0145 ✅

> picks
🎯 Generating picks...

🏈 BETTING PICKS
1. Georgia @ Alabama
   Pick: home (Alabama)
   Edge: 7.2%
   Bet: $135.00

2. Ohio State @ Penn State
   Pick: away (Ohio State)
   Edge: 5.8%
   Bet: $105.00

> exit
Goodbye!
```

---

**Ready to go! Start with:**
```bash
python ncaa_agent.py --manual
```

Good luck! 🏈💰
