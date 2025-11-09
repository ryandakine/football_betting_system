# Prop Data Collection Guide
## How to Get 7 Years of NFL Prop Backtesting Data (2018-2024)

This guide explains how to collect the historical data needed to train Model 12 (Prop Intelligence).

---

## 📊 Data Sources

### 1. **Pro Football Reference** (PRIMARY SOURCE)
**URL**: `https://www.pro-football-reference.com/`

**What to get:**
- Player game logs (2018-2024)
- Team defense stats
- Game results with context

**How to scrape:**
```python
import requests
from bs4 import BeautifulSoup

# Example: Get Patrick Mahomes 2024 game log
url = "https://www.pro-football-reference.com/players/M/MahoPa00/gamelog/2024/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Parse table for passing yards, TDs, etc.
```

**CSV Export Option:**
- Many tables have "Share & Export" → "Get as CSV"
- Faster than scraping for bulk data

---

### 2. **ESPN API** (FREE, NO KEY REQUIRED)
**URL**: `https://site.api.espn.com/apis/site/v2/sports/football/nfl/`

**Endpoints:**
```bash
# Get all players
curl "https://site.api.espn.com/apis/site/v2/sports/football/nfl/athletes"

# Get player stats
curl "https://site.api.espn.com/apis/site/v2/sports/football/nfl/athletes/{player_id}/statistics"

# Get game data
curl "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=20241117&limit=100"
```

**Advantages:**
- No API key needed
- JSON format (easy to parse)
- Real-time updates

---

### 3. **Sports Reference API**
**URL**: `https://www.sports-reference.com/`

**Coverage:**
- Historical stats back to 1920
- Advanced metrics (EPA, DVOA, etc.)
- Playoff data

---

### 4. **Historical Prop Lines** (HARDEST TO GET)

**Sources:**
- **SportsOddsHistory.com** - Paid service ($$$)
- **OddsPortal.com** - Free historical odds
- **Archive.org** - Wayback Machine for DraftKings/FanDuel

**Challenge:**
- Most sportsbooks don't publish historical prop lines
- Need to scrape daily and store yourself
- Or buy from third-party data provider

---

## 🛠️ Step-by-Step Collection Process

### **Step 1: Get Player Universe**
Collect all relevant players (QB, RB, WR, TE) from 2018-2024.

```python
# Script: collect_player_universe.py
import requests
import json

seasons = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
all_players = []

for season in seasons:
    # Get all players for this season
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/athletes?season={season}"
    response = requests.get(url)
    data = response.json()

    # Filter to skill positions
    for player in data.get('athletes', []):
        if player['position']['abbreviation'] in ['QB', 'RB', 'WR', 'TE']:
            all_players.append({
                'name': player['displayName'],
                'id': player['id'],
                'position': player['position']['abbreviation'],
                'team': player.get('team', {}).get('abbreviation'),
                'season': season,
            })

# Save
with open('data/player_universe.json', 'w') as f:
    json.dump(all_players, f, indent=2)

print(f"✅ Collected {len(all_players)} players across {len(seasons)} seasons")
```

---

### **Step 2: Get Game Logs for Each Player**

For every player, get their week-by-week stats.

```python
# Script: collect_game_logs.py
import time

player_game_logs = []

for player in all_players:
    print(f"Fetching game log for {player['name']}...")

    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/athletes/{player['id']}/gamelog/{player['season']}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # Extract game-by-game stats
        for game in data.get('entries', []):
            game_log = {
                'player_name': player['name'],
                'player_id': player['id'],
                'position': player['position'],
                'team': player['team'],
                'season': player['season'],
                'week': game.get('week'),
                'opponent': game.get('opponent'),
                'is_home': game.get('homeAway') == 'home',

                # Stats
                'passing_yards': game.get('stats', {}).get('passingYards', 0),
                'passing_tds': game.get('stats', {}).get('passingTouchdowns', 0),
                'interceptions': game.get('stats', {}).get('interceptions', 0),
                'rushing_yards': game.get('stats', {}).get('rushingYards', 0),
                'rushing_tds': game.get('stats', {}).get('rushingTouchdowns', 0),
                'receiving_yards': game.get('stats', {}).get('receivingYards', 0),
                'receiving_tds': game.get('stats', {}).get('receivingTouchdowns', 0),
                'receptions': game.get('stats', {}).get('receptions', 0),
            }
            player_game_logs.append(game_log)

    time.sleep(0.5)  # Rate limit (be nice to ESPN!)

# Save
with open('data/player_game_logs_2018_2024.json', 'w') as f:
    json.dump(player_game_logs, f, indent=2)

print(f"✅ Collected {len(player_game_logs)} game logs")
```

**Expected output:** ~50,000+ game logs (500 players × 17 weeks × 7 years)

---

### **Step 3: Get Game Context (Spreads, Totals, Referees)**

For each game, get betting context.

```python
# Script: collect_game_context.py

game_contexts = []

for season in seasons:
    for week in range(1, 19):  # 18 weeks
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={season}&week={week}"
        response = requests.get(url)
        data = response.json()

        for game in data.get('events', []):
            context = {
                'game_id': game['id'],
                'season': season,
                'week': week,
                'date': game['date'],
                'home_team': game['competitions'][0]['competitors'][0]['team']['abbreviation'],
                'away_team': game['competitions'][0]['competitors'][1]['team']['abbreviation'],
                'home_score': game['competitions'][0]['competitors'][0]['score'],
                'away_score': game['competitions'][0]['competitors'][1]['score'],

                # Betting lines (if available)
                'spread': game.get('competitions', [{}])[0].get('odds', [{}])[0].get('details'),
                'total': game.get('competitions', [{}])[0].get('odds', [{}])[0].get('overUnder'),
            }
            game_contexts.append(context)

# Save
with open('data/game_contexts_2018_2024.json', 'w') as f:
    json.dump(game_contexts, f, indent=2)
```

---

### **Step 4: Get Historical Prop Lines** (MANUAL PROCESS)

**Option A: Daily Scraping (Forward-Looking)**
```python
# Run this DAILY to capture today's prop lines
import requests
from datetime import datetime

def scrape_todays_props():
    # DraftKings API (unofficial, may break)
    url = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/88808/categories/1000/subcategories/4511"

    response = requests.get(url)
    props = response.json()

    # Save with timestamp
    filename = f"data/props/props_{datetime.now().strftime('%Y%m%d')}.json"
    with open(filename, 'w') as f:
        json.dump(props, f)

    print(f"✅ Saved props to {filename}")

# Run daily via cron job
scrape_todays_props()
```

**Option B: Buy Historical Data**
- **SportsOddsHistory.com** - ~$500/year
- **Sportradar** - Enterprise pricing
- **The Odds API** - Limited free tier

---

## 📈 Expected Data Volume

After collecting 7 years (2018-2024):

| Data Type | Count | Size |
|-----------|-------|------|
| Players | ~800 | 50 KB |
| Game Logs | ~60,000 | 30 MB |
| Game Contexts | ~1,900 | 2 MB |
| Prop Lines | ~50,000 | 20 MB |
| **TOTAL** | **~112,000 records** | **~52 MB** |

---

## 🤖 Automated Collection Scripts

Put it all together in one master script:

```python
# master_data_collector.py

import os
import time
from pathlib import Path

def collect_all_data():
    """Run all collection scripts in sequence."""

    scripts = [
        "collect_player_universe.py",
        "collect_game_logs.py",
        "collect_game_context.py",
        "collect_defense_stats.py",
    ]

    for script in scripts:
        print(f"\n{'='*60}")
        print(f"Running: {script}")
        print(f"{'='*60}\n")

        os.system(f"python {script}")

        time.sleep(2)  # Pause between scripts

    print("\n✅ All data collection complete!")
    print("Data saved to: data/")

if __name__ == "__main__":
    collect_all_data()
```

---

## 🔥 Quick Start (No Scraping Required)

If you want to test the model without collecting 7 years of data:

```bash
# Run the framework with sample data
python prop_backtest_framework.py
```

This creates sample data files that demonstrate the structure.

Once you have real data, replace these files:
- `data/prop_backtest_data.json`
- `data/defense_vs_position_stats.json`
- `data/game_contexts_2018_2024.json`

---

## 🚀 Next Steps

1. **Collect the data** (use scripts above)
2. **Run training**: `python train_prop_model.py --seasons 2018-2023 --validate 2024`
3. **Backtest**: `python prop_backtest_framework.py --min-confidence 0.60`
4. **Integrate**: Add props to weekly analyzer

---

## ⚠️ Legal / Ethical Notes

- **Respect rate limits** when scraping (don't hammer servers)
- **Check robots.txt** before scraping any site
- **Terms of Service** - Some sites prohibit automated scraping
- **Sports Reference** - Consider supporting them with a subscription if using heavily
- **Personal use only** - Don't resell scraped data

---

## 📚 Resources

- [Pro Football Reference Python Library](https://github.com/roclark/sportsipy)
- [nfl_data_py](https://github.com/cooperdff/nfl_data_py) - Awesome NFL data loader
- [ESPN API Docs (Unofficial)](https://gist.github.com/akeaswaran/b48b02f1c94f873c6655e7129910fc3b)
- [The Odds API](https://the-odds-api.com/) - Live odds & props

---

**Ready to train on real data? Start with Step 1! 🏈**
