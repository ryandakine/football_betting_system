# Action Network Scraping Guide

**WHY THIS IS CHALLENGING:**
Action Network frequently changes their HTML structure and uses dynamic loading (JavaScript). This guide shows multiple approaches.

---

## 🔧 Approach 1: Selenium (Most Reliable)

**Pros:**
- Handles JavaScript rendering
- Can wait for dynamic content
- Most robust for complex sites

**Cons:**
- Requires browser driver
- Slower than direct requests

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_with_selenium():
    # Setup
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run without GUI
    driver = webdriver.Chrome(options=options)

    try:
        # Navigate to Action Network
        driver.get("https://www.actionnetwork.com/nfl")

        # Wait for content to load
        wait = WebDriverWait(driver, 10)
        games = wait.until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "game-card"))
        )

        # Extract data
        for game in games:
            # Find betting percentages
            # (Selectors will change - inspect page to find current ones)
            public_pct = game.find_element(By.CLASS_NAME, "public-percent").text
            money_pct = game.find_element(By.CLASS_NAME, "money-percent").text

            print(f"Public: {public_pct}, Money: {money_pct}")

    finally:
        driver.quit()
```

---

## 🔧 Approach 2: API Endpoint (Fastest if Available)

**Check browser dev tools (Network tab) while browsing Action Network.**

```python
import requests

def scrape_with_api():
    # Sometimes Action Network loads data via API
    # Check Network tab in browser to find endpoint

    headers = {
        'User-Agent': 'Mozilla/5.0...',
        'Referer': 'https://www.actionnetwork.com/nfl'
    }

    # Example endpoint (check actual endpoint in browser)
    url = "https://api.actionnetwork.com/web/v1/leagues/7/games"

    response = requests.get(url, headers=headers)
    data = response.json()

    # Parse JSON (structure varies)
    for game in data.get('games', []):
        public_pct = game.get('public_betting_percent')
        money_pct = game.get('money_percent')
        print(f"Public: {public_pct}%, Money: {money_pct}%")
```

---

## 🔧 Approach 3: BeautifulSoup (Fast but Fragile)

```python
import requests
from bs4 import BeautifulSoup

def scrape_with_bs4():
    url = "https://www.actionnetwork.com/nfl"
    headers = {'User-Agent': 'Mozilla/5.0...'}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find game cards (inspect page for current structure)
    games = soup.find_all('div', class_='game-card')

    for game in games:
        # Extract data (selectors will vary)
        teams = game.find('div', class_='teams').text
        public = game.find('span', class_='public-percent').text
        money = game.find('span', class_='money-percent').text

        print(f"{teams}: Public {public}, Money {money}")
```

---

## 🎯 RECOMMENDED: Hybrid Approach

```python
def fetch_handle_data(week: int):
    """Try multiple methods in order"""

    # Try Method 1: API endpoint (fastest)
    try:
        data = _try_api_endpoint()
        if data:
            return data
    except:
        pass

    # Try Method 2: Selenium (most reliable)
    try:
        data = _try_selenium()
        if data:
            return data
    except:
        pass

    # Try Method 3: BeautifulSoup (fallback)
    try:
        data = _try_beautifulsoup()
        if data:
            return data
    except:
        pass

    # Fallback: Use sample/cached data
    return _use_sample_data()
```

---

## 📊 Alternative: Paid Data Services

**If scraping is too brittle, consider:**

1. **The Action Network Pro** ($50-200/month)
   - Official API access
   - Reliable data feed
   - No scraping needed

2. **SportsDataIO** (https://sportsdata.io)
   - NFL odds and betting percentages
   - API access: ~$50-150/month

3. **OddsJam** (https://oddsjam.com)
   - Line movement tracking
   - Public vs sharp money
   - API available

---

## 🔨 DIY: Manual Data Entry Tool

For now, you can manually enter handle data:

```bash
python manual_handle_entry.py

Enter game: BAL @ PIT
Home ML: -150
Away ML: +130
Home handle %: 85
Away handle %: 15

✅ Saved! Trap score: -100 (EXTREME TRAP)
```

---

## 📝 Current Status

**action_network_scraper.py:**
- ✅ Framework complete
- ✅ Sample data structure shown
- ⏸️ Actual scraping logic needs implementation
- ⏸️ Requires checking current HTML structure

**To implement:**
1. Inspect Action Network page in browser
2. Find current CSS selectors for:
   - Game cards
   - Public betting %
   - Money %
   - Line movement
3. Update scraper with current selectors
4. Test and adjust as needed

---

## 💡 Recommendation

**Short term:** Use manual entry or sample data to test trap detector

**Long term:** Either:
- Pay for Action Network Pro (~$100/month for reliable data)
- Build Selenium scraper (most reliable but slower)
- Find their API endpoint (check Network tab in browser)

The trap detector math is solid - you just need the handle data source!
