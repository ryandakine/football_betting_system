# Warp AI Setup & Usage Guide 🤖

**Your AI-powered terminal for running the betting system!**

---

## 🎯 What is Warp AI?

**Warp** is a modern terminal with **built-in AI assistance**.

**Why it's perfect for your betting system:**
- ✅ AI helps you write commands
- ✅ Explains errors in plain English
- ✅ Suggests fixes automatically
- ✅ Learns your workflow
- ✅ Works with all your Python scripts

**Think of it as:** ChatGPT inside your terminal!

---

## 📥 How to Get Warp

### **Step 1: Download Warp**

**Website:** https://www.warp.dev

**Platforms:**
- ✅ macOS (fully supported)
- ✅ Linux (beta)
- ⏸️ Windows (coming soon - use WSL)

### **Step 2: Install**

**macOS:**
```bash
# Download from website and drag to Applications
# OR use Homebrew:
brew install --cask warp
```

**Linux:**
```bash
# Download .deb or .rpm from warp.dev
# OR use their install script:
curl -fsSL https://warp.dev/install.sh | bash
```

**Windows:**
```bash
# Use WSL (Windows Subsystem for Linux)
# Then follow Linux instructions
```

### **Step 3: First Launch**

1. Open Warp
2. Sign in (free account)
3. Complete tutorial (5 minutes)
4. You're ready! 🎉

---

## 🤖 How to Use Warp AI

### **Activating Warp AI**

**Keyboard Shortcut:** `Ctrl + ` (backtick) or `Cmd + ` on Mac

**Or type:** `#` at the start of a command line

### **Basic Usage:**

```bash
# Type your question after #
# ai: how do I list all Python files?

# Warp AI responds:
ls *.py

# OR
find . -name "*.py"
```

**Then you can:**
- Press Enter to run the command
- Edit it first
- Ask a follow-up question

---

## 🎯 EXACT PROMPTS FOR YOUR BETTING SYSTEM

Copy-paste these into Warp AI (Ctrl + `)

### **🚀 Prompt 1: Setup & Installation**

```
I have a football betting system in Python. Help me:

1. Install all dependencies:
   - selenium
   - webdriver-manager
   - requests
   - beautifulsoup4
   - other requirements if any

2. Set up environment variables:
   - ODDS_API_KEY (for The Odds API)
   - OPENROUTER_API_KEY (for DeepSeek-R1)

3. Test that Chrome/ChromeDriver works for Selenium

4. Verify all Python scripts are executable

Show me the exact commands to run.
```

### **📊 Prompt 2: Run Weekly Analysis**

```
I need to run my NFL betting workflow for Week 11. The workflow is:

1. Scrape Action Network for handle data (selenium scraper)
2. Run contrarian intelligence analysis
3. Run trap detection
4. Generate betting recommendations
5. Execute auto_execute_bets.py

The main script is: auto_execute_bets.py
Other scripts: action_network_selenium_scraper.py, trap_detector.py

Show me the commands to run the complete workflow.
If any errors occur, help me debug them.
```

### **🔍 Prompt 3: Debug Selenium Scraper**

```
My Selenium scraper (action_network_selenium_scraper.py) isn't finding game elements.

The script uses these selectors:
- [data-testid='game-card']
- .game-box
- .game-card

Help me:
1. Run the scraper with browser visible (--show-browser)
2. If it fails, tell me how to inspect the page
3. Suggest updated selectors based on Action Network's current HTML
4. Test the updated scraper

The scraper file is: action_network_selenium_scraper.py
```

### **💰 Prompt 4: Check Bankroll & Results**

```
I want to check my betting system's performance. Help me:

1. Check current bankroll: python bankroll_tracker.py --balance
2. Show statistics: python bankroll_tracker.py --stats
3. Check circuit breaker status: python circuit_breaker.py --status
4. Show recent bet history

Parse the output and tell me:
- Current ROI
- Win rate
- Any warnings or concerns
```

### **🏈 Prompt 5: Week 11 Game Analysis**

```
I need to analyze Week 11 NFL games. Help me:

1. Get referee assignments for Week 11
2. Run trap detection on these games:
   - KC @ BUF
   - SF @ GB
   - BAL @ PIT

3. For each game, show:
   - Trap score
   - Contrarian signals
   - Referee intelligence
   - Betting recommendation

Use these scripts:
- referee_fetcher.py
- trap_detector.py
- contrarian_intelligence.py
```

### **🐛 Prompt 6: Fix Any Error**

```
I got this error when running [SCRIPT_NAME]:

[PASTE ERROR HERE]

Help me:
1. Explain what's wrong in simple terms
2. Show the exact fix
3. Prevent this error in the future

Context: This is part of a Python betting system for NFL games.
```

### **📦 Prompt 7: Backup & Deploy**

```
I want to backup and deploy my betting system. Help me:

1. Create a git commit with all changes
2. Push to remote repository
3. Create a requirements.txt with all dependencies
4. Set up a cron job to run weekly analysis every Friday at 5 PM

My system is in: /home/user/football_betting_system
Branch: claude/api-key-setup-011CUsB5yQx4pT7x9waepBqd
```

---

## 💡 WARP AI TIPS & TRICKS

### **Tip 1: Conversational Debugging**

```bash
# ai: I ran trap_detector.py and got a KeyError. The error says 'home_handle' is missing.

Warp AI: Let me help debug this...
1. Check if the input data has 'home_handle' field
2. Add error handling for missing fields
3. Here's the fixed code...
```

### **Tip 2: Command History with AI**

```bash
# ai: show me the last 5 commands I ran that involved bankroll

Warp AI shows:
1. python bankroll_tracker.py --balance
2. python bankroll_tracker.py --stats
3. ...
```

### **Tip 3: Multi-Step Workflows**

```bash
# ai: Create a workflow to:
# 1. Scrape Action Network
# 2. Run trap detection
# 3. Save results to JSON
# 4. Open results in browser

Warp AI creates a script for you!
```

### **Tip 4: Explain Any Output**

```bash
python trap_detector.py --game "KC @ BUF" [outputs trap score]

# ai: explain this output

Warp AI: This shows a trap score of -80, which means...
```

### **Tip 5: Quick Scripts**

```bash
# ai: create a bash script that runs all my betting analysis

Warp AI writes the script for you!
```

---

## 🔄 DAILY WORKFLOW WITH WARP AI

### **Monday Morning (Check Results)**

```bash
# In Warp, press Ctrl + `

# ai: show me my betting results from this weekend
```

Warp AI will:
1. Run `python bankroll_tracker.py --stats`
2. Parse the output
3. Summarize wins/losses/ROI

### **Friday Evening (Week Analysis)**

```bash
# ai: run my Week 11 betting analysis

# Auto-runs:
# 1. Scraper
# 2. Trap detection
# 3. Contrarian analysis
# 4. Generates betting card
```

### **Saturday (Place Bets)**

```bash
# ai: show me this week's betting recommendations

# Displays:
# - Best bets
# - Trap games to avoid
# - Contrarian opportunities
```

---

## 🎯 WHY WARP AI > REGULAR TERMINAL

| Task | Regular Terminal | Warp AI |
|------|-----------------|---------|
| **Fix error** | Google it, debug manually | Ask Warp, get instant fix |
| **Write command** | Remember syntax | Describe what you want |
| **Debug** | Read stack traces | Plain English explanation |
| **Workflow** | Write bash script manually | Describe workflow, AI writes it |
| **Learn new tool** | Read docs | Ask Warp how to use it |

---

## 🚀 EXAMPLE SESSION

```bash
# You open Warp terminal

$ cd /home/user/football_betting_system

# Press Ctrl + `

# ai: I want to run my betting workflow for this week

Warp AI: I'll help you run the betting workflow. Here's what I'll do:

1. Check current bankroll
2. Scrape Action Network for Week 11 data
3. Run trap detection
4. Run contrarian analysis
5. Generate betting recommendations

Let's start:

$ python bankroll_tracker.py --balance
$100.00

$ python action_network_selenium_scraper.py --week 11 --save
🔧 Setting up Chrome driver...
✅ Chrome driver ready
🌐 Loading Action Network NFL page...
✅ Page loaded successfully
📊 Extracting game data...
✅ Extracted 14 games
💾 Saved to data/action_network_scraped_20251112_170000.json

$ python trap_detector.py --game "KC @ BUF" --home-ml -130 --away-ml +110 --home-handle 0.75 --away-handle 0.25
🎯 TRAP DETECTION: KC @ BUF
Trap Score: -60 (STRONG TRAP)
RECOMMENDATION: FADE BUF - Bet KC

# ai: what does trap score -60 mean?

Warp AI: A trap score of -60 means "STRONG TRAP":
- Public is too heavily betting on BUF (75%)
- Expected handle at -130 odds: only 57%
- Divergence of 18% = sharps are on KC
- RECOMMENDATION: Bet on KC (fade the public)

This is a strong contrarian signal!

# ai: should I bet on KC?

Warp AI: Based on the trap score of -60:
- This is a STRONG trap signal
- Historically, fading traps ≥60 has +8% ROI
- I recommend betting KC

BUT also check:
1. DeepSeek-R1 analysis (your main model)
2. Referee intelligence
3. Contrarian signals

Run: python auto_execute_bets.py --auto
This will give you the complete analysis.
```

**See how natural it is?** 🎯

---

## 📝 GETTING STARTED CHECKLIST

### **Setup (Do Once)**

- [ ] Download Warp from https://warp.dev
- [ ] Install Warp
- [ ] Sign in (free account)
- [ ] Complete tutorial
- [ ] Open your betting system directory
- [ ] Test Warp AI with: `# ai: list all Python files`

### **First Analysis (Try Now!)**

- [ ] Open Warp
- [ ] `cd` to your betting system directory
- [ ] Press `Ctrl + `` to activate Warp AI
- [ ] Paste "Prompt 2: Run Weekly Analysis" (from above)
- [ ] Watch Warp AI run your workflow!

### **Daily Usage**

- [ ] Use Warp as your main terminal
- [ ] Press `Ctrl + `` whenever you need help
- [ ] Let AI explain errors
- [ ] Let AI write commands for you

---

## 🎓 LEARN MORE

**Warp Docs:** https://docs.warp.dev
**Warp AI Docs:** https://docs.warp.dev/features/warp-ai
**Community:** https://discord.gg/warpdotdev

---

## 🔥 PRO TIP

**Create Warp Workflows:**

1. In Warp, click "Workflows" (left sidebar)
2. Create new workflow: "Run Weekly Betting Analysis"
3. Add your commands:
   ```bash
   python bankroll_tracker.py --balance
   python action_network_selenium_scraper.py --week 11 --save
   python auto_execute_bets.py --auto
   ```
4. Save workflow
5. Now just click "Run Weekly Betting Analysis" button!

**One-click weekly analysis!** 🚀

---

## 💰 BOTTOM LINE

**Warp AI makes running your betting system:**
- ✅ Faster (no memorizing commands)
- ✅ Easier (AI writes commands for you)
- ✅ Smarter (AI explains what's happening)
- ✅ Less error-prone (AI catches mistakes)

**Download:** https://warp.dev
**Start using:** Press `Ctrl + `` and paste the prompts above!

**You'll never go back to a regular terminal!** 🎯
