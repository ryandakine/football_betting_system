# NCAA Daily Betting Schedule - Tuesday Through Saturday

## 🎯 Why NCAA is BETTER Than NFL for Betting

**NFL:** Mostly Sunday/Monday (2 days/week)
**NCAA:** Tuesday-Saturday (5 days/week) = **2.5X more opportunities!**

---

## 📅 Typical Weekly Game Distribution

### **Tuesday - "MACtion Nights"** 🌙
**Games:** 5-10 games
**Conferences:** MAC (Mid-American Conference)
**Kickoffs:** 7:00 PM - 8:00 PM ET
**Volume:** Low

**Why it matters:**
- Lesser-known teams = market less efficient
- Lower betting limits = less sharp action
- More opportunity for model edge

**Example Teams:**
- Ball State, Toledo, Bowling Green
- Western Michigan, Northern Illinois
- Buffalo, Ohio, Kent State

**Typical Lines Come Out:** Monday afternoon

---

### **Wednesday - "MACtion Continues"** 🌙
**Games:** 5-10 games
**Conferences:** MAC, occasional Conference USA
**Kickoffs:** 7:00 PM - 8:00 PM ET
**Volume:** Low

**Why it matters:**
- Same as Tuesday - inefficient markets
- Books focus on preparing for Thursday slate
- Public bettors less engaged midweek

**Typical Lines Come Out:** Tuesday afternoon

---

### **Thursday - "Conference Slates Begin"** 🔥
**Games:** 10-20 games
**Conferences:**
- Conference USA
- Sun Belt
- Mountain West
- Occasional Pac-12/Big Ten/SEC

**Kickoffs:** 7:00 PM - 10:30 PM ET
**Volume:** Medium

**Why it matters:**
- First major slate of the week
- NFL Thursday game competes for attention
- Books splitting focus = opportunities

**Key Matchups:**
- Often features ranked teams (if Pac-12 or Big Ten)
- Rivalry games sometimes scheduled here
- ESPN primetime games

**Typical Lines Come Out:** Tuesday/Wednesday

---

### **Friday - "Pac-12 After Dark" 🌃
**Games:** 15-25 games
**Conferences:**
- Pac-12 (heavy)
- Big Ten
- ACC
- AAC (American)

**Kickoffs:** 7:00 PM - 10:30 PM ET (1:30 AM ET for West Coast!)
**Volume:** Medium-High

**Why it matters:**
- **Pac-12 After Dark is LEGENDARY** for chaos
- Late West Coast games (10:30 PM ET starts)
- Public tired by late games = less sharp betting
- Books preparing for Saturday = less attention

**Famous Pac-12 After Dark chaos:**
- Unranked teams upset Top 10 teams regularly
- Weird game scripts (shootouts, defensive battles)
- Less public betting = softer lines

**Typical Lines Come Out:** Wednesday/Thursday

---

### **Saturday - "THE MAIN EVENT" 🏈🏈🏈
**Games:** 50-70+ games
**Conferences:** ALL CONFERENCES
**Kickoffs:** Noon - 10:30 PM ET (12+ hour window!)
**Volume:** MASSIVE

**Time Slots:**
- **Noon ET:** Big Ten, ACC, SEC early games (15-20 games)
- **3:30 PM ET:** Main CBS/FOX windows (20-25 games)
- **7:00 PM ET:** ABC/ESPN primetime (10-15 games)
- **10:30 PM ET:** Pac-12 After Dark (5-10 games)

**Why it matters:**
- Most games = most opportunities
- Top teams play = tighter lines
- But volume = still find edge in lesser games
- Multiple windows = shop for best odds throughout day

**Typical Lines Come Out:** Thursday/Friday (adjust through Saturday morning)

---

## 🎯 Optimal Daily Betting Strategy

### **Monday (Rest Day)**
- ❌ No games
- ✅ Review weekend results
- ✅ Calculate ROI for past week
- ✅ Prepare bankroll for new week

### **Tuesday Morning (9 AM)**
```bash
python ncaa_daily_predictions.py YOUR_API_KEY
```
- Check Tuesday night MACtion lines
- Place bets on 5%+ edge opportunities
- Expected: 2-3 bets

### **Wednesday Morning (9 AM)**
```bash
python ncaa_daily_predictions.py YOUR_API_KEY
```
- Check Wednesday MACtion + early Thursday lines
- Place bets on midweek games
- Expected: 3-5 bets

### **Thursday Morning (9 AM)**
```bash
python ncaa_daily_predictions.py YOUR_API_KEY
```
- Check Thursday + early Friday lines
- Place bets on conference slate games
- Expected: 5-8 bets

### **Friday Morning (9 AM)**
```bash
python ncaa_daily_predictions.py YOUR_API_KEY
```
- Check Friday Pac-12 + early Saturday lines
- Place bets on Friday games + best Saturday opportunities
- Expected: 8-12 bets

### **Saturday Morning (9 AM)**
```bash
python ncaa_daily_predictions.py YOUR_API_KEY
```
- Check full Saturday slate
- Final adjustments as lines move
- Place remaining Saturday bets
- Expected: 10-15 additional bets

---

## 📊 Weekly Volume Analysis

| Day | Games | Expected Bets | Edge Opportunities |
|-----|-------|---------------|-------------------|
| Tuesday | 5-10 | 2-3 | ⭐⭐ (High edge, low volume) |
| Wednesday | 5-10 | 2-3 | ⭐⭐ (High edge, low volume) |
| Thursday | 10-20 | 5-8 | ⭐⭐⭐ (Medium edge, medium volume) |
| Friday | 15-25 | 8-12 | ⭐⭐⭐⭐ (Good edge, good volume) |
| Saturday | 50-70 | 10-15 | ⭐⭐⭐⭐⭐ (Best volume!) |
| **WEEKLY TOTAL** | **85-135** | **27-41** | **Excellent** |

---

## 💡 Pro Tips for Daily NCAA Betting

### **1. MACtion Midweek (Tue/Wed) = Best Edge**
**Why:**
- Books don't spend as much time on these lines
- Public bettors less engaged
- Lower betting limits = less sharp money

**Strategy:**
- Jump on these EARLY (lines come out Monday/Tuesday)
- Models have MOST edge here
- Even 3-5% edge is huge for these games

### **2. Thursday = First Major Test**
**Why:**
- First big slate shows how sharp your models are
- Often features ranked teams
- Good balance of opportunity vs. competition

**Strategy:**
- Compare your lines vs. market carefully
- Books are sharper on these than midweek
- Need 7-10% edge to be comfortable

### **3. Friday Pac-12 After Dark = Chaos Factor**
**Why:**
- Famous for upsets and weird games
- Late kickoffs (10:30 PM ET / 7:30 PM PT)
- Public bettors tired/drunk = soft late lines

**Strategy:**
- Model might show edge but chaos factor is real
- Reduce stake sizes slightly (1-2% instead of 3%)
- Late line movement can be dramatic

### **4. Saturday = Volume Game**
**Why:**
- 50-70 games = can't ALL be efficient
- Focus on non-marquee games for most edge
- Top 25 matchups have sharpest lines

**Strategy:**
- Avoid betting Alabama vs Georgia (line is sharp)
- Target: Sun Belt vs Conference USA type games
- Spread bets across multiple games (diversify)

---

## 🤖 Automation Setup (Recommended)

### **Cron Job - Run Daily at 9 AM**

```bash
# On your local machine or server:
crontab -e

# Add this line:
0 9 * * * cd /path/to/football_betting_system && python ncaa_daily_predictions.py YOUR_API_KEY > /path/to/logs/ncaa_$(date +\%Y\%m\%d).log 2>&1
```

**What this does:**
- Runs automatically every day at 9 AM
- Fetches games in next 48 hours
- Generates predictions
- Saves to log file
- You wake up to fresh opportunities!

### **Email Notifications (Optional)**

```bash
# Install mail utility:
sudo apt-get install mailutils

# Update cron job:
0 9 * * * cd /path/to/football_betting_system && python ncaa_daily_predictions.py YOUR_API_KEY | mail -s "NCAA Daily Predictions" your@email.com
```

---

## 💰 Expected Weekly Bankroll Usage

**Starting Bankroll:** $1,000
**Stake per bet:** 1-3% ($10-30)
**Weekly bets:** 27-41 bets
**Total weekly risk:** $270-$1,230 (27-123% of bankroll)

**Why this is safe:**
- Bets happen across 5 days (not all at once)
- Win/loss variance smooths out
- Confidence-based sizing (bet less on lower confidence)
- Kelly Criterion prevents over-betting

**Expected Weekly Results (at 53% win rate):**
- Wins: 14-22 bets
- Losses: 13-19 bets
- Net: +1 to +3 bets per week
- **Weekly profit: $10-90 (1-9% ROI)**

**Over 13-week season:**
- Expected profit: $130-$1,170
- **Season ROI: 13-117%** (if models have edge)

---

## 📈 Tracking Daily Performance

### **Create Daily Log:**

```csv
Date,Day,Games_Analyzed,Bets_Placed,Avg_Edge,Total_Stake,Result,Profit
2025-09-02,Tuesday,8,3,12.5%,$60,2-1,+$91
2025-09-03,Wednesday,7,2,8.2%,$40,1-1,-$5
2025-09-04,Thursday,18,6,9.8%,$120,4-2,+$164
2025-09-05,Friday,23,10,7.5%,$200,6-4,+$82
2025-09-06,Saturday,64,15,6.1%,$300,8-7,+$9
WEEKLY,TOTAL,120,36,8.4%,$720,21-15,+$341
```

**Key Metrics:**
- **Best day:** Tuesday/Wednesday (highest edge %)
- **Most volume:** Saturday (most bets)
- **Best ROI:** Typically Thursday (balance of edge + volume)

---

## 🎯 Bottom Line: NCAA Daily Betting Advantage

### **NFL Betting:**
- 16 games per week
- Mostly Sunday
- 1-2 days of action
- Lines are SHARP (everyone bets NFL)

### **NCAA Betting:**
- 85-135 games per week
- **5 days of action** (Tue-Sat)
- More opportunities = more edge
- Midweek lines softer (less attention)

**NCAA gives you:**
- 5X more betting days
- 5-8X more game volume
- Better edge on midweek games
- More opportunities to find value

---

## 🚀 Your Daily Routine (During Season)

**Every Morning (9 AM):**
1. Run: `python ncaa_daily_predictions.py YOUR_API_KEY`
2. Review top opportunities
3. Place bets on 5%+ edge games
4. Set alerts for kickoff times
5. Track results after games

**Takes:** 10-15 minutes per day

**Weekly time commitment:** 75-100 minutes (less than 2 hours!)

**Expected ROI:** 5-10% per season (if models have edge)

---

## ✅ You're Ready for Daily NCAA Action!

Your system is built for this:
- ✅ Checks games in next 48 hours
- ✅ Groups by day (Tue/Wed/Thu/Fri/Sat)
- ✅ Calculates edge for each game
- ✅ Prioritizes best opportunities
- ✅ Saves predictions for tracking

**Run it daily and print money Tuesday through Saturday!** 💰🏈
