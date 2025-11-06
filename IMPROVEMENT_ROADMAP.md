# NFL Betting System Intelligence Roadmap

## ✅ What We Have Now
- ✅ HRM Beast neural net (referee conspiracy detection)
- ✅ Sniper rule engine
- ✅ Conspiracy bot with narrative scraping
- ✅ Primetime home conspiracy boost
- ✅ Contrarian fade logic (day-of-week aware)
- ✅ Real Reddit sentiment scraping
- ✅ Learning system for tracking results

## 🎯 Priority Improvements

### 1. **Weather Integration** (High Impact)
**Why**: Weather kills overs (wind, rain, snow)
**What to add**:
- [ ] Real-time weather API (OpenWeatherMap)
- [ ] Wind speed thresholds (>15mph = under)
- [ ] Rain/snow detection
- [ ] Temperature extremes (<20°F or >95°F)
- [ ] Dome game detection (ignore weather)

**Expected Impact**: +5-8% accuracy on totals

---

### 2. **Injury Impact Analysis** (High Impact)
**Why**: Key injuries move lines but public overreacts
**What to add**:
- [ ] Starting QB out = major impact
- [ ] Top 2 WRs out = under bias
- [ ] O-line injuries = under bias
- [ ] Defensive injuries = over bias
- [ ] Scrape injury reports from ESPN/FantasyPros

**Expected Impact**: +3-5% accuracy

---

### 3. **Team Pace/Style Analysis** (Medium Impact)
**Why**: Some teams play fast (over), some play slow (under)
**What to add**:
- [ ] Offensive pace (plays per game)
- [ ] Defensive style (bend-don't-break vs aggressive)
- [ ] Run-heavy vs pass-heavy
- [ ] Time of possession stats
- [ ] Red zone efficiency

**Expected Impact**: +4-6% accuracy

---

### 4. **Line Movement Tracking** (High Impact - Sharp Money)
**Why**: Sharp money moves lines, public follows
**What to add**:
- [ ] Track opening line vs current line
- [ ] Detect reverse line movement (RLM)
  - Line moves AGAINST public money = sharps on other side
- [ ] Steam moves (sudden 1+ point move)
- [ ] Bet percentage vs line movement divergence

**Expected Impact**: +6-10% accuracy (HUGE)

---

### 5. **Historical Matchup Data** (Medium Impact)
**Why**: Division rivals play differently
**What to add**:
- [ ] Head-to-head history (last 5 games)
- [ ] Division game detection (lower scoring)
- [ ] Revenge game detection (QB playing old team)
- [ ] Playoff implications (intensity factor)

**Expected Impact**: +3-5% accuracy

---

### 6. **Referee Crew Analysis** (Medium Impact - You Already Have This!)
**Why**: Some refs call more penalties than others
**What to add** (enhance existing):
- [ ] Ref crew historical under/over record
- [ ] Flag-happy refs = under (more stoppages)
- [ ] Ref home bias detection
- [ ] Ref primetime performance

**Expected Impact**: +2-4% accuracy

---

### 7. **Situational Spots** (High Impact - Smart Money)
**Why**: Sharps exploit specific situations
**What to add**:
- [ ] **Look-ahead spot**: Team plays big rival next week
- [ ] **Sandwich game**: Between 2 tough opponents
- [ ] **Short rest**: Thursday after Monday/Sunday Night
- [ ] **Travel fatigue**: West Coast → East Coast early game
- [ ] **Divisional home game after loss**: Desperate team

**Expected Impact**: +5-8% accuracy

---

### 8. **Market Efficiency Detection** (Advanced)
**Why**: Not all lines are sharp
**What to add**:
- [ ] Compare consensus across 10+ sportsbooks
- [ ] Detect soft lines (outliers)
- [ ] Identify which books are sharp vs square
- [ ] Arbitrage opportunity detection

**Expected Impact**: +3-5% win rate

---

### 9. **Correlation Analysis** (Advanced)
**Why**: Certain factors compound
**What to add**:
- [ ] Weather + slow pace = STRONG under
- [ ] Primetime + division rival = unpredictable
- [ ] Injury + road game = amplified effect
- [ ] Public hype + sharp fade = max value

**Expected Impact**: +4-6% accuracy

---

### 10. **Live Game Adjustments** (Advanced)
**Why**: Lines move during games
**What to add**:
- [ ] Live total tracking
- [ ] Halftime model adjustments
- [ ] Momentum detection
- [ ] Second half predictions

**Expected Impact**: New betting opportunities

---

## 🚀 Quick Wins (Can Do Now)

### A. **Enhanced Narrative Detection**
Add these storyline keywords to scraper:
- "Must-win game"
- "Playoff race"
- "Revenge game"
- "Coaching hot seat"
- "Bounce-back spot"
- "Trap game"

### B. **Model Weight Learning**
Track which model (Beast/Sniper/Conspiracy) is most accurate:
- By weekday
- By team
- By total range
- Dynamically adjust fusion weights

### C. **Bankroll Management**
Add Kelly Criterion bet sizing:
- High confidence (>75%) = 3% bankroll
- Medium (55-75%) = 1.5% bankroll
- Low (<55%) = skip or 0.5% bankroll

### D. **Alert System**
Send alerts when:
- Line moves >3 points (steam move)
- Sharp money detected
- Weather alert
- Key injury reported

---

## 📊 Expected Overall Impact

If we add all improvements:
- **Current accuracy**: 53.3% (primetime only)
- **With weather**: 58-61%
- **With injuries**: 61-66%
- **With line movement**: 67-76%
- **With situational spots**: 72-80%

**Target**: 60-65% win rate (very profitable)

---

## 🎯 Recommended Priority Order

1. **Line Movement Tracking** (biggest edge)
2. **Weather Integration** (easy + high impact)
3. **Injury Analysis** (public overreacts = value)
4. **Situational Spots** (sharp money loves these)
5. **Team Pace Analysis** (fundamental factor)
6. **Enhanced Narrative Detection** (improve existing)
7. **Model Weight Learning** (optimize what we have)

---

## 💡 Implementation Notes

- Use **asyncio** for parallel data fetching
- Cache data to avoid rate limits
- Store historical predictions for learning
- Backtest each new feature on 2024 data
- Track feature importance (which signals win)

---

## 🔥 Secret Sauce Ideas

1. **Fade the Fade**: When public is 90%+ on one side on TNF, maybe they're right (sharps ARE the public)
2. **Ref + Weather Combo**: Flag-happy ref + wind = MEGA under
3. **Narrative Reversal**: When storyline is TOO obvious (revenge game), fade it
4. **Line Movement + Weather**: If line drops AND weather turns bad = SMASH under
5. **TNF Survivor**: Teams that win TNF often lose the following Sunday (short rest)

---

Want me to start implementing any of these? I'd recommend **Line Movement Tracking** first as it has the biggest edge.
