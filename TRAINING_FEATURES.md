# 🏈 NFL AI Council Training Features

## Complete Feature Set for 10-Year Historical Training

### 📊 Basic Game Features
- **Teams**: Home/Away team encodings
- **Week**: Early season (1-6), Mid (7-12), Late (13+)
- **Spread/Total**: Line values, large spreads, high/low totals
- **Venue**: Stadium, attendance, neutral site

### 🌤️ Weather Features
- **Temperature**: Cold weather games (<32°F)
- **Wind Speed**: High wind impact (>15 mph)
- **Precipitation**: Rain/Snow conditions
- **Dome Detection**: Indoor stadiums (no weather impact)
- **Weather Impact Score**: Combined weather factor

### 🏥 Injury Features
- **Position-Weighted Injuries**:
  - QB injuries: 3.0x weight
  - RB/WR injuries: 1.5x weight
  - Other positions: 1.0x weight
- **Injury Status**: Out (1.0), Doubtful (0.7), Questionable (0.3)
- **Team Injury Scores**: Home vs Away
- **Injury Differential**: Advantage calculation

### 😴 Rest & Fatigue Features
- **Rest Days**: Days since last game
- **Short Rest**: <6 days (Thursday games)
- **Rest Advantage**: Home vs Away differential
- **Bye Week Detection**: Full week off

### ✈️ Travel Features
- **Travel Distance**: Away team miles traveled
- **Cross-Country Games**: >2000 miles
- **Travel Fatigue Score**: Distance/1000
- **Time Zone Changes**: East/West coast games

### 🏁 Referee Crew Features
- **Crew Home Bias**: Historical home team penalty rate (0.48-0.55)
- **Penalties Per Game**: Flag-heavy vs lenient crews (11.9-15.1)
- **Crew Variance**: Consistency score
- **Flag Heavy Crews**: >14 penalties/game
- **Home Favoring Crews**: >53% home bias

**Top Ref Crews by Home Bias:**
1. Shawn Smith (55%) - 15.1 penalties/game
2. Brad Allen (54%) - 14.2 penalties/game
3. Ron Torbert (53%) - 14.8 penalties/game
4. Clay Martin (53%) - 14.3 penalties/game

### 🎯 Situational Features
- **Division Games**: Rivalry matchups
- **Prime Time**: National TV games
- **Playoff Implications**: Late season importance
- **Historical Matchups**: Head-to-head records

## 🧠 AI Council Models

### 1. Spread Expert
- **Focus**: Against-the-spread predictions
- **Key Features**: Team performance, injuries, referee bias
- **Target Accuracy**: 56-58%

### 2. Total Expert  
- **Focus**: Over/Under predictions
- **Key Features**: Weather, pace, referee flag tendency
- **Target Accuracy**: 54-56%

### 3. Contrarian Model
- **Focus**: Fade public money
- **Key Features**: Line movement, public betting %
- **Target Accuracy**: 52-54%

### 4. Home Advantage Model
- **Focus**: Home field edge
- **Key Features**: Rest, travel, crowd, referee home bias
- **Target Accuracy**: 55-57%

## 📈 Training Data Stats

- **Total Games**: ~2,500+ (10 seasons)
- **Complete Data**: ~2,000+ games with all features
- **Training Set**: 80% (~1,600 games)
- **Test Set**: 20% (~400 games)

## 🎯 Expected Performance

With all features combined:
- **Consensus Accuracy**: 58-62%
- **ROI**: 8-15% (after vig)
- **Sharp Ratio**: >1.5
- **Max Drawdown**: <15%

## 🚀 Usage

```bash
# Collect 10 years of data
python3 collect_historical_nfl.py

# Train AI Council
python3 train_ai_council.py

# Deploy to Lambda
aws lambda update-function-code --function-name FootballBacktestAnalyzer ...
```

## 📊 Feature Importance (Estimated)

1. **Team Performance** (30%)
2. **Referee Crew Bias** (15%)
3. **Injuries** (15%)
4. **Weather** (12%)
5. **Rest/Fatigue** (10%)
6. **Travel** (8%)
7. **Situational** (10%)

---

**The AI Council learns from 10 years of these features to make informed predictions on current games.**
