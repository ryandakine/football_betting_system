# Comprehensive MLB Betting Data Sources

## 🎯 **Data Sources for Maximum Predictive Power**

### **1. 📺 YouTube Analysis**
**What to collect:**
- Daily picks videos (sentiment analysis)
- Injury reports and updates
- Team news and lineup changes
- Public confidence indicators ("lock", "guaranteed", etc.)
- Most mentioned teams and players
- Betting community sentiment

**Search terms:**
- "MLB daily picks today"
- "MLB injury updates"
- "MLB lineup changes"
- "MLB betting picks"

### **2. 📱 Reddit Analysis**
**Subreddits to monitor:**
- r/baseball
- r/MLB
- r/sportsbook
- r/fantasybaseball
- r/baseballcirclejerk

**What to collect:**
- Injury discussions and rumors
- Lineup speculation
- Weather concerns
- Public betting sentiment
- Team performance discussions
- Player hot/cold streaks

### **3. 🐦 Twitter/X Analysis**
**Accounts to follow:**
- MLB official accounts
- Team beat writers
- Injury reporters
- Betting analysts
- Weather accounts

**What to collect:**
- Real-time injury updates
- Lineup announcements
- Weather reports
- Breaking news
- Public sentiment trends

### **4. 📰 Sports News APIs**
**Sources:**
- ESPN API
- Sports Illustrated
- MLB.com
- Local team news

**What to collect:**
- Official injury reports
- Lineup confirmations
- Weather forecasts
- Team news and updates
- Player statistics

### **5. 🌤️ Weather Data**
**What to collect:**
- Game-time weather conditions
- Wind speed and direction
- Temperature
- Humidity
- Rain probability

### **6. 📊 Advanced Analytics**
**What to collect:**
- Player performance trends
- Head-to-head statistics
- Ballpark factors
- Umpire tendencies
- Historical betting patterns

### **7. 💰 Betting Market Data**
**What to collect:**
- Line movements
- Public betting percentages
- Sharp money indicators
- Bookmaker consensus
- Value bet identification

## 🔍 **Key Predictive Factors**

### **Injury Impact Analysis:**
- **Star player injuries** (major impact)
- **Pitcher injuries** (rotation changes)
- **Bullpen availability** (late-game impact)
- **Lineup changes** (offensive production)

### **Public Sentiment Indicators:**
- **Confidence levels** in picks
- **Consensus picks** vs contrarian
- **Sharp money** vs public money
- **Social media buzz** around teams

### **Weather Factors:**
- **Wind direction** (home run impact)
- **Temperature** (ball flight)
- **Humidity** (pitching grip)
- **Rain probability** (game delays)

### **Market Indicators:**
- **Line movement** (sharp money)
- **Public betting percentages** (fade opportunities)
- **Value discrepancies** between books
- **Consensus vs individual picks**

## 🎯 **n8n Workflow Structure**

### **Data Collection Nodes:**
1. **YouTube Search Node** - Find daily picks videos
2. **YouTube Video Details Node** - Get video content
3. **Reddit API Node** - Monitor subreddits
4. **Twitter API Node** - Track mentions and news
5. **Weather API Node** - Get game conditions
6. **News API Node** - Official updates

### **Processing Nodes:**
1. **Code Node - YouTube Analysis** - Extract sentiment, injuries, picks
2. **Code Node - Reddit Analysis** - Community sentiment, injury rumors
3. **Code Node - Twitter Analysis** - Real-time news, sentiment
4. **Code Node - Weather Analysis** - Impact on game conditions
5. **Code Node - Data Fusion** - Combine all sources

### **AI Analysis Node:**
- **OpenAI/Perplexity** - Advanced reasoning
- **Injury impact assessment**
- **Weather factor analysis**
- **Public sentiment vs sharp money**
- **Final betting recommendations**

### **Output Nodes:**
1. **Supabase** - Store all analysis
2. **Slack** - Send alerts
3. **Email** - Daily reports
4. **Dashboard** - Visual analytics

## 🚀 **Implementation Priority**

### **Phase 1 (Immediate):**
- YouTube sentiment analysis
- Basic injury detection
- Public sentiment tracking

### **Phase 2 (Next):**
- Reddit integration
- Weather data
- Advanced injury analysis

### **Phase 3 (Advanced):**
- Twitter/X integration
- News API integration
- Machine learning predictions

## 📋 **Next Steps**

1. **Set up YouTube analysis** (already have API key)
2. **Get Reddit API credentials**
3. **Set up weather API**
4. **Create comprehensive n8n workflow**
5. **Test data collection and analysis**
6. **Implement AI reasoning**

Would you like me to start with the YouTube analysis workflow, or do you want to set up Reddit API credentials first?
