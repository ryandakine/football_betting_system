# 🏈 Football Betting Master System

**Enterprise-level AI-powered football betting analysis with real data sources and intelligent learning.**

## 🎯 System Overview

This system provides comprehensive football betting analysis using multiple AI providers, real-time data sources, and continuous learning capabilities.

### 🤖 AI Intelligence
- **Premium Providers**: Claude, Perplexity AI, Grok, Gemini, ChatGPT
- **Free Backups**: Ollama (local), HuggingFace (with user permission)
- **Smart Fallbacks**: Automatically switches to backups only with user approval

### 📊 Data Sources
- **The Odds API**: Real FanDuel and sportsbook odds
- **ESPN**: Live NFL/NCAFF scores and game data
- **NFL Official**: Official league statistics and game information

### 🎮 Key Features
- **Predict All Games**: Mass AI analysis of every available game
- **Individual Analysis**: Single-game predictions with detailed reasoning
- **Learning System**: Tracks accuracy and improves over time
- **Mobile Responsive**: Works on any screen size
- **Offline Caching**: Data persists between sessions

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### API Keys Setup
Your `.env` file should contain:
```bash
# The Odds API - Real sportsbook data
THE_ODDS_API_KEY=your_odds_api_key

# Premium AI Providers
CLAUDE_API_KEY=your_claude_key
OPENAI_API_KEY=your_openai_key
PERPLEXITY_API_KEY=your_perplexity_key
GROK_API_KEY=your_grok_key
GOOGLE_API_KEY=your_gemini_key  # Optional
```

### Launch System
```bash
python3 launch_system.py
```

## 🎮 Using the System

### 1. Refresh Data
- Click "🔄 Refresh Data" to load live odds and game data
- System fetches from ESPN, NFL Official, and The Odds API

### 2. Predict Games
- **"🎯 Predict All Games"**: Analyzes every game with AI consensus
- **Individual Games**: Click "🎯 Analyze This Game" on any game card

### 3. View Predictions
- Game cards show AI predictions with confidence levels
- Color-coded: 🟢 High confidence, 🟡 Medium, 🔴 Low
- Fallback usage is clearly indicated

### 4. Learning & Analytics
- System tracks every prediction and learns from outcomes
- Completed games automatically update accuracy metrics
- Performance dashboard shows improvement over time

## 🤖 AI Provider Priority

1. **Primary**: Premium AI providers (your API keys)
2. **Fallback**: Free LLMs (only with user permission)
3. **Transparent**: Clear notifications when fallbacks are used

## 📱 Mobile Support

- Responsive layouts adapt to any screen size
- Touch-friendly controls and gestures
- Progressive loading for mobile networks
- Swipe navigation in mobile mode

## 🔧 Troubleshooting

### API Keys Not Working
- Check `.env` file format
- Ensure API keys are valid and have credits
- System will automatically use fallbacks if premium providers fail

### No Games Showing
- Click "Refresh Data" to load current games
- Check internet connection
- System uses offline cache when APIs are unavailable

### AI Analysis Failing
- Check API key validity
- System will prompt for fallback usage if needed
- HuggingFace works offline once models are downloaded

## 📈 System Architecture

- **Modular Design**: 200+ Python files organized by function
- **Real-time Updates**: Live odds and scores
- **Learning System**: Continuous improvement from outcomes
- **Offline Capability**: Caches data for offline use
- **Error Resilience**: Graceful handling of API failures

## 🎯 Advanced Features

- **Parlay Optimization**: EV-based combination analysis
- **Correlation Detection**: Identifies risky bet combinations
- **Performance Tracking**: Detailed analytics and reporting
- **Self-Learning**: Adapts strategies based on results

---

**Built for serious football bettors who demand intelligence, accuracy, and reliability.**