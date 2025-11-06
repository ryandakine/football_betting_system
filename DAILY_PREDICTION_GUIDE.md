# Daily Prediction System Guide

## Overview

Your MLB betting system now includes a **Daily Prediction System** that makes predictions on **EVERY MLB game** each day, regardless of betting value. This ensures continuous learning and testing of the system's reasoning, even when there are no good betting opportunities.

## Key Features

### 🎯 **Predictions on Every Game**
- Makes predictions on ALL MLB games daily
- Tests system reasoning on every opportunity
- No wasted learning days

### 📚 **Continuous Learning**
- Learns from every prediction, win or lose
- Tracks accuracy and performance over time
- Identifies patterns in successful predictions

### 💰 **Value Bet Identification**
- Identifies high-value betting opportunities
- Separates learning from actual betting
- Maximizes profit on good opportunities

### 🔄 **Automated Daily Cycle**
- Runs predictions every morning
- Records outcomes every evening
- Weekly model retraining
- Monthly performance reviews

## How It Works

### Daily Prediction Cycle

1. **Morning (9 AM)**: System makes predictions on all MLB games
2. **Throughout Day**: System tracks odds movements and sentiment
3. **Evening (11 PM)**: System records actual game outcomes
4. **Learning**: System learns from prediction accuracy
5. **Improvement**: Models retrain with new insights

### Prediction Process

```
Daily Games → Odds Analysis → Sentiment Analysis → ML Prediction → Value Assessment
     ↓
Learning System → Outcome Recording → Model Retraining → Performance Tracking
```

## Setup Instructions

### Step 1: Install Dependencies

```bash
pip install schedule requests asyncio
```

### Step 2: Configure Environment Variables

Add to your `aci.env`:

```bash
# Daily Prediction System
ODDS_API_KEY=your_odds_api_key_here
YOUTUBE_API_KEY=your_youtube_api_key_here
SLACK_WEBHOOK_URL=your_slack_webhook_url

# Learning System API
LEARNING_API_URL=http://localhost:8000
```

### Step 3: Start the Learning System

```bash
# Start the learning API server
python learning_api_server.py
```

### Step 4: Initialize Daily Prediction System

```bash
# Initialize the daily prediction system
python daily_prediction_system.py
```

### Step 5: Start the Scheduler

```bash
# Start automated daily predictions
python daily_prediction_scheduler.py
```

## Daily Workflow

### Morning Predictions (9 AM)

The system automatically:

1. **Fetches all MLB games** for the day
2. **Gets odds from multiple bookmakers** (FanDuel, DraftKings, BetMGM, etc.)
3. **Analyzes sentiment** from YouTube and Reddit
4. **Makes ML predictions** on every game
5. **Identifies value bets** (edge > 2%, confidence > 60%)
6. **Saves predictions** for learning
7. **Sends daily summary** via Slack

### Evening Outcome Recording (11 PM)

The system automatically:

1. **Fetches game results** from The Odds API
2. **Records actual outcomes** vs predictions
3. **Calculates profit/loss** for learning
4. **Updates learning system** with new data
5. **Retrains models** if needed
6. **Tracks performance** metrics

## Manual Operations

### Run Predictions for Today

```bash
python -c "
from daily_prediction_system import DailyPredictionSystem
from self_learning_system import SelfLearningSystem
import asyncio

learning_system = SelfLearningSystem()
daily_system = DailyPredictionSystem(learning_system, 'your_odds_api_key')

async def main():
    summary = await daily_system.make_daily_predictions()
    print(f'Predictions made: {summary[\"predictions_made\"]}')
    print(f'Value bets found: {summary[\"value_bets_found\"]}')

asyncio.run(main())
"
```

### Run Predictions for Specific Date

```bash
python -c "
from daily_prediction_system import DailyPredictionSystem
from self_learning_system import SelfLearningSystem
import asyncio

learning_system = SelfLearningSystem()
daily_system = DailyPredictionSystem(learning_system, 'your_odds_api_key')

async def main():
    summary = await daily_system.make_daily_predictions('2025-01-07')
    print(f'Predictions for 2025-01-07: {summary[\"predictions_made\"]}')

asyncio.run(main())
"
```

### Record Outcomes for Date

```bash
python -c "
from daily_prediction_system import DailyPredictionSystem
from self_learning_system import SelfLearningSystem
import asyncio

learning_system = SelfLearningSystem()
daily_system = DailyPredictionSystem(learning_system, 'your_odds_api_key')

async def main():
    await daily_system.record_daily_outcomes('2025-01-07')
    print('Outcomes recorded for 2025-01-07')

asyncio.run(main())
"
```

## Enhanced n8n Workflow

### Import Enhanced Workflow

1. Import `enhanced_n8n_workflow_with_daily_predictions.json` to your n8n cloud
2. Configure the "Learning System Prediction" node:
   - URL: `http://localhost:8000/learning/predict`
   - Method: POST
   - Headers: `Content-Type: application/json`

3. Configure the "Learning System Integration" node:
   - URL: `http://localhost:8000/learning/process`
   - Method: POST
   - Headers: `Content-Type: application/json`

### Workflow Features

- **Daily Game Fetching**: Gets all MLB games for the day
- **Multi-Book Odds**: Compares odds across 5 bookmakers
- **Sentiment Analysis**: YouTube and Reddit sentiment
- **ML Predictions**: Learning system predictions on every game
- **Value Bet Identification**: Finds high-value opportunities
- **AI Analysis**: OpenAI analysis of daily predictions
- **Database Storage**: Saves to Supabase
- **Learning Integration**: Feeds data to learning system
- **Slack Notifications**: Daily summary alerts

## Monitoring and Analytics

### Daily Summary

```bash
# Get today's prediction summary
curl http://localhost:8000/learning/summary
```

### Performance Metrics

```bash
# Get detailed performance metrics
curl http://localhost:8000/learning/performance
```

### Learning Insights

```bash
# Get learning system insights
curl http://localhost:8000/learning/summary
```

## File Structure

```
mlb_betting_system/
├── daily_prediction_system.py          # Core daily prediction system
├── daily_prediction_scheduler.py       # Automated scheduler
├── enhanced_n8n_workflow_with_daily_predictions.json  # Enhanced n8n workflow
├── predictions/                        # Daily prediction files
│   ├── predictions_2025-01-07.json
│   ├── predictions_2025-01-08.json
│   └── ...
├── logs/                              # System logs
│   ├── daily_summary_2025-01-07.log
│   ├── weekly_summary_2025-01-05_to_2025-01-11.log
│   └── ...
└── data/
    └── learning_system.db             # Learning system database
```

## Expected Daily Output

### Morning Prediction Summary

```
🤖 Daily MLB Predictions Complete

📊 Games Analyzed: 15
🎯 Predictions Made: 15
💰 Value Bets Found: 3
📈 Learning Opportunities: 15

📊 Performance:
- Avg Confidence: 0.67
- Avg Edge: 0.023

🎯 Top Value Bets:
  • NYY vs BOS: home (0.72 confidence, 0.045 edge)
  • LAD vs SF: away (0.68 confidence, 0.038 edge)
  • HOU vs OAK: home (0.65 confidence, 0.032 edge)

📚 Learning System: success
```

### Evening Learning Summary

```
📚 Daily Learning Summary

📊 Today's Performance:
- Predictions: 15
- Correct: 9
- Accuracy: 60.0%
- Profit: $45.20

📈 Overall System Performance:
- Total Predictions: 1,247
- Overall Accuracy: 58.3%
- Overall ROI: 12.7%
- Trend: Improving

🎯 Insights:
- High confidence predictions (≥70%) have 65.2% accuracy
- Positive edge predictions (>2%) have 61.8% accuracy
- Recent 50 predictions: 62.0% accuracy
```

## Benefits

### 🎯 **Continuous Learning**
- System learns from every game, every day
- No wasted opportunities for learning
- Constant improvement in prediction accuracy

### 💰 **Value Bet Identification**
- Only bet when there's clear value
- Avoid betting on low-confidence predictions
- Maximize profit on good opportunities

### 📊 **Performance Tracking**
- Track accuracy over time
- Identify successful patterns
- Monitor ROI and profit trends

### 🔄 **Automated Operation**
- Runs automatically every day
- No manual intervention required
- Consistent data collection and learning

## Troubleshooting

### Common Issues

1. **No Games Found**
   ```bash
   # Check if it's an off-day or API issue
   curl "https://api.the-odds-api.com/v4/sports/baseball_mlb/scores?apiKey=YOUR_KEY&date=2025-01-07"
   ```

2. **Learning System Not Responding**
   ```bash
   # Check if API server is running
   curl http://localhost:8000/health
   ```

3. **Predictions Not Saving**
   ```bash
   # Check file permissions
   ls -la predictions/
   ```

### Logs and Debugging

- **Daily Logs**: Check `logs/daily_summary_YYYY-MM-DD.log`
- **Error Logs**: Check `logs/errors.log`
- **API Logs**: Check console output from `learning_api_server.py`

## Next Steps

1. **Start the System**: Follow setup instructions above
2. **Run Manual Test**: Test predictions for today
3. **Monitor Performance**: Track daily accuracy and ROI
4. **Optimize Strategy**: Use insights to improve betting approach
5. **Scale Up**: Add more data sources and features

The daily prediction system ensures your MLB betting system is constantly learning and improving, making it smarter and more profitable over time! 🚀
