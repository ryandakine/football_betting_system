# 🏀 Women's College Basketball Betting Intelligence System

Advanced betting intelligence system specifically designed for NCAA Women's Basketball (WCBB).

## Features

- **AI-Powered Analysis**: 5-AI Council with specialized agents for women's basketball
- **Game Prioritization**: Intelligent ordering of games based on edge potential
- **Social Sentiment**: Real-time social media sentiment analysis
- **Parlay Optimization**: Smart parlay construction with correlation penalties
- **Real-Time Monitoring**: Live odds tracking and alert system
- **Performance Tracking**: Comprehensive bet tracking and ROI analysis

## Quick Start

```python
from womens_college_basketball_system.main_analyzer import UnifiedWomensCollegeBasketballAnalyzer

# Initialize analyzer with $50,000 bankroll
analyzer = UnifiedWomensCollegeBasketballAnalyzer(bankroll=50000.0)

# Run complete analysis
import asyncio
results = asyncio.run(analyzer.run_complete_analysis())
```

## System Architecture

### Core Components

1. **Main Analyzer** (`main_analyzer.py`)
   - Unified intelligence system
   - Multi-AI consensus engine
   - Meta-learning integration

2. **Game Prioritizer** (`game_prioritization.py`)
   - Conference-based weighting
   - Edge potential scoring
   - Mid-major opportunity detection

3. **Social & Weather Analyzer** (`social_weather_analyzer.py`)
   - Social sentiment tracking
   - Minimal weather impact (indoor sport)

4. **Parlay Optimizer** (`parlay_optimizer.py`)
   - Smart parlay construction
   - Correlation penalty application
   - Kelly criterion sizing

5. **Real-Time Monitor** (`realtime_monitor.py`)
   - Live odds tracking
   - Line movement alerts
   - In-game updates

## Conference Weights

Women's basketball power conferences (default weights):

- **Big Ten**: 0.95
- **SEC**: 0.92
- **ACC**: 0.88
- **Pac-12**: 0.85
- **Big 12**: 0.82
- **Big East**: 0.80

Mid-major conferences receive bonus weighting for hidden value opportunities.

## API Configuration

The system uses the Odds API sport key: `basketball_ncaaw`

Required API keys:
- Odds API
- Anthropic (optional, for enhanced AI)
- OpenAI (optional, for enhanced AI)

## Usage Examples

### Analyze Today's Games

```python
import asyncio
from womens_college_basketball_system.main_analyzer import UnifiedWomensCollegeBasketballAnalyzer

async def analyze_today():
    analyzer = UnifiedWomensCollegeBasketballAnalyzer(bankroll=50000.0)
    results = await analyzer.run_complete_analysis()

    print(f"High Edge Games: {len(results['high_edge_games'])}")
    print(f"Best Parlays: {len(results['parlays'])}")

    return results

asyncio.run(analyze_today())
```

### Prioritize Games by Edge

```python
from womens_college_basketball_system.game_prioritization import GamePrioritizer

prioritizer = GamePrioritizer()
games = [...]  # Your game list
prioritized = prioritizer.optimize_processing_order(games)

# View prioritization reasoning
for item in prioritizer.get_prioritization_log()[:5]:
    print(f"{item['matchup']}: Priority {item['priority']:.4f}")
    print(f"  Reasons: {', '.join(item['reasons'][:3])}")
```

## Performance Metrics

The system tracks:
- Edge per game
- Confidence scores
- ROI by conference
- Win rate by bet type
- Parlay performance

## Best Practices

1. **Bankroll Management**: Never risk more than 3% on a single game
2. **Edge Threshold**: Only bet games with edge ≥ 5%
3. **Parlay Limits**: Stick to 2-4 leg parlays
4. **Conference Focus**: Prioritize power conferences for consistency
5. **Mid-Major Opportunities**: Don't ignore smaller conferences with high edges

## Integration

Works seamlessly with:
- Performance Tracker
- Meta Learner
- Crew Prediction Engine
- CloudGPU AI Ensemble

## License

MIT License - Part of the Unified Betting Intelligence System
