# 🏀 WNBA Betting Intelligence System

Advanced betting intelligence system specifically designed for the Women's National Basketball Association (WNBA).

## Features

- **AI-Powered Analysis**: 5-AI Council with specialized agents for professional women's basketball
- **Game Prioritization**: Intelligent ordering based on team strength and market liquidity
- **Social Sentiment**: Real-time social media sentiment analysis (WNBA has strong social presence)
- **Parlay Optimization**: Smart parlay construction optimized for smaller game slates
- **Real-Time Monitoring**: Live odds tracking and alert system
- **Performance Tracking**: Comprehensive bet tracking and ROI analysis

## Quick Start

```python
from wnba_system.main_analyzer import UnifiedWNBAAnalyzer

# Initialize analyzer with $50,000 bankroll
analyzer = UnifiedWNBAAnalyzer(bankroll=50000.0)

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
   - Team-based weighting
   - Edge potential scoring
   - Market liquidity consideration

3. **Social & Weather Analyzer** (`social_weather_analyzer.py`)
   - Social sentiment tracking (WNBA has engaged fanbase)
   - Minimal weather impact (indoor sport)

4. **Parlay Optimizer** (`parlay_optimizer.py`)
   - Smart parlay construction for smaller slates
   - Conservative correlation penalties
   - Kelly criterion sizing

5. **Real-Time Monitor** (`realtime_monitor.py`)
   - Live odds tracking
   - Line movement alerts
   - In-game updates

## Team Weights

WNBA top teams (default weights):

- **Las Vegas Aces**: 0.95
- **New York Liberty**: 0.92
- **Seattle Storm**: 0.88
- **Minnesota Lynx**: 0.88
- **Connecticut Sun**: 0.85
- **Phoenix Mercury**: 0.82

## API Configuration

The system uses the Odds API sport key: `basketball_wnba`

Required API keys:
- Odds API
- Anthropic (optional, for enhanced AI)
- OpenAI (optional, for enhanced AI)

## Usage Examples

### Analyze Today's Games

```python
import asyncio
from wnba_system.main_analyzer import UnifiedWNBAAnalyzer

async def analyze_today():
    analyzer = UnifiedWNBAAnalyzer(bankroll=50000.0)
    results = await analyzer.run_complete_analysis()

    print(f"High Edge Games: {len(results['high_edge_games'])}")
    print(f"Best Parlays: {len(results['parlays'])}")

    return results

asyncio.run(analyze_today())
```

### Prioritize Games by Edge

```python
from wnba_system.game_prioritization import GamePrioritizer

prioritizer = GamePrioritizer()
games = [...]  # Your game list
prioritized = prioritizer.optimize_processing_order(games)

# View prioritization reasoning
for item in prioritizer.get_prioritization_log()[:5]:
    print(f"{item['matchup']}: Priority {item['priority']:.4f}")
    print(f"  Reasons: {', '.join(item['reasons'][:3])}")
```

## WNBA-Specific Features

### Smaller Game Slates
- WNBA typically has 3-6 games per day during season
- System optimized for thorough analysis of limited games
- Conservative parlay limits (2-3 legs max)

### Social Media Engagement
- WNBA has highly engaged fan base on social media
- Social sentiment analysis weighted higher than other sports
- Player-specific sentiment tracking

### Season Structure
- Regular season: May - September
- Playoffs: September - October
- All-Star break: Mid-July
- Olympic breaks (every 4 years)

## Performance Metrics

The system tracks:
- Edge per game
- Confidence scores
- ROI by team matchup
- Win rate by bet type
- Parlay performance
- Social sentiment correlation

## Best Practices

1. **Bankroll Management**: Never risk more than 2.5% on a single game
2. **Edge Threshold**: Only bet games with edge ≥ 6% (higher than college)
3. **Parlay Limits**: Stick to 2-3 leg parlays due to smaller slate
4. **Team Focus**: Prioritize games with elite teams for consistency
5. **Social Monitoring**: Track player/team social media for injury/sentiment updates

## WNBA Season Schedule

- **Pre-Season**: April - Early May
- **Regular Season**: Mid-May through Mid-September (~40 games per team)
- **Commissioner's Cup**: June (in-season tournament)
- **All-Star Break**: Mid-July
- **Playoffs**: Late September - Mid-October (best-of-3 and best-of-5 series)

## Integration

Works seamlessly with:
- Performance Tracker
- Meta Learner
- Crew Prediction Engine (referee analysis)
- CloudGPU AI Ensemble

## Key Differences from College Basketball

1. **Smaller slate**: Fewer games = more thorough analysis per game
2. **Higher skill level**: More consistent performance = tighter lines
3. **Better data**: Professional league has more comprehensive stats
4. **Social presence**: WNBA has strong social media engagement
5. **Referee consistency**: Professional refs = more predictable officiating

## License

MIT License - Part of the Unified Betting Intelligence System
