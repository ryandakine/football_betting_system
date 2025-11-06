# Unified College Football Betting System

**Single source of truth for all college football betting analysis.**

## Structure

```
college_football_system/
├── main_analyzer.py           # Main orchestrator (ALL features combined)
├── game_prioritization.py     # Conference weights & edge scoring
├── social_weather_analyzer.py # Social sentiment + weather impact
├── parlay_optimizer.py         # Parlay building with correlation analysis
├── realtime_monitor.py         # Live alerts & performance tracking
└── README.md                   # This file
```

## Features

✅ **CloudGPU AI Ensemble** - HuggingFace models  
✅ **5-AI Council** - Specialized betting agents  
✅ **Game Prioritization** - Conference-weighted edge detection  
✅ **Social & Weather** - Sentiment + weather impact  
✅ **Parlay Optimizer** - Correlation-aware parlay building  
✅ **Real-Time Monitor** - Live game tracking with alerts  
✅ **Performance Tracking** - Fake-money bet tracking  
✅ **Backtesting** - Historical validation (integrated)

## Usage

```python
from college_football_system import UnifiedCollegeFootballAnalyzer

analyzer = UnifiedCollegeFootballAnalyzer(bankroll=50000.0)
results = await analyzer.run_complete_analysis()
```

## Quick Start

```bash
# Run full analysis
python -m college_football_system.main_analyzer

# Or use the convenience script
python run_college_today.sh
```

## What Happened to Old Files?

**MERGED INTO THIS SYSTEM:**
- `comprehensive_college_football_analyzer.py` → `main_analyzer.py`
- `college_5ai_council.py` → Integrated into `main_analyzer.py`
- `college_football_today.py` → Use `main_analyzer.py` instead
- `college_football_saturday_monitor.py` → Use `realtime_monitor.py`
- `college_system/` → Renamed to `college_football_system/`

**All features preserved, just better organized!**
