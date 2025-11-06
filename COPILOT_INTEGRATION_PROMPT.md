# Copilot Integration Prompt

**Paste this into GitHub Copilot to wire in all 7 components:**

---

You are integrating a unified AI Council betting system. Use these exact files and integration points:

## Current Foundation
- File: `ai_council_narrative_unified.py`
- Already has: Narratives, Sentiment, Referees integrated
- Method to extend: `make_unified_prediction(game_data)`

## 7 Components to Wire In

### 1. Agent Influence Engine
```
File: agent_influence_engine.py
Import: from agent_influence_engine import AgentInfluenceEngine
Init: self.agent_engine = AgentInfluenceEngine() in __init__
Usage: adjustments = self.agent_engine.compute_adjustments(game_data)
Apply: 
  - final_edge *= adjustments['edge_multiplier']
  - final_confidence += adjustments['confidence_delta']
```

### 2. Crew Predictive Model
```
File: crew_predictive_model.py
Load: pickle.load(open('data/referee_conspiracy/crew_prediction_model.pkl', 'rb'))
Usage: crew_margin = predict_game_margin(model, crew_id, team_id, week, year, encoders)
Apply: Add crew_margin_adjustment to spread_prediction['adjusted_line']
```

### 3. Parlay Optimizer
```
File: nfl_system/parlay_optimizer.py
Import: from nfl_system.parlay_optimizer import NFLParlayOptimizer
Init: self.parlay_optimizer = NFLParlayOptimizer() in __init__
Usage: parlays = self.parlay_optimizer.optimize_parlays(predictions, bankroll=1000)
Apply: Add to recommendation['secondary_plays'] as parlay suggestions
```

### 4. Meta-Learner
```
File: meta_learner.py
Purpose: Adapt model weights based on recent performance
Apply: weight_adjustment = meta_learner.get_weights()
Use: Multiply each model's confidence by its weight
```

### 5. Crew Betting Strategy
```
File: crew_betting_strategy.py
Import: from crew_betting_strategy import CrewBettingStrategy
Usage: strategy_signal = crew_strategy.get_recommendation(game_data)
Apply: Add to edge_signals if strategy triggers
```

### 6. Backtesting
```
File: backtesting/referee_backtest.py
Import: from backtesting.referee_backtest import validate_prediction
Usage: validation = validate_prediction(prediction, historical_data)
Apply: Adjust confidence based on backtest confidence_adjustment
```

### 7. Monitoring
```
File: record_game_result.py
Import: from record_game_result import record_result
Usage: record_result(game_id, prediction, actual_result)
Apply: Log all predictions for future model improvement
```

## Integration Pattern

For each component:
1. Add import at top of file
2. Initialize in `__init__(self)` 
3. Call in `make_unified_prediction(game_data)`
4. Apply results to final prediction

## Output Structure

Final prediction should include:
```python
{
    'game_id': str,
    'predictions': {spread, total, ml},
    'narrative': NarrativeContext,
    'sentiment': SentimentContext,
    'referee': RefereeContext,
    'agent_adjustments': dict,  # NEW
    'crew_bias': float,  # NEW
    'recommendation': dict,
    'parlays': list,  # NEW
    'strategy_signals': list,  # NEW
    'confidence': float,
    'edge_signals': list,
    'risk_level': str
}
```

## Task

Generate the complete updated `ai_council_narrative_unified.py` with all 7 components integrated. Maintain the existing code structure and add the new components following the pattern above.

---

**How to use:**
1. Copy this prompt
2. Open `ai_council_narrative_unified.py` in VS Code
3. Open GitHub Copilot chat (Ctrl+Shift+I)
4. Paste the prompt
5. Copilot will generate the integrated file
6. Review and apply the changes
