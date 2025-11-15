# NCAA World Models Guide

## Overview

Your NCAA betting system now includes two advanced "world models" that understand relationships between your 12 prediction models and discover causal patterns in game outcomes.

## InteractionWorldModel

Learns 2-way and 3-way interactions between models. When multiple models agree, confidence boosts automatically.

**Example:**
- Spread Ensemble: 75% confidence
- Moneyline Ensemble: 73% confidence
- Within 15% range → Interaction fires → Confidence boosts to 78%

## CausalDiscovery

Discovers actual cause-effect relationships in NCAA game data using PC/GES algorithms.

**Examples:**
- temperature → passing_yards → total_score
- key_injury → defensive_efficiency → allowed_points

## Quick Start

```python
from college_football_system.core.interaction_world_model import InteractionWorldModel
from college_football_system.causal_discovery.ncaa_causal_learner import NCAACousalLearner

# Initialize models
interaction_model = InteractionWorldModel()
causal_model = NCAACousalLearner()

# Record prediction
interaction_model.record_prediction_batch(
    game_id='game_123',
    model_predictions={'spread_ensemble': 0.75, 'moneyline_ensemble': 0.73},
    actual_result=1
)

# Apply boosts
boosted_conf, details = interaction_model.boost_prediction(0.70, model_predictions)
```

## Production Integration

The production runner automatically:
1. Applies interaction boosts when models align
2. Applies causal adjustments based on game context
3. Shows confidence breakdown: Base → Calibrated → Interactions → Causal → Final

## Data Requirements

- **Interactions**: 10+ predictions with results to start learning
- **Causality**: 50+ games recommended, ideal 1500+ (10 years 2015-2024)

## Monitoring

```python
interaction_status = predictor.interaction_model.to_dict()
causal_status = predictor.causal_learner.to_dict()
```

See college_football_system/core/run_ncaa_predictions_production.py for full integration.
