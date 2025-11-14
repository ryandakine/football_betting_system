# NCAA World Models Guide

## Overview

This system implements two advanced world models for NCAA football betting predictions:

1. **InteractionWorldModel** - Learns 2-way and 3-way interactions between your 12 prediction models
2. **NCAA Causal Discovery** - Discovers actual cause-effect relationships in 10 years of NCAA data

Together, these systems boost prediction confidence by 3-8% when known patterns align.

---

## Table of Contents

- [Quick Start](#quick-start)
- [InteractionWorldModel](#interactionworldmodel)
- [NCAA Causal Discovery](#ncaa-causal-discovery)
- [Integration Examples](#integration-examples)
- [Understanding the Output](#understanding-the-output)
- [Training & Maintenance](#training--maintenance)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install causal-learn dowhy numpy pandas

# Verify installation
python3 -c "from college_football_system.core.interaction_world_model import InteractionWorldModel; print('✅ Ready')"
```

### Basic Usage

```python
from college_football_system.core.interaction_world_model import InteractionWorldModel
from college_football_system.causal_discovery.ncaa_causal_learner import NCAACousalLearner

# Initialize both systems
interaction_model = InteractionWorldModel()
causal_model = NCAACousalLearner()

# Your 12 model predictions for a game
model_predictions = {
    'spread_ensemble': 0.72,
    'total_ensemble': 0.68,
    'moneyline_ensemble': 0.75,
    'ncaa_rf': 0.70,
    'ncaa_gb': 0.71,
    'spread_edges': 0.73,
    'total_edges': 0.67,
    'moneyline_edges': 0.74,
    'Market Consensus': 0.69,
    'Contrarian Model': 0.76,
    'Referee Model': 0.65,
    'Injury Model': 0.68
}

# Base confidence from R1/meta-learner
base_confidence = 0.70

# Apply interaction boost
boosted_conf, boost_details = interaction_model.boost_prediction(
    base_confidence,
    model_predictions
)

print(f"Base: {base_confidence:.1%}")
print(f"Boosted: {boosted_conf:.1%}")
print(f"Boost: {boost_details['total_boost']:.1%}")
```

---

## InteractionWorldModel

### What It Does

Discovers when your 12 models **agree strongly** and boosts confidence accordingly.

**Example:**
- If `spread_ensemble` and `ncaa_rf` both predict 72% → likely a strong signal
- If `spread_ensemble`, `ncaa_rf`, AND `Market Consensus` all agree → even stronger

### How It Learns

The model automatically learns from your prediction history:

1. **Record predictions** - Every game prediction is logged
2. **Detect patterns** - Every 10 games with results, it analyzes correlations
3. **Build interactions** - Models with >65% correlation = interaction discovered
4. **Apply boosts** - Future predictions get boosted when interactions fire

### Configuration

```python
# Default cache location
interaction_model = InteractionWorldModel(
    cache_path="models/interaction_world_model.pkl"
)

# Custom model names (if you add/remove models)
interaction_model.model_names = [
    'your_custom_model_1',
    'your_custom_model_2',
    # ... up to 12 models
]
```

### Recording Predictions

**IMPORTANT:** You must record predictions for the system to learn!

```python
# After making a prediction
interaction_model.record_prediction_batch(
    game_id="OKST_vs_TCU_2024_W12",
    model_predictions=model_predictions,
    actual_result=None  # Set to 1 (win) or 0 (loss) when known
)

# Update with actual result later
interaction_model.record_prediction_batch(
    game_id="OKST_vs_TCU_2024_W12",
    model_predictions=model_predictions,
    actual_result=1  # Home team won
)
```

### Boost Details

```python
boosted_conf, details = interaction_model.boost_prediction(
    base_confidence=0.70,
    model_predictions=model_predictions
)

# Details structure:
{
    'base_confidence': 0.70,
    'final_confidence': 0.745,
    'total_boost': 0.045,  # 4.5% boost
    'interaction_count': 2,
    'boosts_applied': [
        {
            'type': '3-way',
            'models': ('spread_ensemble', 'ncaa_rf', 'Market Consensus'),
            'boost': 0.032,
            'strength': 0.78
        },
        {
            'type': '2-way',
            'models': ('Contrarian Model', 'Referee Model'),
            'boost': 0.013,
            'strength': 0.68
        }
    ]
}
```

### Active Interactions

Check which interactions are currently firing:

```python
active = interaction_model.get_active_interactions(model_predictions)

print(f"2-way interactions: {len(active['2way'])}")
print(f"3-way interactions: {len(active['3way'])}")

for interaction in active['3way']:
    print(f"  {interaction['models']} - Strength: {interaction['strength']:.2f}")
```

### Statistics

```python
stats = interaction_model.to_dict()

print(f"Total 2-way interactions learned: {stats['interactions_2way_count']}")
print(f"Total 3-way interactions learned: {stats['interactions_3way_count']}")
print(f"Prediction history size: {stats['history_size']}")
print("\nTop 2-way interactions:")
for interaction in stats['top_2way']:
    print(f"  {interaction['models']} - Freq: {interaction['frequency']}")
```

---

## NCAA Causal Discovery

### What It Does

Discovers **actual cause-effect relationships** in NCAA game data, not just correlations.

**Example Discoveries:**
- `temperature` → `passing_yards` → `total_score` (weather affects passing)
- `key_injury` → `defensive_efficiency` → `allowed_points` (injuries impact defense)
- `rest_days` → `injury_risk` → `team_performance` (rest prevents injuries)

### Running Causal Discovery

**One-time setup** on your historical data:

```python
from college_football_system.causal_discovery.ncaa_causal_learner import discover_causality_from_file

# Run on 10 years of NCAA data (2015-2024)
results = discover_causality_from_file(
    csv_path='data/football/historical/ncaaf/ncaa_combined_2015_2024.csv',
    output_dir='models/causal_models',
    method='pc'  # or 'ges' for more conservative
)

print(f"Found {results['edge_count']} direct causal edges")
print(f"Found {results['confounder_count']} confounders")
print(f"Found {results['mediator_count']} mediation paths")
```

### Algorithms

**PC Algorithm (Default - Fast):**
- Constraint-based discovery
- Good for large datasets
- Faster execution (~2-5 min on 10 years)
- Use for exploratory analysis

**GES Algorithm (Conservative):**
- Score-based discovery
- Fewer false positives
- Slower execution (~10-15 min)
- Use for production models

```python
# Run both and compare
pc_results = causal_model.discover_causality(data, method='pc')
ges_results = causal_model.discover_causality(data, method='ges')
```

### Applying Causal Adjustments

```python
# Initialize with cached results
causal_model = NCAACousalLearner(cache_dir='models/causal_models')

# Your game context (feature values)
causal_context = {
    'temperature': -0.5,  # Cold weather (standardized)
    'key_injury': 0.8,    # Important player injured
    'rest_days': 0.3,     # Well-rested team
    'referee_penalty_rate': 0.6  # High penalty rate expected
}

# Apply causal boost
adjusted_conf, adjustment_details = causal_model.apply_causal_adjustments(
    prediction=0.70,
    causal_context=causal_context
)

print(f"Base: 70.0%")
print(f"Causal adjusted: {adjusted_conf:.1%}")
print(f"Adjustments: {adjustment_details['adjustments']}")
```

### Predictive Paths

Get only the causal paths relevant to game outcomes:

```python
predictive_paths = causal_model.get_predictive_paths()

print("Direct causal edges to game outcomes:")
for edge in predictive_paths['causal_edges']:
    print(f"  {edge['predictor']} → {edge['target']}")
    print(f"    Action: {edge['action']}")

print("\nMediation paths:")
for path in predictive_paths['mediation_paths']:
    print(f"  {path['path']}")
    print(f"    Action: {path['action']}")
```

---

## Integration Examples

### Full Confidence Boost Chain

Recommended order: **R1 Base → Calibration → Interaction → Causal**

```python
from college_football_system.core.interaction_world_model import InteractionWorldModel
from college_football_system.causal_discovery.ncaa_causal_learner import NCAACousalLearner

# Initialize systems
interaction_model = InteractionWorldModel()
causal_model = NCAACousalLearner()

# 1. R1 Base Confidence (from your meta-learner)
r1_confidence = 0.70  # 70%

# 2. Calibration boost (high confidence predictions)
if r1_confidence > 0.75:
    calibrated = r1_confidence * 1.05  # 5% boost
else:
    calibrated = r1_confidence

# 3. Interaction boost
interaction_conf, interaction_details = interaction_model.boost_prediction(
    calibrated,
    model_predictions
)

# 4. Causal adjustment
final_conf, causal_details = causal_model.apply_causal_adjustments(
    interaction_conf,
    causal_context
)

# Log the boost chain
print(f"R1 Base:       {r1_confidence:.1%}")
print(f"Calibrated:    {calibrated:.1%} (+{calibrated - r1_confidence:.1%})")
print(f"Interaction:   {interaction_conf:.1%} (+{interaction_conf - calibrated:.1%})")
print(f"Final:         {final_conf:.1%} (+{final_conf - interaction_conf:.1%})")
print(f"Total boost:   +{final_conf - r1_confidence:.1%}")
```

### Production Prediction Pipeline

```python
def predict_game_with_world_models(
    game_data: dict,
    model_predictions: dict,
    r1_confidence: float
) -> dict:
    """
    Full prediction pipeline with world model boosting.
    """
    # Initialize
    interaction_model = InteractionWorldModel()
    causal_model = NCAACousalLearner()

    # Apply boost chain
    calibrated = r1_confidence * (1.05 if r1_confidence > 0.75 else 1.0)

    interaction_conf, interaction_details = interaction_model.boost_prediction(
        calibrated,
        model_predictions
    )

    causal_context = extract_causal_features(game_data)
    final_conf, causal_details = causal_model.apply_causal_adjustments(
        interaction_conf,
        causal_context
    )

    # Record for learning
    interaction_model.record_prediction_batch(
        game_id=game_data['game_id'],
        model_predictions=model_predictions,
        actual_result=None
    )

    return {
        'game_id': game_data['game_id'],
        'final_confidence': final_conf,
        'r1_base': r1_confidence,
        'calibrated': calibrated,
        'interaction_boosted': interaction_conf,
        'total_boost': final_conf - r1_confidence,
        'interaction_details': interaction_details,
        'causal_details': causal_details,
        'active_interactions': interaction_model.get_active_interactions(model_predictions)
    }

def extract_causal_features(game_data: dict) -> dict:
    """Extract causal variables from game data."""
    return {
        'temperature': game_data.get('temperature', 0) / 50 - 1,  # Standardize
        'wind_speed': game_data.get('wind_mph', 0) / 20,
        'key_injury': game_data.get('injury_impact', 0),
        'rest_days': (game_data.get('days_rest', 7) - 7) / 3,
        'referee_penalty_rate': game_data.get('ref_penalty_rate', 0.5)
    }
```

---

## Understanding the Output

### Prediction Output Format

```json
{
  "game_id": "OKST_vs_TCU_2024_W12",
  "final_confidence": 0.785,
  "r1_base": 0.70,
  "calibrated": 0.735,
  "interaction_boosted": 0.762,
  "total_boost": 0.085,
  "interaction_details": {
    "boosts_applied": [
      {
        "type": "3-way",
        "models": ["spread_ensemble", "ncaa_rf", "Market Consensus"],
        "boost": 0.032,
        "strength": 0.78
      }
    ],
    "interaction_count": 2
  },
  "causal_details": {
    "adjustments": [
      {
        "variable": "temperature",
        "value": -0.5,
        "adjustment": 0.023
      }
    ],
    "total_adjustment": 0.023
  }
}
```

### When to Bet

**Confidence Thresholds:**
- **0.78+** - Strong bet (interaction + causal alignment)
- **0.75-0.78** - Good bet (some boosting present)
- **0.70-0.75** - Moderate bet (base confidence only)
- **< 0.70** - Pass (insufficient edge)

**Interaction Count:**
- **3+ interactions** - Very strong signal (models aligned)
- **1-2 interactions** - Good signal
- **0 interactions** - Rely on base confidence

**Causal Adjustment:**
- **> 0.05** - Significant causal factors present
- **0.02-0.05** - Moderate causal influence
- **< 0.02** - Minimal causal impact

---

## Training & Maintenance

### Auto-Training Schedule

The InteractionWorldModel learns automatically:

1. **Every prediction** - Records model outputs
2. **Every 10 results** - Analyzes patterns and updates interactions
3. **Saves automatically** - Caches learned patterns to disk

**No manual training required!**

### Manual Causal Discovery Updates

Run causal discovery when:
- ✅ New season starts (e.g., 2025 data added)
- ✅ Significant rule changes occur
- ✅ Every 6-12 months for refresh

```bash
# Update causal models with new data
python3 -c "
from college_football_system.causal_discovery.ncaa_causal_learner import discover_causality_from_file
results = discover_causality_from_file('data/ncaa_2015_2025.csv')
print(f'Updated: {results[\"edge_count\"]} causal edges')
"
```

### Cache Management

**Interaction Model Cache:**
```bash
# Location
models/interaction_world_model.pkl

# View size
ls -lh models/interaction_world_model.pkl

# Clear cache (forces retraining)
rm models/interaction_world_model.pkl
```

**Causal Model Cache:**
```bash
# Location
models/causal_models/causal_graph.pkl
models/causal_models/causal_paths.json

# View cached relationships
cat models/causal_models/causal_paths.json

# Clear cache (forces rediscovery)
rm -rf models/causal_models/
```

### Performance Monitoring

Track boost effectiveness:

```python
# Log predictions with boost amounts
prediction_log = []

for game in games:
    result = predict_game_with_world_models(game, models, r1_conf)

    prediction_log.append({
        'game_id': game['game_id'],
        'boost': result['total_boost'],
        'final_conf': result['final_confidence'],
        'actual_result': None  # Update later
    })

# After results known
accuracy_by_boost = {}
for pred in prediction_log:
    boost_tier = 'high' if pred['boost'] > 0.05 else 'low'
    # Calculate accuracy for each tier
```

---

## Troubleshooting

### Issue: No Interactions Found

**Symptoms:** `interaction_count` always 0

**Causes:**
- Not enough prediction history (need 10+ predictions)
- Models not agreeing (all predictions too different)
- Cache not loading

**Solution:**
```python
# Check stats
stats = interaction_model.to_dict()
print(f"History size: {stats['history_size']}")
print(f"2-way interactions: {stats['interactions_2way_count']}")

# If history_size < 10, keep recording predictions
# If interactions_2way_count = 0 after 50+ predictions, models may be too independent
```

### Issue: Causal Discovery Fails

**Symptoms:** Error during `discover_causality()`

**Causes:**
- Missing dependencies
- Data format issues
- Insufficient data variance

**Solution:**
```bash
# Verify dependencies
pip install causal-learn dowhy --upgrade

# Check data format
python3 -c "
import pandas as pd
data = pd.read_csv('your_data.csv')
print(data.shape)
print(data.columns)
print(data.isnull().sum())
"
```

### Issue: Boost Too High/Low

**Symptoms:** Confidence jumping to 95%+ or no boost

**High Boost Fix:**
```python
# Adjust boost multipliers
interaction_model.interactions_2way[key].prediction_boost = 0.03  # Lower from 0.05
interaction_model.interactions_3way[key].prediction_boost = 0.05  # Lower from 0.08
```

**Low Boost Fix:**
```python
# Check if interactions are active
active = interaction_model.get_active_interactions(model_predictions)
if not active['2way'] and not active['3way']:
    print("No active interactions - models disagreeing")
    # Models need to agree within 15% for interaction to fire
```

### Issue: Slow Causal Discovery

**Symptoms:** Takes >30 minutes

**Solution:**
```python
# Use PC instead of GES
results = causal_model.discover_causality(data, method='pc')  # Much faster

# Or reduce data size
data_sample = data.sample(n=10000)  # Use 10k games instead of all
```

---

## Advanced Topics

### Custom Interaction Thresholds

```python
# Modify agreement threshold (default 15%)
def _check_interaction_active_custom(self, models_tuple, predictions):
    confidences = [predictions[m] for m in models_tuple if m in predictions]
    if len(confidences) != len(models_tuple):
        return False
    conf_range = max(confidences) - min(confidences)
    return conf_range < 0.10  # Stricter (10% instead of 15%)
```

### Feature Engineering for Causal Discovery

```python
# Prepare NCAA data with relevant features
causal_features = pd.DataFrame({
    'temperature': game_data['temp'],
    'wind_speed': game_data['wind'],
    'home_rest': game_data['home_days_rest'],
    'away_rest': game_data['away_days_rest'],
    'home_injuries': game_data['home_injury_count'],
    'away_injuries': game_data['away_injury_count'],
    'spread': game_data['spread'],
    'total': game_data['total'],
    'home_score': game_data['home_final_score'],
    'away_score': game_data['away_final_score'],
    'cover': (game_data['home_final_score'] - game_data['away_final_score'] > game_data['spread']).astype(int)
})

results = causal_model.discover_causality(causal_features)
```

---

## FAQ

**Q: How many predictions before interactions are learned?**
A: Minimum 10 predictions with results. Optimal: 50-100 for stable interactions.

**Q: Can I use only one world model (not both)?**
A: Yes! They're independent. Use InteractionWorldModel alone or CausalDiscovery alone.

**Q: Do I need to retrain daily?**
A: No. InteractionWorldModel trains automatically. CausalDiscovery runs once per season.

**Q: What if my models change?**
A: Update `interaction_model.model_names` list. Old interactions will be ignored, new ones learned.

**Q: How accurate are the causal relationships?**
A: PC algorithm: ~70-80% precision. GES algorithm: ~80-90% precision. Validate with domain knowledge.

**Q: Can I visualize the causal graph?**
A: Yes! See `college_football_system/causal_discovery/visualize_graph.py` (coming soon)

---

## References

- **Causal-Learn Documentation:** https://causal-learn.readthedocs.io/
- **DoWhy Library:** https://microsoft.github.io/dowhy/
- **PC Algorithm Paper:** Spirtes et al. (2000) - Causation, Prediction, and Search
- **GES Algorithm Paper:** Chickering (2002) - Optimal Structure Identification

---

## Support

For issues or questions:
1. Check this guide first
2. Verify dependencies installed
3. Check cache file permissions
4. Review prediction logs

**System Status:**
```bash
python3 -c "
from college_football_system.core.interaction_world_model import InteractionWorldModel
from college_football_system.causal_discovery.ncaa_causal_learner import NCAACousalLearner
print('✅ World Models Ready')
"
```

---

**Last Updated:** 2025-11-14
**Version:** 1.0.0
**Compatible with:** NCAA 12-Model System
