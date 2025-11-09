# 10-Model Super Intelligence System 🚀

## Overview

This NFL betting system uses **10 coordinated AI models** to find betting edges across multiple markets and angles. Each model brings unique insights, and they work together through an enhanced AI Council.

---

## The 10 Models

### **Models 1-3: Base Ensembles** (Voting Classifiers)
1. **Spread Ensemble** - Predicts if home team covers the spread
   - LogisticRegression + RandomForest + GradientBoosting
   - Output: `spread_model_home_pct` (probability home covers)

2. **Total Ensemble** - Predicts if game goes over the total
   - Same ensemble architecture
   - Output: `total_model_over_pct` (probability over hits)

3. **Moneyline Ensemble** - Predicts game winner
   - Same ensemble architecture
   - Output: `home_advantage_pct` (probability home wins)

### **Models 4-6: New Prediction Targets**
4. **First Half Spread Model** - Predicts first half performance
   - Catches teams that start fast/slow
   - Different edge than full game
   - Output: `first_half_spread_home_pct`

5. **Home Team Total Model** - Predicts home team scoring
   - Individual team offensive/defensive matchup
   - Output: `home_team_total_over_pct`

6. **Away Team Total Model** - Predicts away team scoring
   - Individual team offensive/defensive matchup
   - Output: `away_team_total_over_pct`

### **Models 7-8: Algorithm Variants**
7. **XGBoost Super Ensemble** - Gradient boosting beast
   - Often outperforms sklearn on structured data
   - Handles non-linear patterns better
   - Output: `xgboost_model_pct`

8. **Neural Network Deep Model** - 4-layer deep network
   - Finds complex feature interactions
   - 128→64→32→16 architecture with early stopping
   - Output: `neural_net_model_pct`

### **Model 9: Stacking Meta-Learner**
9. **Stacking Meta-Learner** - Model that learns from other models
   - Takes predictions from Models 1-8 as input
   - Learns optimal weighting strategy
   - Gets **1.5x weight** in final ensemble (highest confidence)
   - Output: `stacking_model_pct`

### **Model 10: Situational Specialist**
10. **Situational Specialist** - Rule-based context adjustments
    - Primetime game boost (+15% confidence)
    - Divisional rivalry boost (+10% confidence)
    - Weather adjustments (rain/snow/wind penalties)
    - Rest differential adjustments (short week vs long rest)
    - No training needed - expert-defined rules

---

## How They Work Together

### Confidence Calculation (Enhanced AI Council)
The Enhanced AI Council combines all 10 model outputs using **weighted averaging**:

```python
Model Weights:
- Spread Ensemble: 1.0
- Total Ensemble: 1.0
- Moneyline Ensemble: 1.0
- First Half Spread: 0.8
- Home Team Total: 0.7
- Away Team Total: 0.7
- XGBoost: 1.2 (usually performs well)
- Neural Net: 0.9
- Stacking Meta: 1.5 (learns from others)
```

**Confidence Boosts:**
- Model agreement: Up to +15% for unanimous agreement
- Situational edges: Applied by Model 10
- Sentiment signals: Contrarian scores, crowd roar
- Narrative strength: Trap games, revenge games, etc.

### Edge Detection

The system detects edge signals when:
- ✅ **UNANIMOUS_10_MODEL_EDGE** - All 10 models agree (very rare, very valuable)
- ✅ **STRONG_MODEL_AGREEMENT** - Models tightly clustered (std dev < 0.15)
- ✅ **ALGO_CONSENSUS_STRONG** - XGBoost, NN, and Stacking all confident
- ✅ **FIRST_HALF_EDGE** - First half model very confident
- ✅ **HOME/AWAY_TOTAL_EDGE** - Team total model very confident
- ✅ **PRIMETIME_BOOST** - Situational specialist activated
- ✅ **CONTRARIAN_EDGE** - Sharp vs public divergence

---

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt

# Optional: Install XGBoost for Model 7
pip install xgboost
```

### 2. Prepare Training Data
Your CSV should have these columns (minimum):
```
home_team, away_team, spread, total, home_ml_odds, away_ml_odds,
home_score, away_score, home_cover, over_hit, home_win
```

Optional first half data:
```
home_1h_score, away_1h_score, home_1h_cover
```

Optional team stats:
```
home_score_avg, away_score_avg, home_win_pct, away_win_pct,
home_ats_record, away_ats_record
```

### 3. Train All 10 Models
```bash
python train_all_10_models.py --data data/historical_games.csv --output models/
```

This will create:
```
models/
├── spread_ensemble.pkl
├── total_ensemble.pkl
├── moneyline_ensemble.pkl
├── first_half_spread.pkl
├── home_team_total.pkl
├── away_team_total.pkl
├── xgboost_ensemble.pkl          # If XGBoost installed
├── neural_net_deep.pkl
├── stacking_meta.pkl
└── training_metrics.json
```

**Training Time:**
- Models 1-3: ~30 seconds each
- Models 4-6: ~30 seconds each
- Model 7 (XGBoost): ~2 minutes
- Model 8 (Neural Net): ~5 minutes (uses early stopping)
- Model 9 (Stacking): ~10 seconds

**Total: ~10-15 minutes** for all models on typical dataset (5000+ games)

---

## Usage

### Using the Enhanced AI Council

```python
from enhanced_ai_council import EnhancedAICouncil
from betting_types import GameData

# Initialize council (loads all 10 models)
council = EnhancedAICouncil()

# Prepare game data
game_data: GameData = {
    "game_id": "BUF_MIA_20251109",
    "home_team": "Miami Dolphins",
    "away_team": "Buffalo Bills",
    "spread": -3.5,
    "total": 47.5,
    "home_ml_odds": -175,
    "away_ml_odds": 145,

    # Model predictions (from your predictor)
    "spread_model_home_pct": 0.58,
    "total_model_over_pct": 0.52,
    "home_advantage_pct": 0.62,
    "first_half_spread_home_pct": 0.55,
    "home_team_total_over_pct": 0.60,
    "away_team_total_over_pct": 0.48,
    "xgboost_model_pct": 0.61,
    "neural_net_model_pct": 0.57,
    "stacking_model_pct": 0.64,

    # Situational context
    "kickoff_window": "SNF",  # Triggers primetime boost
    "division": "AFC East",   # Triggers divisional boost
    "weather_tag": "clear",
}

# Get prediction from all 10 models
prediction = council.make_unified_prediction(game_data)

# View results
print(f"Overall Confidence: {prediction.confidence:.1%}")
print(f"Edge Signals: {prediction.edge_signals}")
print(f"Primary Play: {prediction.recommendation['primary_play']}")
print(f"Bet Size: {prediction.recommendation['size']}")

# Enhanced predictions
if prediction.first_half_spread_prediction:
    print(f"1H Pick: {prediction.first_half_spread_prediction.pick}")
    print(f"1H Confidence: {prediction.first_half_spread_prediction.confidence:.1%}")

# Ensemble metadata
print(f"\nModel Breakdown:")
for model, prob in prediction.ensemble_metadata['individual_probabilities'].items():
    print(f"  {model}: {prob:.1%}")
```

### Example Output

```
Overall Confidence: 78.5%
Edge Signals: ['ALGO_CONSENSUS_STRONG', 'PRIMETIME_BOOST', 'DIVISIONAL_BOOST', 'STRONG_MODEL_AGREEMENT']
Primary Play: Miami Dolphins -3.5
Bet Size: large
Risk Level: low

1H Pick: home
1H Confidence: 72.0%

Secondary Plays:
  - 1H MIA -2.0 (First half model 72% confident)
  - MIA Over 24.5 (Team total model 76% confident)

Model Breakdown:
  spread_ensemble: 58.0%
  total_ensemble: 52.0%
  moneyline_ensemble: 62.0%
  first_half_spread: 55.0%
  home_team_total: 60.0%
  away_team_total: 48.0%
  xgboost: 61.0%
  neural_net: 57.0%
  stacking_meta: 64.0%

Situational Adjustments: ['PRIMETIME_BOOST', 'DIVISIONAL_BOOST']
```

---

## AWS Lambda Deployment

The Enhanced AI Council works on Lambda (replace your existing handler):

```python
# lambda_function.py
from enhanced_ai_council import handler

# That's it! The handler function is already defined
```

Package dependencies:
```bash
cd football_betting_system
pip install -r requirements.txt -t package/
cp *.py package/
cp -r models/ package/
cd package && zip -r ../lambda_function.zip .
```

Upload to Lambda and test with:
```json
{
  "game_data": {
    "game_id": "test",
    "home_team": "Team A",
    "away_team": "Team B",
    "spread": -3.5,
    "total": 47.5,
    "spread_model_home_pct": 0.58,
    "total_model_over_pct": 0.52,
    "home_advantage_pct": 0.62,
    "first_half_spread_home_pct": 0.55,
    "home_team_total_over_pct": 0.60,
    "away_team_total_over_pct": 0.48,
    "xgboost_model_pct": 0.61,
    "neural_net_model_pct": 0.57,
    "stacking_model_pct": 0.64
  }
}
```

---

## Model Tuning

### Adjust Model Weights
Edit `enhanced_ai_council.py`:

```python
self.model_weights = {
    'spread_ensemble': 1.0,
    'total_ensemble': 1.0,
    'moneyline_ensemble': 1.0,
    'first_half_spread': 0.8,     # Lower = less influence
    'home_team_total': 0.7,
    'away_team_total': 0.7,
    'xgboost_ensemble': 1.2,
    'neural_net_deep': 0.9,
    'stacking_meta': 1.5,          # Higher = more influence
}
```

### Retrain Individual Models
```bash
# Just retrain one model
python train_all_10_models.py --data data/new_games.csv --output models/
```

Models are trained independently, so you can retrain selectively.

---

## Performance Tracking

Check `models/training_metrics.json` after training:

```json
{
  "spread_ensemble": {
    "accuracy": 0.553,
    "log_loss": 0.682,
    "auc_roc": 0.591
  },
  "xgboost_ensemble": {
    "accuracy": 0.567,
    "log_loss": 0.671,
    "auc_roc": 0.608
  },
  "stacking_meta": {
    "accuracy": 0.574,
    "log_loss": 0.665,
    "auc_roc": 0.619
  }
}
```

**What's Good Performance?**
- Accuracy > 52.4% = Profitable (betting against -110 lines)
- AUC-ROC > 0.55 = Beating random chance
- Log Loss < 0.69 = Well-calibrated probabilities

---

## Architecture Diagram

```
                    ┌─────────────────────────────┐
                    │   Enhanced AI Council       │
                    │   (Orchestrator)            │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
        ┌─────▼─────┐      ┌─────▼─────┐      ┌─────▼─────┐
        │  Models   │      │  Models   │      │  Models   │
        │   1-3     │      │   4-6     │      │   7-10    │
        │  Base     │      │ Enhanced  │      │ Advanced  │
        │ Ensembles │      │  Targets  │      │ Variants  │
        └───────────┘      └───────────┘      └───────────┘
              │                   │                   │
     ┌────────┼────────┐  ┌──────┼──────┐    ┌───────┼────────┐
     ▼        ▼        ▼  ▼      ▼      ▼    ▼       ▼        ▼
  Spread   Total    ML  1H   Home   Away   XGB    NN    Stacking
                           Team   Team
                           Total  Total

                           ALL FEED INTO ▼

                    ┌─────────────────────────────┐
                    │   Weighted Ensemble         │
                    │   + Situational Specialist  │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  Unified Prediction         │
                    │  - Confidence (40-100%)     │
                    │  - Edge Signals             │
                    │  - Recommendations          │
                    │  - Risk Level               │
                    └─────────────────────────────┘
```

---

## FAQ

**Q: Do I need all 10 models?**
A: No! The system gracefully handles missing models. Models 1-3 are core. Models 4-10 enhance edge detection but aren't required.

**Q: Can I add more models?**
A: Yes! Just follow the pattern in `enhanced_model_architectures.py`. Add to the registry and update council weights.

**Q: Why does the stacking model get 1.5x weight?**
A: It learns from all other models, so it should be trusted more. You can adjust this in `enhanced_ai_council.py`.

**Q: How often should I retrain?**
A: Retrain weekly during the season as new games finish. More data = better predictions.

**Q: What if XGBoost isn't available?**
A: The system works without it. Model 7 is skipped if XGBoost isn't installed.

---

## Files Reference

| File | Purpose |
|------|---------|
| `enhanced_model_architectures.py` | Defines Models 4-10 |
| `simple_ensemble.py` | Defines Models 1-3 |
| `enhanced_ai_council.py` | Orchestrates all 10 models |
| `train_all_10_models.py` | Training script |
| `betting_types.py` | Data structures (updated with new prediction types) |
| `unified_betting_intelligence.py` | Base AI Council (Models 1-3 only) |

---

## Next Steps

1. ✅ Train all 10 models
2. ✅ Test Enhanced AI Council locally
3. ✅ Deploy to AWS Lambda
4. ✅ Backtest on historical data
5. ✅ Track performance and tune weights
6. ✅ Start crushing the sportsbooks! 🎯

---

**Built with 🔥 by the 10-Model Super Intelligence Team**
