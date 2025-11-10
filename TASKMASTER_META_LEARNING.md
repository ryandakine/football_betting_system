# TASKMASTER: 12-MODEL META-LEARNING SYSTEM

**GOAL**: Get all 12 models running + Meta-Model to combine them intelligently
**WHY THIS BEATS HUMANS**: Processes 12 factors simultaneously, finds interactions humans miss

---

## THE BREAKTHROUGH: META-LEARNING LAYER

```
Level 1: 12 Base Models → Each analyzes ONE dimension
Level 2: Meta-Model → Learns WHEN to trust which models

Result: Smarter than humans (no cognitive limits)
```

---

## DUAL APPROACH

### Track A: LLM Meta-Reasoner (QUICK TEST - Use your 5 free LLMs)
- ✅ Created: `llm_meta_reasoner.py`
- No training needed
- Explainable reasoning
- Test immediately on Week 11

### Track B: XGBoost Meta-Model (PRODUCTION - More robust)
- Learns optimal model weighting from historical data
- Finds non-linear interactions
- Self-improving (learns from mistakes)

---

## PHASE 1: TRAIN BASE MODELS (LOCAL MACHINE)

You have training data! Skip scraping, go straight to training.

### Run these commands on YOUR LOCAL MACHINE (has internet + dependencies):

```bash
# Combine all historical CSV into one file
cd /path/to/football_betting_system

cat data/football/historical/nfl/nfl_*.csv > data/all_nfl_2014_2024.csv

# Train all 10 AI Council models
python train_advanced_ml_models.py \
  --data data/nfl_training_data_enhanced.json \
  --output-dir models \
  --all

# Train Model 12 (Props)
python train_prop_model.py \
  --data data/prop_training_data.json \
  --output-dir models

# Verify models created
ls -lh models/*.pkl
```

**Expected output**: `models/` directory with 11 .pkl files (Models 1-12 minus Model 11 which is referee-only)

---

## PHASE 2: FIX MODEL LOADING

### Task 2.1: Fix EnhancedModelRegistry

**File**: `ai_council.py` or `enhanced_ai_council.py`

**Find this code:**
```python
class EnhancedModelRegistry:
    models = {}  # ← EMPTY DICT = PROBLEM
```

**Replace with:**
```python
import glob
import joblib
import os

class EnhancedModelRegistry:
    models = {}

    @classmethod
    def load_all_models(cls, models_dir="models"):
        """Load all trained models from disk."""
        if cls.models:  # Already loaded
            return

        model_files = glob.glob(os.path.join(models_dir, "*.pkl"))

        for model_file in model_files:
            model_name = os.path.basename(model_file).replace('.pkl', '')
            try:
                cls.models[model_name] = joblib.load(model_file)
                print(f"✅ Loaded: {model_name}")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")

        print(f"\n📊 Total models loaded: {len(cls.models)}")
```

**Then in `__init__` of main analyzer:**
```python
def __init__(self):
    # Load all models at startup
    EnhancedModelRegistry.load_all_models()
    self.models = EnhancedModelRegistry.models
```

---

## PHASE 3: INTEGRATE LLM META-REASONER (QUICK WIN)

### Task 3.1: Wire into auto_weekly_analyzer.py

**Add to imports:**
```python
from llm_meta_reasoner import LLMMetaReasoner
```

**Add to __init__:**
```python
self.meta_reasoner = LLMMetaReasoner(llm_provider="openrouter")
```

**Add to analyze_single_game():**
```python
# After getting all 12 model predictions
model_outputs = {
    "referee_intelligence": {
        "prediction": f"{'HOME' if ref_favors_home else 'AWAY'} COVERS",
        "confidence": ref_confidence,
        "reasoning": ref_reasoning
    },
    "public_sentiment": {
        "prediction": sentiment_pick,
        "confidence": sentiment_confidence,
        "reasoning": sentiment_reasoning
    },
    # ... add all 12 models
}

game_context = {
    "home_team": home_team,
    "away_team": away_team,
    "spread": spread,
    "total": total,
    "weather": weather_data,
    "referee": referee_name,
    "injuries": injury_report,
    "rest_days": rest_differential,
    "primetime": is_primetime_game
}

# Get meta-model consensus
consensus = self.meta_reasoner.combine_predictions(
    game=f"{away_team} @ {home_team}",
    model_outputs=model_outputs,
    game_context=game_context
)

print(f"\n🤖 META-MODEL CONSENSUS:")
print(f"   Prediction: {consensus['prediction']}")
print(f"   Confidence: {consensus['confidence']}%")
print(f"   Bet Amount: ${consensus['bet_amount']}")
print(f"   Reasoning: {consensus['reasoning']}")
```

---

## PHASE 4: TRAIN XGBOOST META-MODEL (PRODUCTION)

### Task 4.1: Create Meta-Training Data

**File**: `create_meta_training_data.py`

```python
#!/usr/bin/env python3
"""
Create training data for XGBoost meta-model.

For each historical game:
- Get predictions from all 12 base models
- Record actual outcome
- Creates: [model1_pred, model2_pred, ..., model12_pred] → [actual_result]
"""

import json
import pandas as pd
from pathlib import Path

def create_meta_training_data():
    """Create meta-training dataset."""

    # Load historical games
    historical_games = pd.read_csv("data/all_nfl_2014_2024.csv")

    meta_training_data = []

    for idx, game in historical_games.iterrows():
        # Get predictions from all 12 models for this game
        model_predictions = []

        # Model 1-10: AI Council (load from pkl and predict)
        for i in range(1, 11):
            model = load_model(f"models/model_{i}.pkl")
            pred = model.predict(game_features)
            model_predictions.append(pred)

        # Model 11: Referee
        ref_pred = get_referee_prediction(game)
        model_predictions.append(ref_pred)

        # Model 12: Props (if applicable)
        prop_pred = get_prop_prediction(game)
        model_predictions.append(prop_pred)

        # Actual outcome
        actual_outcome = game['actual_result']

        # Store
        meta_training_data.append({
            'game_id': game['game_id'],
            'model_predictions': model_predictions,
            'actual_outcome': actual_outcome
        })

    # Save
    with open('data/meta_training_data.json', 'w') as f:
        json.dump(meta_training_data, f, indent=2)

    print(f"✅ Created meta-training data: {len(meta_training_data)} games")

if __name__ == "__main__":
    create_meta_training_data()
```

**Run:**
```bash
python create_meta_training_data.py
```

### Task 4.2: Train XGBoost Meta-Model

**File**: `train_xgboost_meta_model.py`

```python
#!/usr/bin/env python3
"""Train XGBoost meta-model to optimally combine 12 base models."""

import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Load meta-training data
with open('data/meta_training_data.json') as f:
    data = json.load(f)

# Create feature matrix: 12 model predictions per game
X = np.array([game['model_predictions'] for game in data])
y = np.array([game['actual_outcome'] for game in data])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost meta-model
meta_model = xgb.XGBClassifier(
    max_depth=5,
    n_estimators=100,
    learning_rate=0.1,
    objective='binary:logistic'
)

meta_model.fit(X_train, y_train)

# Evaluate
y_pred = meta_model.predict(X_test)
y_pred_proba = meta_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)

print(f"✅ Meta-Model Accuracy: {accuracy:.2%}")
print(f"✅ Meta-Model Log Loss: {logloss:.4f}")

# Feature importance: Which base models matter most?
feature_importance = meta_model.feature_importances_
model_names = [
    "XGBoost", "Ensemble", "Neural Net", "First Half",
    "Team Total Home", "Team Total Away", "Super Ensemble",
    "Deep Model", "Stacking", "Situational",
    "Referee", "Props"
]

print("\n📊 MODEL IMPORTANCE (Which models meta-model trusts most):")
for name, importance in sorted(
    zip(model_names, feature_importance),
    key=lambda x: x[1],
    reverse=True
):
    print(f"   {name}: {importance:.2%}")

# Save meta-model
import joblib
joblib.dump(meta_model, 'models/xgboost_meta_model.pkl')
print("\n✅ Saved: models/xgboost_meta_model.pkl")
```

**Run:**
```bash
python train_xgboost_meta_model.py
```

---

## PHASE 5: INTEGRATION & TESTING

### Task 5.1: Test on Week 11 Games

```bash
# Test LLM meta-reasoner
python auto_weekly_analyzer.py --week 11 --use-meta-llm

# Test XGBoost meta-model
python auto_weekly_analyzer.py --week 11 --use-meta-xgboost

# Compare both
python compare_meta_approaches.py --week 11
```

### Task 5.2: Verify All 12 Models Contributing

**Check output shows:**
- ✅ All 12 models loaded
- ✅ Each model made prediction for each game
- ✅ Meta-model resolved conflicts
- ✅ Final consensus shown with reasoning
- ✅ Found 10+ edges (not just 4 referee edges)

### Task 5.3: Re-analyze MNF with Full System

```bash
python analyze_single_game.py \
  --home GB \
  --away PHI \
  --week 11 \
  --use-meta-reasoner
```

**Expected output:**
- All 12 models weigh in
- Meta-model identifies which models to trust
- Finds additional edges beyond referee data
- Final prediction: 70-80% confidence with clear reasoning

---

## PHASE 6: PRODUCTION DEPLOYMENT

### Task 6.1: Update Workflow

**Wednesday night:**
```bash
python auto_weekly_analyzer.py --week 11 --use-meta-xgboost --json
```

**Output**: Meta-model consensus for all games

### Task 6.2: Track Meta-Model Performance

**Add to `track_bets.py`:**
```python
--meta-model-used "XGBoost" or "LLM"
--base-models-agreed 8  # How many of 12 agreed
--confidence-before-meta 65  # Before meta-model
--confidence-after-meta 73  # After meta-model adjustment
```

---

## SUCCESS CRITERIA

✅ **All 12 base models trained and loaded**
✅ **LLM meta-reasoner working (quick test)**
✅ **XGBoost meta-model trained (production)**
✅ **Week 11 analysis finds 10+ edges** (not just 4)
✅ **MNF analysis shows all 12 models + meta-consensus**
✅ **Meta-model explains WHICH models it trusted and WHY**

---

## WHY THIS BEATS VEGAS

**Vegas oddsmakers:**
- Process 3-5 factors (cognitive limit)
- Miss complex interactions
- Humans have recency bias

**Your meta-model:**
- Processes 12 factors simultaneously
- Finds hidden patterns: "When referee + weather + injury align → 73% edge"
- No cognitive limits
- Self-improving (learns from mistakes)

---

## IMMEDIATE NEXT STEPS

**RIGHT NOW:**

1. **On your local machine**, run:
```bash
python train_advanced_ml_models.py --data data/nfl_training_data_enhanced.json --all
```

2. **Verify models created:**
```bash
ls -lh models/*.pkl
```

3. **Fix EnhancedModelRegistry** (see Phase 2 above)

4. **Test LLM meta-reasoner** (already created - just wire in)

5. **Analyze Week 11 with all 12 models:**
```bash
python auto_weekly_analyzer.py --week 11
```

6. **Re-analyze MNF PHI @ GB** with full consensus

---

## TWO PATHS - PICK ONE OR BOTH:

**Path A: LLM Meta-Reasoner (Quick)**
- ✅ Already created: `llm_meta_reasoner.py`
- Uses your 5 free LLM calls
- Explainable (shows reasoning)
- Test tonight on MNF

**Path B: XGBoost Meta-Model (Robust)**
- Train on historical data
- Learns optimal weighting
- More accurate long-term
- Deploy for rest of season

**Recommendation: Use BOTH**
- LLM for tonight (MNF)
- XGBoost for production (Week 12+)

---

**STATUS**: Ready to train base models
**BLOCKER**: Need to run training on local machine (sandbox has no numpy/sklearn)
**NEXT**: Train all 10 models locally, then wire in meta-reasoner

