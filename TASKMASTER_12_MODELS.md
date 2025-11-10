# TASKMASTER PLAN: GET ALL 12 MODELS RUNNING

**MISSION**: Get all 12 models operational and analyzing games
**STATUS**: Currently only 1.5 models running (Referee + Sentiment)
**GOAL**: All 12 models feeding into intelligent_model_selector.py

---

## CURRENT STATE

✅ **Model 11**: Referee Intelligence (WORKING)
✅ **Model 2**: Public Sentiment Contrarian (WORKING)
❌ **Models 1-10**: AI Council (NOT TRAINED, NOT LOADED)
❌ **Model 12**: Props Intelligence (SKIPPED)

---

## PHASE 1: COLLECT TRAINING DATA

### Task 1.1: Scrape Historical Game Data (2018-2024)
- **Script**: Create `scrape_historical_nfl_data.py`
- **Data needed**: Game results, scores, spreads, totals, team stats
- **Source**: Pro Football Reference or ESPN
- **Output**: `data/historical_games_2018_2024.json`

### Task 1.2: Scrape Historical Props Data
- **Script**: Create `scrape_historical_props.py`
- **Data needed**: Player stats, prop lines, prop results
- **Output**: `data/historical_props_2018_2024.json`

### Task 1.3: Verify Data Quality
- **Check**: Minimum 7 seasons of data (2018-2024)
- **Check**: At least 2,000 games
- **Check**: Complete fields (no missing spreads/totals)

---

## PHASE 2: TRAIN MODELS 1-10 (AI COUNCIL)

### Task 2.1: XGBoost Model (Model 1)
- **File**: `train_advanced_ml_models.py` (already exists)
- **Action**: Run with historical data
- **Command**: `python train_advanced_ml_models.py --model xgboost`
- **Output**: `models/xgboost_model.pkl`

### Task 2.2: Neural Network (Model 3)
- **File**: `train_advanced_ml_models.py`
- **Command**: `python train_advanced_ml_models.py --model neural_net`
- **Output**: `models/neural_net_model.pkl`

### Task 2.3: Ensemble Model (Model 4)
- **File**: `train_ensemble.py` (already exists)
- **Command**: `python train_ensemble.py`
- **Output**: `models/ensemble_model.pkl`

### Task 2.4: Models 5-10 (Additional Models)
- **File**: `train_all_10_models.py` (already exists)
- **Command**: `python train_all_10_models.py --all`
- **Output**: `models/model_{1-10}.pkl`

---

## PHASE 3: TRAIN MODEL 12 (PROPS)

### Task 3.1: Train Prop Model
- **File**: `train_prop_model.py` (already exists)
- **Command**: `python train_prop_model.py`
- **Output**: `models/prop_model.pkl`

---

## PHASE 4: FIX MODEL LOADING

### Task 4.1: Fix EnhancedModelRegistry
- **File**: `ai_council.py` or `enhanced_ai_council.py`
- **Problem**: `EnhancedModelRegistry.models = {}` (empty dict)
- **Fix**: Load all trained .pkl files from `models/` directory
- **Code change**:
```python
class EnhancedModelRegistry:
    models = {}

    @classmethod
    def load_all_models(cls):
        """Load all trained models from disk"""
        model_files = glob.glob('models/*.pkl')
        for model_file in model_files:
            model_name = os.path.basename(model_file).replace('.pkl', '')
            cls.models[model_name] = joblib.load(model_file)
```

### Task 4.2: Wire Models into intelligent_model_selector.py
- **File**: `intelligent_model_selector.py`
- **Add**: All 10 AI Council model types to ModelType enum
- **Add**: Selection logic for each model based on data availability
- **Update**: Priority decision tree

---

## PHASE 5: INTEGRATE INTO WORKFLOW

### Task 5.1: Update auto_weekly_analyzer.py
- **Add**: Load all 12 models on startup
- **Add**: Pass game data to intelligent_model_selector
- **Add**: Get predictions from all applicable models
- **Add**: Display multi-model consensus in report

### Task 5.2: Update analyze_single_game.py
- **Replace**: Hardcoded 50/50 predictions
- **Add**: Real model predictions from AI Council
- **Add**: Model confidence scores

---

## PHASE 6: TEST ALL 12 MODELS

### Task 6.1: Test on Week 11 Games
- **Command**: `python auto_weekly_analyzer.py --week 11`
- **Verify**: All 12 models analyzed
- **Verify**: Multiple edges found (not just referee edges)
- **Check**: Model consensus displayed

### Task 6.2: Verify Model Outputs
- **Check**: Each model returns prediction + confidence
- **Check**: No models showing 0% or 50% (dead models)
- **Check**: intelligent_model_selector picks best model per game

### Task 6.3: Re-analyze PHI @ GB MNF
- **Command**: `python analyze_single_game.py --home GB --away PHI`
- **Expected**: All 12 models weigh in
- **Expected**: Find additional edges beyond just referee data

---

## PHASE 7: PRODUCTION DEPLOYMENT

### Task 7.1: Update Documentation
- **File**: Update `SYSTEM_AUDIT_REPORT.md`
- **Change**: Remove "only 1.5 models" warnings
- **Add**: "All 12 models operational" confirmation

### Task 7.2: Create Model Performance Tracker
- **File**: Create `track_model_performance.py`
- **Track**: Which model recommended each bet
- **Track**: Win rate by model
- **Track**: ROI by model

### Task 7.3: Update autonomous_betting_agent.py
- **Verify**: Calls all 12 models
- **Verify**: Uses intelligent_model_selector
- **Verify**: Logs which model made each pick

---

## EXECUTION ORDER

1. ✅ **PHASE 1**: Collect data (if not already available)
2. ✅ **PHASE 2**: Train Models 1-10
3. ✅ **PHASE 3**: Train Model 12
4. ✅ **PHASE 4**: Fix model loading
5. ✅ **PHASE 5**: Integrate into workflow
6. ✅ **PHASE 6**: Test everything
7. ✅ **PHASE 7**: Production deployment

---

## IMMEDIATE NEXT STEPS

1. **Check if historical data exists**: `ls -la data/`
2. **If data exists**: Go straight to PHASE 2 (train models)
3. **If data missing**: Start PHASE 1 (scrape data)
4. **After models trained**: Fix EnhancedModelRegistry (PHASE 4)
5. **After registry fixed**: Test on Week 11 (PHASE 6)
6. **After testing**: Analyze MNF with all 12 models

---

## SUCCESS CRITERIA

✅ All 12 models loaded in `EnhancedModelRegistry.models`
✅ `auto_weekly_analyzer.py` shows predictions from 10+ models per game
✅ Week 11 analysis finds 10+ edges (not just 4 referee edges)
✅ MNF PHI @ GB shows consensus from all applicable models
✅ `track_bets.py` logs which model recommended each bet

---

## NOTES

- **No time estimates** - work 24/7 until complete
- **Priority**: Get models trained and loaded FIRST
- **Secondary**: Refine integration and testing
- **Focus**: More edges = more money = faster bankroll growth

---

**STATUS**: READY TO START
**FIRST ACTION**: Check for existing historical data
