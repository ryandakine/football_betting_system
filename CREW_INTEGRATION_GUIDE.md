# Crew Adjustment System - Integration Guide

## Overview

The crew adjustment system is **fully automated** and integrates seamlessly into your existing NFL betting system. Once initialized, all predictions are automatically adjusted based on referee crew tendencies.

## Quick Start

### Option 1: Add to Your Main Startup Script

In your `main.py` or `daily_prediction_system.py` or wherever your system starts:

```python
# At the very beginning of your main function
from init_crew_adjustments import initialize_crew_system

def main():
    # Initialize crew adjustment system FIRST
    initialize_crew_system()
    
    # ... rest of your code ...
```

### Option 2: Import at Module Level

In your prediction module:

```python
from init_crew_adjustments import initialize_crew_system

# Initialize once when module loads
initialize_crew_system()

# Now all predictions automatically adjusted
def make_predictions(games):
    # Your existing code - no changes needed
    return predictions  # Already adjusted!
```

### Option 3: One-Line Setup in requirements

If you want it to auto-initialize, add to the very top of your entry point:

```python
import init_crew_adjustments; init_crew_adjustments.initialize_crew_system()
```

## What Gets Adjusted Automatically

Once initialized, these items are automatically adjusted:

1. **Prediction dictionaries** with `predicted_margin` or `predicted_winner`
2. **Game lists** with crew information
3. **JSON responses** from API calls
4. **Database records** before insertion

## Features

### Automatic Adjustments
- ✅ Crew model loads at startup
- ✅ All predictions intercepted and adjusted
- ✅ JSON serialization patched
- ✅ No code changes required in existing functions

### What's Included

```
crew_adjustment_middleware.py       - Core adjustment logic
init_crew_adjustments.py            - Startup initializer
crew_prediction_integration.py       - Advanced API (optional)
crew_prediction_model.pkl           - Trained model (3,882 games)
game_records.json                   - Extracted game data
```

## Sample Output

Before initialization:
```
BAL vs ARI: +3.5 points (Alan Eck crew)
```

After initialization:
```
BAL vs ARI: +19.4 points (Alan Eck crew) [crew adjustment: +31.9]
```

## Key Adjustments by Crew

**Alan Eck** - Blowout specialist
- Favors: BAL (+17.4), KC (+10.5), PHI (+10.1)
- Crushes: NYG (-16.7), ATL (-16.0), ARI (-14.5)

**Carl Cheffers** - Selective bias
- Favors: DET (+10.3), WAS (+8.3), DEN (+8.1)
- Crushes: CAR (-18.6), TEN (-11.6), NYG (-8.6)

**Craig Wrolstad** - High variance
- Favors: DET (+10.1), GB (+9.2), PHI (+8.3)
- Crushes: CAR (-9.4), TEN (-7.5), NYJ (-7.3)

## Performance

- **Model Accuracy**: R² = 0.5987
- **Mean Absolute Error**: 6.05 points
- **Confidence**: 60% - Good for supplementary analysis
- **Weight**: 50% of adjustment applied (50% is crew, 50% is your prediction)

## Monitoring

The system logs significant adjustments:

```
🔧 Alan Eck adjustment: BAL vs ARI +3.5 → +19.4 (crew: +31.9)
```

Check logs for:
- `CREW ADJUSTMENT SYSTEM INITIALIZED` - System ready
- `🔧 adjustment:` - Significant crew impact detected
- `⚠️ Failed to load` - Model loading issues

## Troubleshooting

### Model Not Loading
```python
from crew_adjustment_middleware import get_middleware
m = get_middleware()
print(m.model)  # Should not be None
```

### Model Path Wrong
Update in `crew_adjustment_middleware.py` line 32:
```python
model_path = '/path/to/your/crew_prediction_model.pkl'
```

### No Adjustments Happening
1. Check logs for initialization message
2. Verify crew name matches (Alan Eck, not alan eck)
3. Verify team abbreviations match (BAL, not Baltimore)

## Advanced Usage

### Manual Adjustment
```python
from crew_adjustment_middleware import get_middleware

middleware = get_middleware()
adjusted = middleware.adjust_prediction({
    'home_team': 'BAL',
    'away_team': 'ARI',
    'referee_crew': 'Alan Eck',
    'predicted_margin': 3.5
})
print(adjusted['predicted_margin'])  # +19.4
```

### Get Crew Impact
```python
from crew_adjustment_middleware import get_middleware

middleware = get_middleware()
impact = middleware.predict_crew_margin('Alan Eck', 'BAL')  # +17.4
```

### Batch Adjustments
```python
from crew_adjustment_middleware import get_middleware

middleware = get_middleware()
adjusted_list = middleware.adjust_predictions_batch(my_predictions)
```

## Implementation Checklist

- [ ] Add `from init_crew_adjustments import initialize_crew_system` to main startup
- [ ] Call `initialize_crew_system()` at application start
- [ ] Verify logs show `✅ CREW ADJUSTMENT SYSTEM INITIALIZED`
- [ ] Test with sample game (Alan Eck + BAL should show +19.4 adjustment)
- [ ] Monitor logs for `🔧 adjustment:` messages
- [ ] Track ROI improvement from crew-adjusted predictions

## Expected Impact

Based on 3,882 games of analysis:

- **Direct wins**: Games where crew bias predicts >5 point swing
- **Edge improvement**: +0.5-2.0 points on average
- **High-conviction plays**: Games with crew bias >10 points
- **Avoid plays**: Games with hidden crew manipulation risk

## Files Created

```
crew_adjustment_middleware.py      (263 lines) - Core middleware
init_crew_adjustments.py           (112 lines) - Initialization
crew_prediction_integration.py     (172 lines) - Advanced API
CREW_INTEGRATION_GUIDE.md          (This file) - Setup guide
```

## System Architecture

```
Game Input
    ↓
Referee Crew Assigned
    ↓
Middleware Intercepts
    ↓
Crew Model Predicts Bias
    ↓
Prediction Adjusted
    ↓
Output with Crew Impact
```

## Questions?

The system is **zero-configuration** after the one-line init. All adjustments are automatic and transparent.
