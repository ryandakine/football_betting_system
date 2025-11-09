# Pickle Module Path Fix

## Problem

Models were being trained in scripts where `SimpleEnsemble` was defined in `__main__`, causing pickle to save them with module path `__main__.SimpleEnsemble`. When loading these models in other scripts, pickle couldn't find the class because it was looking in `__main__` of a different script.

**Error:**
```
AttributeError: Can't get attribute 'SimpleEnsemble' on <module '__main__'>
```

## Solution

1. **Created `simple_ensemble.py` module** - Defines `SimpleEnsemble` in a proper module (not `__main__`)

2. **Created `retrain_simple_models.py`** - Training script that:
   - Imports `SimpleEnsemble` from `simple_ensemble` module
   - Trains models correctly so pickle saves with proper module path: `simple_ensemble.SimpleEnsemble`
   - Models can now be loaded from any script

3. **Updated `simple_model_predictor.py`** - Added:
   - Import of `SimpleEnsemble` from `simple_ensemble` module
   - Backward compatibility shim for legacy models pickled with `__main__.SimpleEnsemble`

## How to Fix Your Models

### Option 1: Retrain (Recommended)

```bash
python3 retrain_simple_models.py
```

This will:
- Load your historical game data
- Train new models with correct module paths
- Save to `models/spread_ensemble.pkl`, `models/total_ensemble.pkl`, `models/moneyline_ensemble.pkl`

### Option 2: Keep Using Old Models (Temporary)

The updated `simple_model_predictor.py` includes a compatibility shim that allows loading old models pickled with `__main__.SimpleEnsemble`. This works but is not ideal long-term.

## Verifying the Fix

After retraining, verify models have correct module path:

```python
import pickle

with open('models/spread_ensemble.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Module: {model.__class__.__module__}")  # Should be: simple_ensemble
print(f"Class: {model.__class__.__name__}")     # Should be: SimpleEnsemble
```

## Why This Matters

Proper module paths ensure:
- ✅ Models can be loaded from any script
- ✅ Models can be deployed to Lambda/production
- ✅ Models work in backtesting scripts
- ✅ No `AttributeError` when unpickling
- ✅ Cleaner, more maintainable code

## Files Changed

- ✅ `simple_ensemble.py` - New module defining SimpleEnsemble
- ✅ `retrain_simple_models.py` - Training script with correct imports
- ✅ `simple_model_predictor.py` - Updated with import and compatibility shim

## Technical Details

### What Pickle Stores

When you pickle an object, Python stores:
- The module path where the class is defined
- The class name
- The object's state

Example:
```python
# If defined in a script run as __main__:
__main__.SimpleEnsemble  # ❌ Only works in that exact script

# If imported from a module:
simple_ensemble.SimpleEnsemble  # ✅ Works everywhere
```

### The Shim

For backward compatibility, `simple_model_predictor.py` temporarily adds `SimpleEnsemble` to `__main__` during unpickling:

```python
sys.modules['__main__'].SimpleEnsemble = SimpleEnsemble
# Load model
del sys.modules['__main__'].SimpleEnsemble
```

This allows old models to load, but new models should be trained properly.
