# Remote Work Setup Guide

## Quick Start for Working From Work

### 1. Get Your API Keys Ready

Copy these from your local `.env` file:

**NOTE: API keys are stored locally only - not in the repo.**

When working remotely, you'll need to set these environment variables.
Contact yourself for the actual keys or retrieve from your local machine's `.env` file.

```bash
# Essential keys (minimum required):
THE_ODDS_API_KEY=<your_odds_api_key>
OPENWEATHER_API_KEY=<your_weather_api_key>

# AI keys (for predictions):
CLAUDE_API_KEY=<your_claude_key>
OPENAI_API_KEY=<your_openai_key>
PERPLEXITY_API_KEY=<your_perplexity_key>
GROK_API_KEY=<your_grok_key>
HUGGINGFACE_API_TOKEN=<your_hf_token>
```

### 2. Model Options for Remote Testing

#### Option A: Use Trained Pickle Models (Uploaded to S3)
The system has pre-trained sklearn models already on AWS S3:
- `spread_ensemble.pkl`
- `total_ensemble.pkl` 
- `moneyline_ensemble.pkl`

**Note**: These were trained on backtested features (not raw predictions), so accuracy is currently ~50%. Need GGUF integration to improve.

#### Option B: Use API-Based Models (Recommended for Remote)
Instead of 20GB GGUF models, use AI APIs for predictions:
- **Claude** (Anthropic)
- **GPT-4** (OpenAI)
- **Perplexity**
- **Grok** (xAI)

The `ai_council/` system already supports API-based models.

#### Option C: Use Hugging Face Inference API
Instead of downloading 20GB GGUF models, use Hugging Face's cloud inference:
```python
# In practical_gguf_ensemble.py, replace local loading with:
from huggingface_hub import InferenceClient

client = InferenceClient(token=os.getenv("HUGGINGFACE_API_TOKEN"))
response = client.text_generation(
    "mistralai/Mistral-7B-Instruct-v0.2",
    prompt=game_prompt
)
```

### 3. Testing the System Remotely

#### Run Basic Test:
```bash
# Set environment variables (use your actual keys)
export THE_ODDS_API_KEY="<your_key>"
export OPENWEATHER_API_KEY="<your_key>"

# Test odds collection
python data_collection/odds_api_client.py

# Test AI Council (uses API models)
python ai_council/unified_betting_intelligence.py
```

#### Run Full System Test:
```bash
# Run daily betting workflow (uses S3 models + APIs)
python daily_runner/run_daily_betting.py
```

### 4. What's Available Remotely

✅ **Available:**
- All Python code
- Configuration files
- AI Council system (API-based)
- AWS Lambda deployment (uses S3 models)
- Backtesting framework
- Data collection scripts

❌ **Not Available (Local Only):**
- 20GB GGUF models (use APIs instead)
- Historical backtesting data (can regenerate)
- Referee conspiracy data (can regenerate)

### 5. Recommended Remote Workflow

**For Live Betting Analysis:**
1. Use AWS Lambda (already deployed)
2. Models stored in S3
3. Trigger via API Gateway or scheduled events

**For Development/Testing:**
1. Use AI API-based predictions (Claude, GPT-4)
2. No need for local GGUF models
3. Test with real-time odds data

### 6. To Recreate Full Local Setup Later

When back on your local machine:
```bash
# Pull latest code
cd /home/ryan/code/football_betting_system
git pull

# GGUF models still there (not deleted locally)
ls -lh models/gguf/  # Should show 5 models (20GB)

# Run with local models
python generate_gguf_predictions.py
python unified_end_to_end_backtest_enhanced.py
```

## Key Insight

**You don't need the GGUF models for remote work!** The system has 3 prediction paths:
1. **GGUF models** (local only, 20GB)
2. **Sklearn models** (small, in S3, ~50% accuracy until retrained)
3. **AI APIs** (Claude/GPT-4/Perplexity - best for remote)

Use option #3 (AI APIs) when working remotely.
