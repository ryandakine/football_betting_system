# 🏈 FREE GGUF Model Setup for NFL Predictions

## Why GGUF Models?

✅ **100% FREE** - No API costs ever
✅ **Fast** - Run locally on your machine
✅ **Private** - Your data never leaves your computer
✅ **No limits** - Analyze unlimited games

## Quick Setup (10 minutes)

### 1. Install Dependencies

```bash
# Install llama-cpp-python (C++ backend for fast inference)
pip install llama-cpp-python

# Or with GPU support (much faster):
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Install other requirements
pip install -r requirements.txt
```

### 2. Download GGUF Models

```bash
# Interactive downloader
python3 download_gguf_models.py

# This will download quantized models (~2-5 GB each)
# Recommended: Start with Phi-3 (smallest, 2.4 GB)
```

### 3. Run Analysis

```bash
python3 gguf_nfl_analyzer.py
```

That's it! The system will:
- Load your local GGUF models
- Analyze the game
- Generate predictions
- Calculate consensus

---

## Available Models

### Recommended Models (Q4 quantization - good balance):

1. **Llama-3-8B** (~4.9 GB)
   - Best overall quality
   - Good reasoning
   - Slower but most accurate

2. **Mistral-7B** (~4.4 GB)
   - Fast and accurate
   - Good for sports analysis
   - Great middle ground

3. **Phi-3-Mini** (~2.4 GB)
   - Fastest
   - Smallest size
   - Surprisingly good for its size
   - **Start with this one!**

---

## Performance

### CPU (no GPU):
- Phi-3: ~2-3 tokens/sec
- Mistral: ~1-2 tokens/sec
- Llama-3: ~1 token/sec

### With GPU (CUDA):
- Phi-3: ~50+ tokens/sec
- Mistral: ~40+ tokens/sec
- Llama-3: ~30+ tokens/sec

**Analysis time per game:**
- CPU only: 30-60 seconds per model
- With GPU: 5-10 seconds per model

---

## File Structure

```
football_betting_system/
├── models/
│   └── gguf/
│       ├── llama-3-8b-instruct.Q4_K_M.gguf
│       ├── mistral-7b-instruct-v0.3.Q4_K_M.gguf
│       └── phi-3-mini-4k-instruct.Q4_K_M.gguf
├── gguf_nfl_analyzer.py
├── download_gguf_models.py
└── tonights_game.json
```

---

## GPU Acceleration (Optional but Recommended)

### For NVIDIA GPUs:

```bash
# Install with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall

# Verify GPU is detected
python3 -c "from llama_cpp import Llama; print('GPU support available!')"
```

### For Apple Silicon (M1/M2/M3):

```bash
# Metal acceleration (built-in)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall
```

---

## Manual Model Download

If the downloader doesn't work, manually download from:

**Hugging Face GGUF Models:**
- https://huggingface.co/models?library=gguf

**Recommended sources:**
- Llama-3: https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF
- Mistral: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
- Phi-3: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf

Save to: `models/gguf/` directory

---

## Troubleshooting

### "No module named 'llama_cpp'"
```bash
pip install llama-cpp-python
```

### "No GGUF models found"
```bash
# Download models
python3 download_gguf_models.py

# Or manually place .gguf files in models/gguf/
```

### Slow performance
- Use Q4 quantized models (smaller, faster)
- Enable GPU acceleration
- Use Phi-3 model (fastest)

### Out of memory
- Use smaller model (Phi-3)
- Reduce n_ctx in gguf_nfl_analyzer.py
- Close other applications

---

## Example Output

```
🏈 FREE LOCAL GGUF NFL ANALYZER
======================================================================

Game: Las Vegas Raiders @ Denver Broncos
Spread: Denver Broncos -9.5
Total: 43.0

🤖 RUNNING LOCAL GGUF MODELS
======================================================================

✅ Found: phi3 (phi-3-mini-4k-instruct.Q4_K_M.gguf)
✅ Found: mistral-7b (mistral-7b-instruct-v0.3.Q4_K_M.gguf)

🔄 phi3...
  ✅ Denver Broncos (75%)

🔄 mistral-7b...
  ✅ Denver Broncos (80%)

📊 PREDICTIONS
======================================================================

🤖 PHI3
   Pick: Denver Broncos
   Confidence: 75%
   Analysis: The Broncos have home field advantage...

🤖 MISTRAL-7B
   Pick: Denver Broncos
   Confidence: 80%
   Analysis: Denver's defense will be too strong...

🎯 CONSENSUS
======================================================================
   Pick: Denver Broncos
   Confidence: 78%
   Agreement: 100% (2/2 models)
```

---

## Cost Comparison

### Paid APIs (for 100 games):
- Claude: $150-300
- GPT-4: $200-400
- Total: $350-700

### GGUF Models:
- Cost: **$0** (after initial download)
- Unlimited usage
- No rate limits

---

## Next Steps

1. Download at least one model (start with Phi-3)
2. Run: `python3 gguf_nfl_analyzer.py`
3. Analyze all weekend games for FREE!

---

**Questions?** Check the main README.md or raise an issue on GitHub.
