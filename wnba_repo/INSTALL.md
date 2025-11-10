# Installation Guide - WNBA Betting System

## Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/wnba-betting-system.git
cd wnba-betting-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys

# Run the system
python3 run.py
```

## Detailed Installation

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)
- API key from The Odds API (https://the-odds-api.com)

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/wnba-betting-system.git
cd wnba-betting-system
```

### 3. Set Up Virtual Environment

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Optional: Install with AI enhancements
pip install -r requirements.txt -e ".[ai]"

# Optional: Install development tools
pip install -r requirements.txt -e ".[dev]"
```

### 5. Configuration

#### API Keys

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```bash
   ODDS_API_KEY=your_actual_key_here
   ```

3. (Optional) Add AI service keys for enhanced analysis:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_key
   OPENAI_API_KEY=your_openai_key
   ```

#### Bankroll Settings

Edit `.env` to set your bankroll parameters:

```bash
WNBA_BANKROLL=50000.0          # Your total bankroll
WNBA_UNIT_SIZE=500.0           # Base unit size
WNBA_MAX_EXPOSURE=0.08         # Max 8% exposure per game (conservative)
WNBA_MIN_EDGE_THRESHOLD=0.06   # Minimum 6% edge required
```

### 6. Verify Installation

```bash
# Test the system
python3 run.py

# Or run the config test
python3 wnba_config.py
```

## Optional: Development Setup

### Install with Development Tools

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Formatting

```bash
black .
flake8 .
```

## Troubleshooting

### Common Issues

**Issue: "Module not found"**
- Solution: Make sure you're in the virtual environment and all dependencies are installed

**Issue: "API key not found"**
- Solution: Check that your `.env` file exists and contains valid API keys

**Issue: "No games found"**
- Solution: WNBA season runs from May to October. Check if games are currently scheduled.

### Getting Help

- Check the README.md for usage examples
- Review the documentation in each Python module
- Open an issue on GitHub

## WNBA Season Schedule

- **Pre-Season**: April - Early May
- **Regular Season**: Mid-May through Mid-September (40 games per team)
- **Commissioner's Cup**: June
- **All-Star Break**: Mid-July
- **Playoffs**: Late September - Mid-October

**Note**: The system is most effective during the regular season when there are 3-6 games per day.

## Next Steps

After installation:
1. Read QUICKSTART.md for usage examples
2. Review wnba_config.py for customization options
3. Check the README.md for detailed features
4. Run your first analysis with `python3 run.py`

## System Requirements

- **Minimum**: Python 3.8, 4GB RAM, 1GB disk space
- **Recommended**: Python 3.10+, 8GB RAM, 5GB disk space
- **Network**: Stable internet connection for API calls
- **OS**: Linux, macOS, or Windows

## Updating

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Reactivate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
