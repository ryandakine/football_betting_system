# Standalone Repository Extraction Guide

This guide explains how to extract the standalone Women's Basketball betting systems into their own separate git repositories.

## Overview

Two complete standalone systems have been created:

1. **Women's College Basketball** (`womens_college_basketball_repo/`)
2. **WNBA** (`wnba_repo/`)

Each is fully self-contained with all necessary files to operate independently.

---

## Option 1: Extract to New Git Repositories (Recommended)

### Women's College Basketball

```bash
# Navigate to the standalone repo
cd womens_college_basketball_repo

# Initialize new git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit - Women's College Basketball Betting System v1.0.0"

# Create repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/womens-college-basketball-betting.git
git branch -M main
git push -u origin main
```

### WNBA

```bash
# Navigate to the standalone repo
cd wnba_repo

# Initialize new git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit - WNBA Betting System v1.0.0"

# Create repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/wnba-betting-system.git
git branch -M main
git push -u origin main
```

---

## Option 2: Copy to New Locations

If you want to keep them separate locally first:

```bash
# From the football_betting_system directory
cp -r womens_college_basketball_repo ~/Projects/wcbb-betting
cp -r wnba_repo ~/Projects/wnba-betting

# Then follow Option 1 steps from each new location
```

---

## What's Included in Each Repo

### Core System Files
- `main_analyzer.py` - Main intelligence system
- `game_prioritization.py` - Game scoring and prioritization
- `social_weather_analyzer.py` - Social sentiment analysis
- `parlay_optimizer.py` - Parlay construction
- `realtime_monitor.py` - Live tracking
- `*_config.py` - Pydantic configuration

### Setup & Documentation
- `requirements.txt` - Python dependencies
- `setup.py` - Package installation
- `.gitignore` - Git exclusions
- `LICENSE` - MIT License
- `.env.example` - Configuration template
- `README.md` - Comprehensive documentation
- `INSTALL.md` - Installation guide
- `run.py` - Quick start script
- `init.sh` - Automated setup

---

## Quick Start After Extraction

For either system:

```bash
# 1. Navigate to the repo
cd <repo-directory>

# 2. Run the initialization script
chmod +x init.sh
./init.sh

# 3. Add your API keys
cp .env.example .env
nano .env  # or vim/code/etc

# 4. Run the system
python3 run.py
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add API keys

# Run
python3 run.py [bankroll]
```

---

## Publishing to PyPI (Optional)

To publish as a pip-installable package:

```bash
# Install build tools
pip install build twine

# Build the package
python3 -m build

# Upload to PyPI
python3 -m twine upload dist/*

# Then others can install with:
pip install womens-college-basketball-betting
# or
pip install wnba-betting-system
```

---

## Directory Structure of Each Repo

```
repo/
├── .env.example              # Configuration template
├── .gitignore               # Git exclusions
├── INSTALL.md               # Installation guide
├── LICENSE                  # MIT License
├── README.md                # Documentation
├── __init__.py              # Package init
├── game_prioritization.py   # Game scoring
├── init.sh                  # Setup script
├── main_analyzer.py         # Main system
├── parlay_optimizer.py      # Parlay optimization
├── realtime_monitor.py      # Live monitoring
├── requirements.txt         # Dependencies
├── run.py                   # Quick start
├── setup.py                 # Package setup
├── social_weather_analyzer.py  # Sentiment analysis
└── *_config.py              # Configuration
```

---

## API Requirements

Both systems require:

**Required:**
- The Odds API key (https://the-odds-api.com)
  - Women's College Basketball: sport key `basketball_ncaaw`
  - WNBA: sport key `basketball_wnba`

**Optional (for enhanced AI):**
- Anthropic API key
- OpenAI API key
- Grok API key

---

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 1GB minimum, 5GB recommended
- **Network**: Stable internet connection

---

## Season Information

### Women's College Basketball
- **Season**: November - March
- **Tournament**: March (March Madness)
- **Games per day**: 50-200 during peak season
- **Best time to use**: January - March

### WNBA
- **Regular Season**: Mid-May through Mid-September
- **Playoffs**: Late September - Mid-October
- **Games per day**: 3-6 during regular season
- **Best time to use**: June - August
- **Special events**: Commissioner's Cup (June), All-Star (July)

---

## Support & Documentation

Each repo includes:
- Comprehensive README with examples
- Detailed INSTALL guide
- Inline code documentation
- Configuration examples
- Quick start scripts

For issues or questions:
1. Check the README.md
2. Review INSTALL.md
3. Check configuration in *_config.py
4. Open an issue on GitHub

---

## Customization

Both systems can be customized via:
1. `.env` file - Runtime configuration
2. `*_config.py` - System parameters
3. Direct code modifications

Common customizations:
- Bankroll settings
- Edge thresholds
- Confidence requirements
- Parlay limits
- Conference/team weights

---

## License

Both systems are released under the MIT License.
See LICENSE file in each repo for details.

---

## Next Steps

1. Extract repos using Option 1 or 2 above
2. Follow the installation guide in each repo
3. Configure API keys
4. Run your first analysis
5. Customize as needed
6. (Optional) Share on GitHub
7. (Optional) Publish to PyPI

---

**Note**: These systems are for educational and research purposes.
Always gamble responsibly and within your means.
