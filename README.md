# 🏈 Advanced Football Betting System

[![CI/CD](https://github.com/username/football-betting-system/workflows/Football%20Betting%20System%20CI/CD/badge.svg)](https://github.com/username/football-betting-system/actions)
[![Coverage](https://codecov.io/gh/username/football-betting-system/branch/main/graph/badge.svg)](https://codecov.io/gh/username/football-betting-system)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Advanced NFL/NCAAF betting system with AI analysis, cloud GPU integration, and automated portfolio optimization.

## 🚀 Features

### AI-Powered Analysis
- **Local Inference**: 7B Hugging Face models for cost-effective analysis
- **Cloud GPU**: 30B-70B models on RunPod, Vast.ai, AWS/GCP
- **Ensemble Predictions**: Multi-model consensus with dynamic weighting
- **Uncensored Models**: Honest, unfiltered betting insights

### Advanced Risk Management
- **Portfolio Correlation Analysis**: Identify and limit correlated positions
- **Dynamic Position Sizing**: Kelly Criterion with 25% conservative fraction
- **Real-time Risk Monitoring**: Live drawdown and exposure tracking
- **Drawdown Protection**: Automated position reduction at 5% drawdown

### Real-time Intelligence
- **Line Movement Tracking**: Monitor odds changes across multiple sportsbooks
- **Weather & Injury Integration**: Factor environmental and player impacts
- **Market Efficiency Analysis**: Detect sharp money and arbitrage opportunities
- **Behavioral Intelligence**: Contrarian opportunity identification

### Professional Dashboard
- **Live Recommendations**: Real-time betting opportunities
- **Performance Metrics**: Win rate, ROI, Sharpe ratio tracking
- **Risk Dashboard**: Portfolio correlation and exposure visualization
- **Model Performance**: AI model accuracy and latency monitoring

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- Git
- Docker (optional)

### Quick Setup
```bash
# Clone repository
git clone https://github.com/username/football-betting-system.git
cd football-betting-system

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Initialize database
python setup_database.py

# Run system
python unified_football_production_main.py
```

### Docker Setup
```bash
# Build and run with Docker
docker-compose up --build
```

## 📊 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Win Rate | >55% | TBD |
| ROI | >10% | TBD |
| Max Drawdown | <5% | TBD |
| Response Time | <1s | TBD |
| System Uptime | >99.9% | TBD |

## 🔧 Configuration

### API Keys Required
```bash
# Sports Data
ESPN_API_KEY=your_espn_key
ODDS_API_KEY=your_odds_api_key

# AI Models
ANTHROPIC_API_KEY=your_claude_key
OPENAI_API_KEY=your_openai_key
PERPLEXITY_API_KEY=your_perplexity_key

# Cloud GPU (optional)
RUNPOD_API_KEY=your_runpod_key
VAST_API_KEY=your_vast_key
```

### Risk Management Settings
```python
# Maximum portfolio risk per bet
MAX_PORTFOLIO_RISK = 0.02  # 2%

# Kelly Criterion fraction
KELLY_FRACTION = 0.25  # Conservative 25%

# Correlation threshold
MAX_CORRELATION = 0.7  # 70%

# Drawdown protection trigger
MAX_DRAWDOWN = 0.05  # 5%
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test suite
pytest tests/test_correlation_analysis.py -v

# Run performance tests
pytest tests/performance/ -v
```

## 📈 Usage Examples

### Basic Analysis
```python
from unified_football_production_main import UnifiedFootballProductionSystem

# Initialize system
system = UnifiedFootballProductionSystem()

# Run complete analysis pipeline
results = await system.run_complete_pipeline()

# Get recommendations
recommendations = results.get_recommendations()
for rec in recommendations:
    print(f"{rec.game}: {rec.bet_type} @ {rec.odds} (EV: {rec.expected_value:.1%})")
```

### Cloud GPU Analysis
```python
from huggingface_cloud_gpu import CloudGPUAIEnsemble, CloudGPUConfig

# Configure cloud GPU
config = CloudGPUConfig(
    service="runpod",
    gpu_type="RTX4090",
    vram_gb=24
)

# Initialize ensemble
ensemble = CloudGPUAIEnsemble(cloud_config=config)
await ensemble.initialize_models()

# Analyze game
analysis = await ensemble.analyze_football_game(game_data, 'h2h')
```

### Portfolio Analysis
```python
from portfolio_correlation_analysis import PortfolioCorrelationAnalyzer

# Initialize analyzer
analyzer = PortfolioCorrelationAnalyzer(
    max_correlation_threshold=0.7,
    max_portfolio_risk=0.02
)

# Analyze correlations
result = await analyzer.analyze_portfolio_correlations(opportunities)

# Visualize correlations
analyzer.visualize_correlations(result)
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   AI Analysis   │    │ Risk Management │
│                 │    │                 │    │                 │
│ • ESPN API      │───▶│ • Local 7B LLMs │───▶│ • Correlation   │
│ • Odds APIs     │    │ • Cloud 30B-70B │    │ • Position Size │
│ • Weather APIs  │    │ • Ensemble      │    │ • Drawdown Prot │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Database      │    │  Recommendations│    │   Dashboard     │
│                 │    │                 │    │                 │
│ • Game Data     │    │ • Kelly Sizing  │    │ • Live Updates  │
│ • Odds History  │    │ • Risk Adjusted │    │ • Performance   │
│ • Performance   │    │ • EV Optimized  │    │ • Risk Metrics  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📚 Documentation

- [Setup Guide](docs/SETUP.md)
- [API Reference](docs/API.md)
- [Risk Management](docs/RISK_MANAGEMENT.md)
- [Cloud GPU Guide](docs/CLOUD_GPU.md)
- [Contributing](CONTRIBUTING.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Sports betting involves risk and may not be legal in all jurisdictions. Users are responsible for complying with local laws and regulations. Past performance does not guarantee future results.

## 🆘 Support

- 📧 Email: support@football-betting-system.com
- 💬 Discord: [Join our community](https://discord.gg/football-betting)
- 📖 Wiki: [Documentation](https://github.com/username/football-betting-system/wiki)
- 🐛 Issues: [Report bugs](https://github.com/username/football-betting-system/issues)

---

**Built with ❤️ for the sports betting community**