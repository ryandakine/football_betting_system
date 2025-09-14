# MLB Kelly Betting System - Agents & Workflow

## 🎯 System Overview
This document describes the agents (components) and workflow of the MLB Kelly Betting System. Each agent has a specific role in the betting pipeline.

## 🤖 Agent Architecture

### 1. **Data Collection Agent**
- **Location**: `data_collection/`
- **Purpose**: Fetch real-time odds and game data
- **Frequency**: Every 15-30 minutes
- **APIs**:
  - Odds API (The Odds API)
  - Weather API (OpenWeatherMap)
  - MLB Stats API
- **Output**: Raw odds data, weather conditions, team stats

### 2. **Line Discrepancy Agent**
- **Location**: `analysis_engine/line_discrepancies.js`
- **Purpose**: Identify profitable line differences across sportsbooks
- **Triggers**: After each odds update
- **Features**:
  - Bookmaker reliability weighting
  - Steam move detection
  - Multi-market analysis (ML, spreads, totals)
- **Output**: Discrepancy alerts with edge calculations

### 3. **Value Calculation Agent**
- **Location**: `analysis_engine/value_calculator.py`
- **Purpose**: Calculate expected value and Kelly bet sizing
- **Inputs**: Odds data, model predictions, bankroll
- **Features**:
  - Kelly Criterion implementation
  - Risk management rules
  - Portfolio optimization
- **Output**: Bet recommendations with sizing

### 4. **AI Council Agent**
- **Location**: `ai_council/`
- **Purpose**: Multi-model consensus for game analysis
- **Models**:
  - **PrimaryModel**: Main ML predictions
  - **DumbModel**: Simple contrarian logic (placeholder)
  - **WeatherModel**: Weather impact analysis
  - **TrendModel**: Historical pattern recognition
- **Process**: Weighted consensus voting
- **Output**: Confidence scores and reasoning

### 5. **SGP (Same Game Parlay) Agent** *(Planned)*
- **Location**: `analysis_engine/sgp_builder.js`
- **Purpose**: Build profitable same-game parlays
- **Features**:
  - Correlation analysis
  - Multi-leg optimization
  - Risk-adjusted payouts
- **Target**: 2-3 leg parlays with +150 to +400 odds

### 6. **Backtesting Agent** *(Planned)*
- **Location**: `backtesting/`
- **Purpose**: Validate strategies with historical data
- **Features**:
  - Historical performance analysis
  - Strategy optimization
  - Risk metrics calculation
- **Output**: Performance reports and strategy refinements

### 7. **Execution Agent**
- **Location**: `daily_runner/`
- **Purpose**: Orchestrate daily betting workflow
- **Schedule**: Multiple times per day
- **Workflow**:
  1. Collect fresh odds data
  2. Run line discrepancy analysis
  3. Generate AI predictions
  4. Calculate optimal bet sizes
  5. Filter high-confidence opportunities
  6. Generate bet recommendations
  7. Send alerts/notifications

### 8. **Monitoring Agent**
- **Location**: `monitoring/`
- **Purpose**: Track system performance and results
- **Features**:
  - Real-time alerts
  - Performance dashboards
  - Error tracking
  - Profitability analysis
- **Outputs**: Slack notifications, performance logs

## 🔄 Workflow Process

### **Live Betting Workflow**
```
1. Data Collection Agent → Fetch odds every 15-30 min
2. Line Discrepancy Agent → Scan for profitable differences
3. AI Council Agent → Generate predictions and confidence
4. Value Calculation Agent → Calculate Kelly sizing
5. Execution Agent → Filter and rank opportunities
6. Monitoring Agent → Send alerts and track results
```

### **Development Workflow**
```
1. Backtesting Agent → Validate new strategies
2. AI Council Agent → Test model improvements
3. Value Calculation Agent → Optimize bet sizing
4. Monitoring Agent → Track enhancement performance
```

## 🎯 Agent Communication

### **Data Flow**
- **Odds Data**: Data Collection → Line Discrepancy → Value Calculation
- **Predictions**: AI Council → Value Calculation → Execution
- **Results**: Monitoring → All agents (feedback loop)

### **Trigger Events**
- **New Odds**: Triggers line discrepancy analysis
- **High Discrepancy**: Triggers AI prediction request
- **High Confidence**: Triggers bet recommendation
- **Steam Move**: Triggers immediate alert

## 🔧 Configuration

### **Agent Settings**
- **Line Discrepancy Thresholds**: 15 cents, 5%
- **Kelly Fractional**: 0.25 (quarter Kelly for safety)
- **Minimum Edge**: 5% expected value
- **Maximum Bet**: 3% of bankroll
- **Confidence Threshold**: 70%

### **AI Model Weights**
- **PrimaryModel**: 40%
- **WeatherModel**: 20%
- **TrendModel**: 20%
- **DumbModel**: 10%
- **Contrarian**: 10%

## 🚀 Planned Enhancements

### **Phase 1: Core Improvements**
1. **SGP Agent**: Build same-game parlay opportunities
2. **Real AI Models**: Replace DumbModel with Claude/GPT integration
3. **Steam Detection**: Enhanced line movement tracking

### **Phase 2: Advanced Features**
1. **Backtesting Engine**: Historical validation
2. **Portfolio Optimization**: Multi-game correlation
3. **Live Arbitrage**: Cross-book profit opportunities

### **Phase 3: Automation**
1. **Auto-Execution**: Direct API betting (where legal)
2. **Dynamic Sizing**: Real-time bankroll management
3. **Market Making**: Identify overvalued markets

## 📊 Success Metrics

### **System Performance**
- **ROI**: Target 15-25% annually
- **Sharpe Ratio**: >1.5
- **Max Drawdown**: <10%
- **Win Rate**: >55%

### **Agent Performance**
- **Line Discrepancy**: 5+ opportunities daily
- **AI Accuracy**: >60% prediction accuracy
- **Alert Quality**: <20% false positive rate
- **Response Time**: <2 minutes from odds update

## 🔄 Continuous Improvement

### **Daily Reviews**
- Performance metrics analysis
- Agent effectiveness assessment
- Parameter tuning recommendations

### **Weekly Optimization**
- Model weight adjustments
- Threshold refinements
- Strategy enhancements

### **Monthly Overhauls**
- Major feature additions
- Architecture improvements
- New agent development

---

*This system is designed to be modular, scalable, and continuously improving. Each agent operates independently while contributing to the overall betting strategy.*
