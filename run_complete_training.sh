#!/bin/bash
# Complete AI Council Training Pipeline
# Runs all steps to train on 10 years of NFL data with ALL features

echo "🏈 NFL AI COUNCIL - COMPLETE TRAINING PIPELINE"
echo "============================================================"
echo ""

# Step 1: Collect 10 years of historical data
echo "📊 Step 1: Collecting 10 years of NFL historical data..."
echo "   This will take 20-30 minutes..."
python3 collect_historical_nfl.py
if [ $? -ne 0 ]; then
    echo "❌ Data collection failed!"
    exit 1
fi
echo "✅ Data collection complete!"
echo ""

# Step 2: Integrate ALL advanced features
echo "🚀 Step 2: Integrating advanced features..."
echo "   - EPA (Expected Points Added)"
echo "   - DVOA (Defense-adjusted Value Over Average)"
echo "   - Recent ATS Performance"
echo "   - Line Movement"
echo "   - Team Chemistry"
echo "   - Agent Influence"
python3 integrate_all_features.py
if [ $? -ne 0 ]; then
    echo "❌ Feature integration failed!"
    exit 1
fi
echo "✅ Feature integration complete!"
echo ""

# Step 3: Train AI Council
echo "🧠 Step 3: Training AI Council with enhanced features..."
echo "   Training 4 specialized models..."
python3 train_ai_council.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi
echo "✅ Training complete!"
echo ""

# Summary
echo "============================================================"
echo "🎯 AI COUNCIL TRAINING COMPLETE!"
echo ""
echo "📊 Features Integrated:"
echo "   ✅ Weather (temperature, wind, precipitation)"
echo "   ✅ Injuries (position-weighted)"
echo "   ✅ Rest & Travel (fatigue analysis)"
echo "   ✅ Referee Crews (bias detection)"
echo "   ✅ EPA per Play (expected points)"
echo "   ✅ DVOA (opponent-adjusted metrics)"
echo "   ✅ Recent ATS Performance (last 5 games)"
echo "   ✅ Line Movement (sharp money)"
echo "   ✅ Team Chemistry (continuity)"
echo "   ✅ Agent Influence (conflict detection)"
echo ""
echo "🤖 Models Trained:"
echo "   ✅ Spread Expert"
echo "   ✅ Total Expert"
echo "   ✅ Contrarian Model"
echo "   ✅ Home Advantage Model"
echo ""
echo "💾 Models saved to: models/"
echo "📁 Training data: data/nfl_training_data_enhanced.json"
echo ""
echo "🚀 Ready to deploy to AWS Lambda!"
echo "============================================================"
