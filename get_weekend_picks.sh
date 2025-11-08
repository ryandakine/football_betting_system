#!/bin/bash
# Get NFL Predictions for Weekend Games
# Runs before Sunday/Monday games

echo "══════════════════════════════════════════════════════════════"
echo "🏈 NFL WEEKEND PREDICTIONS (Sunday + Monday Night)"
echo "══════════════════════════════════════════════════════════════"
echo ""

# Check AWS credentials
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured!"
    echo ""
    echo "Run: aws configure"
    echo "Then try again."
    exit 1
fi

echo "✅ AWS credentials valid"
echo ""

# Call Lambda for predictions
echo "══════════════════════════════════════════════════════════════"
echo "Calling AWS Lambda for all weekend games..."
echo "══════════════════════════════════════════════════════════════"
echo ""

python3 call_lambda_predictions.py

if [ $? -eq 0 ]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "✅ PREDICTIONS COMPLETE!"
    echo "══════════════════════════════════════════════════════════════"
    echo ""
    echo "📊 Review your picks:"
    echo "   • Check terminal output above"
    echo "   • See data/lambda_predictions_*.json for details"
    echo ""
    echo "🏈 Games this weekend:"
    echo "   • Sunday: Full slate of games (1pm, 4pm, 8pm ET)"
    echo "   • Monday Night Football"
    echo ""
    echo "Good luck! 🍀"
    echo "══════════════════════════════════════════════════════════════"
else
    echo ""
    echo "❌ Lambda call failed!"
    echo "Check error messages above."
fi
