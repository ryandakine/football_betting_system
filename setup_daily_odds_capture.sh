#!/bin/bash
################################################################################
# Setup Daily Odds Capture Cronjob
# Automatically captures NFL odds every day at noon
################################################################################

set -e

echo "🏈 Setting Up Daily Odds Capture"
echo "="*80

# Get current directory
BETTING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Betting system directory: $BETTING_DIR"
echo ""

# Check if capture script exists
if [ ! -f "$BETTING_DIR/capture_daily_odds.py" ]; then
    echo "❌ Error: capture_daily_odds.py not found"
    exit 1
fi

echo "✅ Found capture_daily_odds.py"

# Make executable
chmod +x "$BETTING_DIR/capture_daily_odds.py"

# Check Python dependencies
echo ""
echo "Checking dependencies..."
python3 -c "import requests" 2>/dev/null || {
    echo "⚠️  requests library not installed"
    echo "   Run: pip install requests"
}

echo "✅ Dependencies OK"

# Check API key
echo ""
echo "Checking API key..."
if [ -z "$ODDS_API_KEY" ] && [ -z "$THE_ODDS_API_KEY" ]; then
    echo "⚠️  No API key found"
    echo "   Set it with: export ODDS_API_KEY='your_key_here'"
    echo "   Add to ~/.bashrc or ~/.zshrc to make permanent"
else
    echo "✅ API key configured"
fi

# Create cronjob
echo ""
echo "Setting up cronjob..."

# Create temporary cron file
CRON_CMD="0 12 * * * cd $BETTING_DIR && /usr/bin/python3 capture_daily_odds.py >> $BETTING_DIR/logs/odds_capture.log 2>&1"

# Check if cronjob already exists
if crontab -l 2>/dev/null | grep -q "capture_daily_odds.py"; then
    echo "⚠️  Cronjob already exists"
    echo "   Current crontab:"
    crontab -l | grep "capture_daily_odds.py"
else
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "✅ Cronjob added"
    echo "   Schedule: Every day at 12:00 PM (noon)"
    echo "   Command: $CRON_CMD"
fi

# Create logs directory
mkdir -p "$BETTING_DIR/logs"

# Test run
echo ""
echo "Testing odds capture..."
if python3 "$BETTING_DIR/capture_daily_odds.py"; then
    echo "✅ Test run successful"
else
    echo "❌ Test run failed"
    echo "   Check your API key and internet connection"
fi

echo ""
echo "="*80
echo "✅ Daily Odds Capture Setup Complete!"
echo "="*80
echo ""
echo "Cronjob Details:"
echo "  Schedule: Every day at 12:00 PM (noon)"
echo "  Script: $BETTING_DIR/capture_daily_odds.py"
echo "  Logs: $BETTING_DIR/logs/odds_capture.log"
echo ""
echo "To view crontab:"
echo "  crontab -l"
echo ""
echo "To remove cronjob:"
echo "  crontab -e  # Then delete the line"
echo ""
echo "To manually run:"
echo "  python3 $BETTING_DIR/capture_daily_odds.py"
echo ""
echo "="*80
