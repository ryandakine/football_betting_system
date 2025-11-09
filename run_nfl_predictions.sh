#!/bin/bash

echo "═══════════════════════════════════════════════════════════════"
echo "🏈 NFL BETTING SYSTEM - COMPLETE PIPELINE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "This script will:"
echo "  1. Scrape live NFL odds from multiple sources"
echo "  2. Run AI analysis (Claude, GPT-4, Grok, Perplexity)"
echo "  3. Generate betting recommendations"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if we're in the right directory
if [ ! -f "nfl_odds_scraper.py" ]; then
    echo "❌ Error: Must run from football_betting_system directory"
    exit 1
fi

# Install required packages
echo "📦 Checking dependencies..."
pip3 install beautifulsoup4 lxml --quiet 2>/dev/null

# Step 1: Scrape odds
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Scraping NFL Odds"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 nfl_odds_scraper.py

if [ $? -ne 0 ]; then
    echo "❌ Scraping failed!"
    exit 1
fi

# Step 2: Analyze odds
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: AI Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 analyze_scraped_odds.py

if [ $? -ne 0 ]; then
    echo "❌ Analysis failed!"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ ALL DONE!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Check data/nfl_analysis_results.json for detailed results"
echo ""
