#!/bin/bash
################################################################################
# Run All Market Spread Scrapers
################################################################################
#
# This script runs multiple scrapers in parallel to get COMPREHENSIVE
# historical market spread coverage from 2015-2024.
#
# USAGE:
#     chmod +x run_all_scrapers.sh
#     ./run_all_scrapers.sh
#
# TIME: ~8-10 hours (runs overnight)
# RESULT: 10 years of FREE historical market spreads
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "🚀 NCAA HISTORICAL MARKET SPREAD SCRAPER"
echo "================================================================================"
echo ""
echo "This will scrape 2015-2024 closing spreads from multiple sources:"
echo "  • TeamRankings.com (closing lines)"
echo "  • Covers.com (historical matchups)"
echo "  • Archive.org (historical snapshots)"
echo ""
echo "Estimated time: 8-10 hours (run overnight)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Create output directory
mkdir -p data/market_spreads
mkdir -p logs

# Log file
LOGFILE="logs/scraper_$(date +%Y%m%d_%H%M%S).log"

echo "" | tee -a "$LOGFILE"
echo "================================================================================" | tee -a "$LOGFILE"
echo "📅 Starting scrape at $(date)" | tee -a "$LOGFILE"
echo "================================================================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Function to scrape a single year
scrape_year() {
    year=$1
    echo "" | tee -a "$LOGFILE"
    echo "================================================================================" | tee -a "$LOGFILE"
    echo "📊 SCRAPING $year SEASON" | tee -a "$LOGFILE"
    echo "================================================================================" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"

    # Run scrapers in parallel for this year
    echo "🌐 Launching scrapers for $year..." | tee -a "$LOGFILE"

    # TeamRankings (full season)
    python scrape_teamrankings_historical.py $year >> "$LOGFILE" 2>&1 &
    PID_TR=$!

    # Covers (week by week)
    python scrape_covers_historical.py $year >> "$LOGFILE" 2>&1 &
    PID_COVERS=$!

    # Archive.org (if available)
    if [ -f "wayback_spreads_scraper.py" ]; then
        python wayback_spreads_scraper.py $year >> "$LOGFILE" 2>&1 &
        PID_ARCHIVE=$!
    fi

    # Wait for all scrapers to finish
    echo "⏳ Waiting for scrapers to complete..." | tee -a "$LOGFILE"
    wait $PID_TR
    wait $PID_COVERS
    [ ! -z "$PID_ARCHIVE" ] && wait $PID_ARCHIVE

    echo "✅ $year complete" | tee -a "$LOGFILE"

    # Small delay between years
    sleep 5
}

# Scrape all years (2015-2024)
for year in {2015..2024}; do
    scrape_year $year
done

echo "" | tee -a "$LOGFILE"
echo "================================================================================" | tee -a "$LOGFILE"
echo "🔗 COMBINING ALL DATA" | tee -a "$LOGFILE"
echo "================================================================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Combine all scraped data
python combine_scraped_data.py | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "================================================================================" | tee -a "$LOGFILE"
echo "✅ SCRAPING COMPLETE" | tee -a "$LOGFILE"
echo "================================================================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Finished at: $(date)" | tee -a "$LOGFILE"
echo "Log saved to: $LOGFILE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Next steps:" | tee -a "$LOGFILE"
echo "1. Check data/market_spreads_YEAR.csv files" | tee -a "$LOGFILE"
echo "2. Run: python backtest_ncaa_parlays_REALISTIC.py" | tee -a "$LOGFILE"
echo "3. Get real ROI with actual market spreads!" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
