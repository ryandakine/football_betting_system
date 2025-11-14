#!/bin/bash
#
# NFL Weekend Quick Start Script
# Automates the entire NFL betting workflow
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

NFL_BANKROLL=${NFL_BANKROLL:-20}
KELLY_FRACTION=${KELLY_FRACTION:-0.25}

# Functions
print_header() {
    echo ""
    echo "================================================================"
    echo -e "${BLUE}$1${NC}"
    echo "================================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

check_dependencies() {
    print_header "Checking Dependencies"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found"
        exit 1
    fi
    print_success "Python3 installed"

    # Check pip packages
    python3 -c "import crawlbase" 2>/dev/null && print_success "Crawlbase installed" || {
        print_warning "Crawlbase not installed"
        echo "Installing crawlbase..."
        pip install crawlbase
    }

    # Check for Crawlbase token
    if [ -z "$CRAWLBASE_TOKEN" ]; then
        print_error "CRAWLBASE_TOKEN not set"
        echo ""
        echo "Setup instructions:"
        echo "1. Sign up at https://crawlbase.com/signup"
        echo "2. Get your free API token"
        echo "3. Run: export CRAWLBASE_TOKEN='your_token_here'"
        echo "   OR add to .env file"
        exit 1
    fi
    print_success "Crawlbase token configured"
}

collect_data() {
    print_header "Collecting NFL Data"

    python3 crawlbase_nfl_scraper.py --no-odds

    if [ $? -eq 0 ]; then
        print_success "Data collection complete"
    else
        print_error "Data collection failed"
        exit 1
    fi
}

run_analysis() {
    print_header "Running NFL Analysis"

    if [ -f "unified_nfl_intelligence_system.py" ]; then
        print_success "Running unified NFL intelligence system..."
        python3 unified_nfl_intelligence_system.py | tee nfl_analysis_latest.txt
    elif [ -f "main.py" ]; then
        print_success "Running main analysis..."
        python3 main.py | tee nfl_analysis_latest.txt
    else
        print_warning "No analysis script found"
        echo "Available scripts:"
        ls -1 *main*.py 2>/dev/null || echo "None found"
        return 1
    fi
}

calculate_bets() {
    print_header "Calculating Bet Sizes"

    python3 kelly_calculator.py \
        --bankroll "$NFL_BANKROLL" \
        --fraction "$KELLY_FRACTION"

    print_success "Kelly sizing complete"
}

show_summary() {
    print_header "NFL Weekend Summary"

    echo "📊 Bankroll: \$$NFL_BANKROLL"
    echo "📈 Kelly Fraction: $KELLY_FRACTION (fractional Kelly)"
    echo ""
    echo "📁 Files generated:"
    echo "   - nfl_crawlbase_data_*.json (latest data)"
    echo "   - nfl_analysis_latest.txt (predictions)"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Review predictions in nfl_analysis_latest.txt"
    echo "   2. Filter for STRONG_BET picks (65%+ confidence)"
    echo "   3. Line shop across DraftKings, FanDuel, BetMGM"
    echo "   4. Place bets 15 minutes before kickoff"
    echo ""
    echo "⏰ Game times (Sunday):"
    echo "   - Early games: 1:00 PM ET"
    echo "   - Late games: 4:05/4:25 PM ET"
    echo "   - Sunday Night: 8:20 PM ET"
    echo ""
}

# Main workflow
main() {
    print_header "🏈 NFL Weekend Quick Start"

    echo "This script will:"
    echo "1. Check dependencies"
    echo "2. Collect NFL data via Crawlbase"
    echo "3. Run prediction analysis"
    echo "4. Calculate Kelly bet sizes"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 0
    fi

    check_dependencies
    collect_data
    run_analysis
    calculate_bets
    show_summary

    print_success "All done! Ready to bet 🎉"
}

# Show usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --bankroll NUM     NFL bankroll (default: \$20)"
    echo "  --kelly-fraction   Kelly fraction (default: 0.25)"
    echo "  --help            Show this help"
    echo ""
    echo "Environment variables:"
    echo "  CRAWLBASE_TOKEN    Crawlbase API token (required)"
    echo "  NFL_BANKROLL       NFL bankroll (default: \$20)"
    echo "  KELLY_FRACTION     Kelly fraction (default: 0.25)"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bankroll)
            NFL_BANKROLL="$2"
            shift 2
            ;;
        --kelly-fraction)
            KELLY_FRACTION="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main
main
