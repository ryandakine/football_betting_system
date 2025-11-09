#!/bin/bash
###############################################################################
# NFL Betting Agent - Weekly Automation Setup
###############################################################################
# Sets up automated weekly analysis via cron job
#
# The agent will run every Thursday at 2 PM (when referee assignments are posted)
# and generate the full weekly report automatically.
#
# Usage:
#   ./setup_weekly_automation.sh install    # Install cron job
#   ./setup_weekly_automation.sh remove     # Remove cron job
#   ./setup_weekly_automation.sh test       # Test manual run
###############################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AGENT_SCRIPT="${SCRIPT_DIR}/autonomous_betting_agent.py"
LOG_FILE="${SCRIPT_DIR}/logs/agent_cron.log"

# Ensure log directory exists
mkdir -p "${SCRIPT_DIR}/logs"

# Cron schedule: Every Thursday at 2 PM
# Minute Hour Day Month DayOfWeek
CRON_SCHEDULE="0 14 * * 4"

# Full cron command
CRON_CMD="cd ${SCRIPT_DIR} && /usr/bin/python3 ${AGENT_SCRIPT} --auto >> ${LOG_FILE} 2>&1"

install_cron() {
    echo "Installing NFL Betting Agent to cron..."

    # Check if already installed
    if crontab -l 2>/dev/null | grep -q "autonomous_betting_agent.py"; then
        echo "⚠️  Agent is already installed in cron"
        echo "Run './setup_weekly_automation.sh remove' first to reinstall"
        return 1
    fi

    # Add to crontab
    (crontab -l 2>/dev/null; echo "${CRON_SCHEDULE} ${CRON_CMD}") | crontab -

    echo "✅ NFL Betting Agent installed!"
    echo ""
    echo "Schedule: Every Thursday at 2:00 PM"
    echo "Log file: ${LOG_FILE}"
    echo ""
    echo "The agent will:"
    echo "  1. Auto-detect current NFL week"
    echo "  2. Analyze all games (spreads, totals, ML, 1H, team totals)"
    echo "  3. Analyze all player props (yards, TDs, receptions)"
    echo "  4. Generate master report"
    echo "  5. Save to reports/week_XX_master_report.txt"
    echo ""
    echo "You'll get a fresh report every Thursday! 🏈💰"
}

remove_cron() {
    echo "Removing NFL Betting Agent from cron..."

    # Remove from crontab
    crontab -l 2>/dev/null | grep -v "autonomous_betting_agent.py" | crontab -

    echo "✅ Agent removed from cron"
}

test_run() {
    echo "Testing manual agent run..."
    echo ""

    cd "${SCRIPT_DIR}"
    python3 "${AGENT_SCRIPT}" --auto

    echo ""
    echo "✅ Test complete!"
    echo "Check reports/ directory for output"
}

show_status() {
    echo "NFL Betting Agent - Automation Status"
    echo "======================================"
    echo ""

    if crontab -l 2>/dev/null | grep -q "autonomous_betting_agent.py"; then
        echo "Status: ✅ INSTALLED"
        echo ""
        echo "Cron schedule:"
        crontab -l 2>/dev/null | grep "autonomous_betting_agent.py"
        echo ""
        echo "Next run: Thursday at 2:00 PM"
    else
        echo "Status: ❌ NOT INSTALLED"
        echo ""
        echo "Run './setup_weekly_automation.sh install' to enable automation"
    fi

    echo ""
    echo "Recent runs:"
    if [ -f "${LOG_FILE}" ]; then
        tail -20 "${LOG_FILE}"
    else
        echo "No runs yet"
    fi
}

# Main
case "$1" in
    install)
        install_cron
        ;;
    remove)
        remove_cron
        ;;
    test)
        test_run
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {install|remove|test|status}"
        echo ""
        echo "Commands:"
        echo "  install  - Install weekly automation (runs every Thursday)"
        echo "  remove   - Remove automation"
        echo "  test     - Test manual run"
        echo "  status   - Show current status"
        exit 1
        ;;
esac
