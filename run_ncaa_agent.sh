#!/bin/bash
# NCAA Betting Agent Runner
# This script can be run via cron for automated operation

# Activate virtual environment if needed
# source venv/bin/activate

# Change to project directory
cd "$(dirname "$0")"

# Run the agent
python3 ncaa_agent.py

# Log the run
echo "Agent run completed at $(date)" >> data/agents/ncaa/agent.log
