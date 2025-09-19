#!/bin/bash
# Pi Injury Monitor Cron - Runs every 5 minutes during games
cd /home/ryan/code/football_betting_system
python3 pi_injury_monitor.py >> /var/log/injury_intel.log 2>&1

# Add to Pi crontab:
# */5 * * * 0,1,4 /home/ryan/code/football_betting_system/pi_injury_cron.sh
# (Every 5 minutes on Sunday, Monday, Thursday - game days)
