#!/bin/bash
# NFL Self-Improving Loop Cron Job for Raspberry Pi
# Runs every Sunday at 11 PM EST (after all games complete)

# Set environment
export PATH="/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="/home/ryan/code/football_betting_system"
cd /home/ryan/code/football_betting_system

# Log start
echo "$(date): Starting NFL self-improving loop" >> /var/log/nfl_betting.log

# Run self-improving loop with timeout (30 min max)
timeout 1800 python3 self_improving_loop.py >> /var/log/nfl_betting.log 2>&1

# Check exit status
if [ $? -eq 124 ]; then
    echo "$(date): NFL loop timed out after 30 minutes" >> /var/log/nfl_betting.log
elif [ $? -eq 0 ]; then
    echo "$(date): NFL loop completed successfully" >> /var/log/nfl_betting.log
else
    echo "$(date): NFL loop failed with exit code $?" >> /var/log/nfl_betting.log
fi

# Send completion notification
curl -X POST "https://api.pushover.net/1/messages.json" \
  -d "token=YOUR_PUSHOVER_TOKEN" \
  -d "user=YOUR_PUSHOVER_USER" \
  -d "message=NFL self-improving loop completed: $(date)" \
  -d "title=NFL Betting System Update" \
  2>/dev/null

# Cleanup old logs (keep last 30 days)
find /var/log -name "nfl_betting.log.*" -mtime +30 -delete 2>/dev/null

# Add to crontab with: crontab -e
# 0 23 * * 0 /home/ryan/code/football_betting_system/nfl_cron_job.sh
