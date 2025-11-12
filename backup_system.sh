#!/bin/bash
"""
Football Betting System - Backup Script
========================================

Backs up critical data to protect against HD failure:
- Trained models (11 models)
- Historical game data (21,522 games)
- Backtest results
- CLV/line shopping history

USAGE:
    ./backup_system.sh

Default: Backs up to ~/Backups/football_betting/
Edit BACKUP_DIR to change location (Google Drive, Dropbox, etc.)
"""

# Configuration
BACKUP_DIR="$HOME/Backups/football_betting"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/backup_$TIMESTAMP"

echo "🔄 Football Betting System Backup"
echo "=================================="
echo ""
echo "Backup location: $BACKUP_PATH"
echo ""

# Create backup directory
mkdir -p "$BACKUP_PATH"

# 1. CRITICAL: Historical game data
echo "📊 Backing up historical data..."
if [ -d "data" ]; then
    cp -r data "$BACKUP_PATH/"
    DATA_SIZE=$(du -sh "$BACKUP_PATH/data" | cut -f1)
    echo "   ✅ Data backed up ($DATA_SIZE)"
else
    echo "   ⚠️  No data directory found"
fi

# 2. IMPORTANT: Trained models
echo "🤖 Backing up trained models..."
if [ -d "models" ]; then
    cp -r models "$BACKUP_PATH/"
    MODEL_COUNT=$(find "$BACKUP_PATH/models" -name "*.pkl" | wc -l)
    echo "   ✅ Models backed up ($MODEL_COUNT .pkl files)"
else
    echo "   ⚠️  No models directory found"
fi

# 3. NICE: Backtest results
echo "📈 Backing up backtest results..."
if [ -d "backtest_results" ]; then
    cp -r backtest_results "$BACKUP_PATH/"
    echo "   ✅ Backtest results backed up"
fi

# Also backup individual result files
if ls *_backtest_results_*.json 1> /dev/null 2>&1; then
    mkdir -p "$BACKUP_PATH/backtest_results"
    cp *_backtest_results_*.json "$BACKUP_PATH/backtest_results/"
    echo "   ✅ Individual result files backed up"
fi

# 4. Configuration files
echo "⚙️  Backing up configuration..."
for file in *.json *.py *.md; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_PATH/"
    fi
done
echo "   ✅ Config files backed up"

# Summary
echo ""
echo "=================================="
echo "✅ BACKUP COMPLETE"
echo "=================================="
echo ""

TOTAL_SIZE=$(du -sh "$BACKUP_PATH" | cut -f1)
FILE_COUNT=$(find "$BACKUP_PATH" -type f | wc -l)

echo "📊 Backup Summary:"
echo "   Location: $BACKUP_PATH"
echo "   Total size: $TOTAL_SIZE"
echo "   Files: $FILE_COUNT"
echo ""

# Keep only last 7 backups
echo "🧹 Cleaning old backups (keeping last 7)..."
cd "$BACKUP_DIR"
ls -t | tail -n +8 | xargs rm -rf 2>/dev/null
REMAINING=$(ls | wc -l)
echo "   ✅ $REMAINING backups remaining"
echo ""

echo "💡 RESTORE INSTRUCTIONS:"
echo "   If HD fails, restore with:"
echo "   cp -r $BACKUP_PATH/* /path/to/football_betting_system/"
echo ""

# Optional: Sync to cloud
if [ -d "$HOME/GoogleDrive" ]; then
    echo "☁️  Syncing to Google Drive..."
    rsync -av "$BACKUP_PATH/" "$HOME/GoogleDrive/football_betting_backup/" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "   ✅ Synced to Google Drive"
    fi
elif [ -d "$HOME/Dropbox" ]; then
    echo "☁️  Syncing to Dropbox..."
    rsync -av "$BACKUP_PATH/" "$HOME/Dropbox/football_betting_backup/" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "   ✅ Synced to Dropbox"
    fi
fi

echo ""
echo "🎯 NEXT STEPS:"
echo "   1. Verify backup: ls -lh $BACKUP_PATH"
echo "   2. Test restore: cp -r $BACKUP_PATH/data /tmp/test_restore/"
echo "   3. Setup cron: crontab -e"
echo "      Add: 0 2 * * * cd $(pwd) && ./backup_system.sh"
echo ""
