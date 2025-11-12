#!/bin/bash
# NFL Betting System - Restore Script
#
# WHY THIS EXISTS:
# Recovers betting system from backup in case of data loss or corruption
#
# WHAT IT RESTORES:
# 1. Betting data (bet_log.json, bankroll files)
# 2. Training data (8.3MB)
# 3. Models (optimal_llm_weights.json, configs)
# 4. System state (circuit breaker, monitoring)
# 5. Configuration files
#
# USAGE:
#   ./nfl_restore.sh /path/to/backup.tar.gz
#   ./nfl_restore.sh /path/to/backup_directory

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No backup specified${NC}"
    echo ""
    echo "Usage: $0 <backup_file_or_directory>"
    echo ""
    echo "Examples:"
    echo "  $0 ~/nfl_betting_backups/nfl_backup_20251112_103000.tar.gz"
    echo "  $0 ~/nfl_betting_backups/nfl_backup_20251112_103000"
    exit 1
fi

BACKUP_SOURCE="$1"
SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR=$(mktemp -d)

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}NFL BETTING SYSTEM - RESTORE${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Check if backup exists
if [ ! -e "$BACKUP_SOURCE" ]; then
    echo -e "${RED}Error: Backup not found: $BACKUP_SOURCE${NC}"
    exit 1
fi

# Extract if tar.gz, otherwise use directory directly
if [[ "$BACKUP_SOURCE" == *.tar.gz ]]; then
    echo -e "${BLUE}Step 1: Extracting backup archive...${NC}"
    tar -xzf "$BACKUP_SOURCE" -C "$TEMP_DIR"
    BACKUP_DIR=$(find "$TEMP_DIR" -maxdepth 1 -type d -name "nfl_backup_*" | head -1)
    echo -e "${GREEN}  ✓${NC} Extracted to $TEMP_DIR"
else
    BACKUP_DIR="$BACKUP_SOURCE"
    echo -e "${BLUE}Step 1: Using backup directory...${NC}"
    echo -e "${GREEN}  ✓${NC} $BACKUP_DIR"
fi
echo ""

# Verify backup
if [ ! -f "$BACKUP_DIR/MANIFEST.txt" ]; then
    echo -e "${RED}Error: Invalid backup (no MANIFEST.txt found)${NC}"
    exit 1
fi

echo -e "${BLUE}Step 2: Verifying backup...${NC}"
echo -e "${YELLOW}Backup Manifest:${NC}"
cat "$BACKUP_DIR/MANIFEST.txt" | head -10
echo ""

# Confirm restore
read -p "$(echo -e ${YELLOW}Restore this backup to ${SYSTEM_DIR}? [y/N]: ${NC})" -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Restore cancelled${NC}"
    rm -rf "$TEMP_DIR"
    exit 0
fi
echo ""

# Create backup of current system before restoring
echo -e "${BLUE}Step 3: Backing up current system (safety)...${NC}"
SAFETY_BACKUP="${HOME}/nfl_betting_backups/pre_restore_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SAFETY_BACKUP"

# Backup current critical files
for file in bet_log.json bankroll.json optimal_llm_weights.json .circuit_breaker; do
    if [ -f "${SYSTEM_DIR}/$file" ]; then
        cp "${SYSTEM_DIR}/$file" "$SAFETY_BACKUP/"
        echo -e "${GREEN}  ✓${NC} Saved current $file"
    fi
done

if [ -d "${SYSTEM_DIR}/data" ]; then
    cp -r "${SYSTEM_DIR}/data" "$SAFETY_BACKUP/"
    echo -e "${GREEN}  ✓${NC} Saved current data directory"
fi

echo -e "${GREEN}Current system backed up to: $SAFETY_BACKUP${NC}"
echo ""

# Restore critical betting files
echo -e "${BLUE}Step 4: Restoring critical betting files...${NC}"
CRITICAL_FILES=(
    "bet_log.json"
    "bankroll.json"
    "bankroll_tracker.py"
    "optimal_llm_weights.json"
    ".circuit_breaker"
)

RESTORED=0
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$BACKUP_DIR/$file" ]; then
        cp "$BACKUP_DIR/$file" "${SYSTEM_DIR}/"
        echo -e "${GREEN}  ✓${NC} Restored $file"
        RESTORED=$((RESTORED + 1))
    fi
done
echo -e "${GREEN}Restored $RESTORED critical files${NC}"
echo ""

# Restore data directory
echo -e "${BLUE}Step 5: Restoring data directory...${NC}"
if [ -d "$BACKUP_DIR/data" ]; then
    # Remove current data directory
    rm -rf "${SYSTEM_DIR}/data"

    # Restore from backup
    cp -r "$BACKUP_DIR/data" "${SYSTEM_DIR}/"

    DATA_SIZE=$(du -sh "${SYSTEM_DIR}/data" | cut -f1)
    echo -e "${GREEN}  ✓${NC} Data directory restored ($DATA_SIZE)"
else
    echo -e "${YELLOW}  ⚠${NC} No data directory in backup"
fi
echo ""

# Restore models
echo -e "${BLUE}Step 6: Restoring models...${NC}"
if [ -d "$BACKUP_DIR/models" ]; then
    rm -rf "${SYSTEM_DIR}/models"
    cp -r "$BACKUP_DIR/models" "${SYSTEM_DIR}/"

    MODEL_COUNT=$(find "${SYSTEM_DIR}/models" -type f | wc -l)
    echo -e "${GREEN}  ✓${NC} Restored $MODEL_COUNT model files"
else
    echo -e "${YELLOW}  ⚠${NC} No models directory in backup"
fi
echo ""

# Restore system files
echo -e "${BLUE}Step 7: Restoring system files...${NC}"
SYSTEM_FILES=(
    "auto_execute_bets.py"
    "circuit_breaker.py"
    "contrarian_intelligence.py"
    "trap_detector.py"
    "deepseek_contrarian_analysis.py"
    "bet_validator.py"
    "nfl_system_monitor.py"
)

SYSTEM_RESTORED=0
for file in "${SYSTEM_FILES[@]}"; do
    if [ -f "$BACKUP_DIR/$file" ]; then
        cp "$BACKUP_DIR/$file" "${SYSTEM_DIR}/"
        SYSTEM_RESTORED=$((SYSTEM_RESTORED + 1))
    fi
done
echo -e "${GREEN}  ✓${NC} Restored $SYSTEM_RESTORED system files"
echo ""

# Restore monitoring data
echo -e "${BLUE}Step 8: Restoring monitoring data...${NC}"
if [ -d "$BACKUP_DIR/data/monitoring" ]; then
    mkdir -p "${SYSTEM_DIR}/data/monitoring"
    cp -r "$BACKUP_DIR/data/monitoring/"* "${SYSTEM_DIR}/data/monitoring/" 2>/dev/null || true
    MONITOR_COUNT=$(find "${SYSTEM_DIR}/data/monitoring" -type f | wc -l)
    echo -e "${GREEN}  ✓${NC} Restored $MONITOR_COUNT monitoring files"
else
    echo -e "${YELLOW}  ⚠${NC} No monitoring data in backup"
fi
echo ""

# Cleanup
rm -rf "$TEMP_DIR"

# Verify restored system
echo -e "${BLUE}Step 9: Verifying restored system...${NC}"

VERIFY_OK=true

# Check critical files exist
for file in bet_log.json bankroll_tracker.py optimal_llm_weights.json; do
    if [ -f "${SYSTEM_DIR}/$file" ]; then
        echo -e "${GREEN}  ✓${NC} $file exists"
    else
        echo -e "${RED}  ✗${NC} $file missing"
        VERIFY_OK=false
    fi
done

# Check data directory
if [ -d "${SYSTEM_DIR}/data" ]; then
    echo -e "${GREEN}  ✓${NC} Data directory exists"
else
    echo -e "${RED}  ✗${NC} Data directory missing"
    VERIFY_OK=false
fi

echo ""

# Summary
if [ "$VERIFY_OK" = true ]; then
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}RESTORE COMPLETE ✓${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo -e "${YELLOW}Restored Files:${NC}"
    echo "  Critical files: $RESTORED"
    echo "  System files: $SYSTEM_RESTORED"
    echo "  Data directory: $DATA_SIZE"
    echo "  Models: $MODEL_COUNT"
    echo "  Monitoring files: $MONITOR_COUNT"
    echo ""
    echo -e "${YELLOW}Safety Backup:${NC}"
    echo "  Your previous system saved to:"
    echo "  $SAFETY_BACKUP"
    echo ""
    echo -e "${GREEN}NFL betting system restored successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Verify bankroll: python bankroll_tracker.py --check"
    echo "  2. Check system monitor: python nfl_system_monitor.py --status"
    echo "  3. Resume betting: python auto_execute_bets.py --auto"
else
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}RESTORE INCOMPLETE ✗${NC}"
    echo -e "${RED}======================================${NC}"
    echo ""
    echo -e "${RED}Some files were not restored successfully.${NC}"
    echo -e "${YELLOW}Your previous system is saved at:${NC}"
    echo "  $SAFETY_BACKUP"
    echo ""
    echo -e "${YELLOW}You can restore it with:${NC}"
    echo "  ./nfl_restore.sh $SAFETY_BACKUP"
    exit 1
fi
