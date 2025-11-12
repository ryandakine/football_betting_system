#!/bin/bash
# NFL Betting System - Automated Backup Script
#
# WHY THIS EXISTS:
# Protects 8.3MB of betting data, bankroll, models, and system state
#
# WHAT IT BACKS UP:
# 1. Betting data (bet_log.json, bankroll files)
# 2. Training data (8.3MB in data/)
# 3. Models (optimal_llm_weights.json, model configs)
# 4. System state (circuit breaker, monitoring data)
# 5. Configuration files
#
# USAGE:
#   ./nfl_backup_system.sh                    # Local backup
#   ./nfl_backup_system.sh --remote s3://...  # Remote backup to S3
#   ./nfl_backup_system.sh --remote /mnt/backup  # Remote backup to mount

set -e  # Exit on error

# Configuration
BACKUP_DIR="${HOME}/nfl_betting_backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="nfl_backup_${TIMESTAMP}"
SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
REMOTE_BACKUP=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --remote)
            REMOTE_BACKUP="$2"
            shift 2
            ;;
        --help)
            echo "NFL Betting System Backup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --remote LOCATION    Copy backup to remote location (S3, mounted drive, etc.)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Local backup only"
            echo "  $0 --remote s3://my-bucket/nfl       # Backup to S3"
            echo "  $0 --remote /mnt/backup              # Backup to mounted drive"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}NFL BETTING SYSTEM - BACKUP${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "${YELLOW}Timestamp:${NC} $TIMESTAMP"
echo -e "${YELLOW}Backup name:${NC} $BACKUP_NAME"
echo -e "${YELLOW}System dir:${NC} $SYSTEM_DIR"
echo ""

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"
cd "${SYSTEM_DIR}"

echo -e "${BLUE}Step 1: Backing up betting data...${NC}"

# Backup critical betting files
CRITICAL_FILES=(
    "bet_log.json"
    "bankroll.json"
    "bankroll_tracker.py"
    "optimal_llm_weights.json"
    ".circuit_breaker"
)

CRITICAL_BACKED_UP=0
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "${BACKUP_DIR}/${BACKUP_NAME}/"
        echo -e "${GREEN}  ✓${NC} $file"
        CRITICAL_BACKED_UP=$((CRITICAL_BACKED_UP + 1))
    else
        echo -e "${YELLOW}  ⚠${NC} $file (not found, skipping)"
    fi
done

echo -e "${GREEN}Backed up $CRITICAL_BACKED_UP critical files${NC}"
echo ""

# Backup data directory
echo -e "${BLUE}Step 2: Backing up data directory (8.3MB)...${NC}"
if [ -d "data" ]; then
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}/data"

    # Backup subdirectories
    for subdir in data/*/; do
        if [ -d "$subdir" ]; then
            dirname=$(basename "$subdir")
            cp -r "$subdir" "${BACKUP_DIR}/${BACKUP_NAME}/data/"
            size=$(du -sh "$subdir" | cut -f1)
            echo -e "${GREEN}  ✓${NC} data/$dirname ($size)"
        fi
    done

    # Backup data files
    for file in data/*.json data/*.sqlite; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            cp "$file" "${BACKUP_DIR}/${BACKUP_NAME}/data/"
            size=$(du -sh "$file" | cut -f1)
            echo -e "${GREEN}  ✓${NC} data/$filename ($size)"
        fi
    done

    DATA_SIZE=$(du -sh "${BACKUP_DIR}/${BACKUP_NAME}/data" | cut -f1)
    echo -e "${GREEN}Data directory backed up: $DATA_SIZE${NC}"
else
    echo -e "${YELLOW}  ⚠${NC} No data directory found"
fi
echo ""

# Backup models directory
echo -e "${BLUE}Step 3: Backing up models...${NC}"
if [ -d "models" ]; then
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}/models"
    cp -r models/* "${BACKUP_DIR}/${BACKUP_NAME}/models/" 2>/dev/null || true
    MODEL_COUNT=$(find "${BACKUP_DIR}/${BACKUP_NAME}/models" -type f | wc -l)
    echo -e "${GREEN}  ✓${NC} Backed up $MODEL_COUNT model files"
else
    echo -e "${YELLOW}  ⚠${NC} No models directory found"
fi
echo ""

# Backup system state and configuration
echo -e "${BLUE}Step 4: Backing up system state...${NC}"

SYSTEM_FILES=(
    "auto_execute_bets.py"
    "circuit_breaker.py"
    "contrarian_intelligence.py"
    "trap_detector.py"
    "deepseek_contrarian_analysis.py"
    "bet_validator.py"
    "nfl_system_monitor.py"
    "DEEPSEEK_CONTRARIAN_PROMPT.md"
    "NFL_MONITORING_GUIDE.md"
    "MOCK_DATA_PURGE_COMPLETE.md"
)

SYSTEM_BACKED_UP=0
for file in "${SYSTEM_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "${BACKUP_DIR}/${BACKUP_NAME}/"
        SYSTEM_BACKED_UP=$((SYSTEM_BACKED_UP + 1))
    fi
done

echo -e "${GREEN}  ✓${NC} Backed up $SYSTEM_BACKED_UP system files"
echo ""

# Backup monitoring data
echo -e "${BLUE}Step 5: Backing up monitoring data...${NC}"
if [ -d "data/monitoring" ]; then
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}/data/monitoring"
    cp -r data/monitoring/* "${BACKUP_DIR}/${BACKUP_NAME}/data/monitoring/" 2>/dev/null || true
    MONITOR_COUNT=$(find "${BACKUP_DIR}/${BACKUP_NAME}/data/monitoring" -type f | wc -l)
    echo -e "${GREEN}  ✓${NC} Backed up $MONITOR_COUNT monitoring files"
else
    echo -e "${YELLOW}  ⚠${NC} No monitoring data found"
fi
echo ""

# Create backup manifest
echo -e "${BLUE}Step 6: Creating backup manifest...${NC}"
cat > "${BACKUP_DIR}/${BACKUP_NAME}/MANIFEST.txt" <<EOF
NFL BETTING SYSTEM BACKUP
========================

Backup Date: $(date)
System Directory: $SYSTEM_DIR
Backup Directory: ${BACKUP_DIR}/${BACKUP_NAME}

Contents:
- Critical betting files: $CRITICAL_BACKED_UP files
- Data directory: $DATA_SIZE
- Model files: $MODEL_COUNT files
- System files: $SYSTEM_BACKED_UP files
- Monitoring files: $MONITOR_COUNT files

Total Backup Size: $(du -sh "${BACKUP_DIR}/${BACKUP_NAME}" | cut -f1)

To restore:
./nfl_restore.sh "${BACKUP_DIR}/${BACKUP_NAME}"
EOF

echo -e "${GREEN}  ✓${NC} Manifest created"
echo ""

# Create compressed archive
echo -e "${BLUE}Step 7: Creating compressed archive...${NC}"
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
ARCHIVE_SIZE=$(du -sh "${BACKUP_NAME}.tar.gz" | cut -f1)
echo -e "${GREEN}  ✓${NC} Archive created: ${BACKUP_NAME}.tar.gz ($ARCHIVE_SIZE)"
echo ""

# Remote backup if specified
if [ -n "$REMOTE_BACKUP" ]; then
    echo -e "${BLUE}Step 8: Copying to remote location...${NC}"

    if [[ "$REMOTE_BACKUP" == s3://* ]]; then
        # S3 backup
        if command -v aws &> /dev/null; then
            aws s3 cp "${BACKUP_NAME}.tar.gz" "$REMOTE_BACKUP/${BACKUP_NAME}.tar.gz"
            echo -e "${GREEN}  ✓${NC} Uploaded to $REMOTE_BACKUP"
        else
            echo -e "${RED}  ✗${NC} AWS CLI not found. Install it to use S3 backups."
            exit 1
        fi
    else
        # Local/mounted drive backup
        mkdir -p "$REMOTE_BACKUP"
        cp "${BACKUP_NAME}.tar.gz" "$REMOTE_BACKUP/"
        echo -e "${GREEN}  ✓${NC} Copied to $REMOTE_BACKUP"
    fi
    echo ""
fi

# Cleanup old backups (keep last 7 days)
echo -e "${BLUE}Step 9: Cleaning up old backups (keeping last 7 days)...${NC}"
find "${BACKUP_DIR}" -name "nfl_backup_*.tar.gz" -mtime +7 -delete 2>/dev/null || true
find "${BACKUP_DIR}" -maxdepth 1 -type d -name "nfl_backup_*" -mtime +7 -exec rm -rf {} \; 2>/dev/null || true
REMAINING=$(find "${BACKUP_DIR}" -name "nfl_backup_*.tar.gz" | wc -l)
echo -e "${GREEN}  ✓${NC} Cleanup complete ($REMAINING backups remaining)"
echo ""

# Summary
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}BACKUP COMPLETE ✓${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${YELLOW}Backup Location:${NC}"
echo "  ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
echo ""
echo -e "${YELLOW}Backup Size:${NC} $ARCHIVE_SIZE"
echo ""
if [ -n "$REMOTE_BACKUP" ]; then
    echo -e "${YELLOW}Remote Backup:${NC} $REMOTE_BACKUP/${BACKUP_NAME}.tar.gz"
    echo ""
fi
echo -e "${YELLOW}To restore this backup:${NC}"
echo "  ./nfl_restore.sh ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
echo ""
echo -e "${GREEN}All NFL betting system data backed up safely!${NC}"
