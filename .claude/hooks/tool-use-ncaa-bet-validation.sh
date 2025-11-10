#!/bin/bash
#
# NCAA Bet Validation Hook
# ========================
# Runs BEFORE any tool execution
# Blocks invalid bets structurally - agent cannot execute wrong bets
#
# PRINCIPLE: Automated Constraints (Not Trust)
# Investment → System: Don't trust agent, enforce at system level

# Only validate if this looks like a bet placement
if [[ ! "$TOOL_INPUT" =~ (bet|stake|wager|place.*bet) ]]; then
    echo '{"permissionDecision": "allow"}'
    exit 0
fi

# Extract bet details (simplified - real version would parse JSON)
CONFIDENCE=$(echo "$TOOL_INPUT" | grep -oP 'confidence["\s:]+\K[0-9.]+' | head -1)
STAKE=$(echo "$TOOL_INPUT" | grep -oP 'stake["\s:$]+\K[0-9.]+' | head -1)
EDGE=$(echo "$TOOL_INPUT" | grep -oP 'edge["\s:]+\K[0-9.]+' | head -1)

# VALIDATION RULES (System-level constraints)
MIN_CONFIDENCE=0.70
MIN_EDGE=0.03
MAX_STAKE=500  # $500 max single bet (5% of $10k bankroll)
MIN_STAKE=20   # $20 minimum

ERRORS=""

# Check confidence threshold
if [[ -n "$CONFIDENCE" ]]; then
    CONF_CHECK=$(echo "$CONFIDENCE < $MIN_CONFIDENCE" | bc -l)
    if [[ "$CONF_CHECK" == "1" ]]; then
        ERRORS="${ERRORS}❌ BLOCKED: Confidence ${CONFIDENCE} < ${MIN_CONFIDENCE} (70% minimum)\n"
    fi
fi

# Check edge threshold
if [[ -n "$EDGE" ]]; then
    EDGE_CHECK=$(echo "$EDGE < $MIN_EDGE" | bc -l)
    if [[ "$EDGE_CHECK" == "1" ]]; then
        ERRORS="${ERRORS}❌ BLOCKED: Edge ${EDGE} < ${MIN_EDGE} (3% minimum)\n"
    fi
fi

# Check stake limits
if [[ -n "$STAKE" ]]; then
    if (( $(echo "$STAKE > $MAX_STAKE" | bc -l) )); then
        ERRORS="${ERRORS}❌ BLOCKED: Stake \$${STAKE} > \$${MAX_STAKE} maximum\n"
    fi
    if (( $(echo "$STAKE < $MIN_STAKE" | bc -l) )); then
        ERRORS="${ERRORS}❌ BLOCKED: Stake \$${STAKE} < \$${MIN_STAKE} minimum\n"
    fi
fi

# Check for dangerous commands
if echo "$TOOL_INPUT" | grep -qE "(rm -rf|drop table|delete from|--force)"; then
    ERRORS="${ERRORS}❌ BLOCKED: Dangerous command detected\n"
fi

# If any errors, DENY
if [[ -n "$ERRORS" ]]; then
    echo '{"permissionDecision": "deny", "reason": "'"$(echo -e "$ERRORS")"'"}'
    exit 0
fi

# All validations passed
echo '{"permissionDecision": "allow"}'
exit 0
