# 🎯 WARP AI EFFICIENCY RULE - Football Betting System Project

**CRITICAL RULE FOR ALL AI AGENTS ON THIS PROJECT**

---

## The Problem
AI agents waste tokens by:
1. Making assumptions instead of checking actual files
2. Saying things don't exist without looking properly
3. Making users repeat themselves multiple times
4. Taking roundabout paths when direct paths exist

---

## The Solution: EFFICIENCY FIRST

### Rule #1: CHECK BEFORE ASSUMING
- **User says**: "The key is in the .env file"
- **What I do**: `cat .env | grep -i odds` IMMEDIATELY (not later)
- **What I DON'T do**: Assume it's not there, tell user to set it, waste tokens

### Rule #2: BE DIRECT
- No preamble, no excuses, no explanation
- If you need info: ASK DIRECTLY
- If you're wrong: ADMIT IT, MOVE ON
- If you're efficient: SHUT UP AND DO IT

### Rule #3: FILE HIERARCHY FOR SEARCHING
When looking for something (API keys, configs, data), check in this order:
1. `.env` file (always first)
2. `data/` directory (second)
3. Project root (third)
4. Anywhere else (last resort)

**Don't just say "it doesn't exist" - CHECK THESE FIRST**

### Rule #4: NO ROUNDABOUTS
- Don't ask "do you have X?" if you can check
- Don't suggest three options when one works
- Don't explain what you're about to do, just do it
- Don't waste tokens on preamble

### Rule #5: TOKEN MATH
- Every token costs money
- "I don't know" + asking user = 2 token costs
- Checking file = 1 token cost
- Always check first

---

## Example: RIGHT WAY vs WRONG WAY

### ❌ WRONG WAY (Wastes Tokens)
```
Agent: "I need your API key. Set it as ODDS_API_KEY in environment"
User: "It's in the .env file"
Agent: "Let me look for it..."
[searches wrong places]
Agent: "I can't find it. Can you provide it?"
User: "It's literally in .env"
[wastes 3x tokens]
```

### ✅ RIGHT WAY (Efficient)
```
Agent: [immediately checks] cat .env | grep -i odds
[finds key, uses it]
[done]
[saves 3x tokens]
```

---

## When This Applies
- **ALWAYS** on this project
- Every API call, every file lookup, every assumption
- No exceptions
- Non-negotiable

---

## The Cost
- Wasting tokens = wasting user's money
- Multiple mistakes = exponential waste
- Every assumption that's wrong = tokens down the drain
- EFFICIENCY IS NOT OPTIONAL

---

**This rule applies to ALL AI agents working on this project going forward.**

**Treat token efficiency like you'd treat money in a real business.**

---

Generated: 2025-11-15
Status: ACTIVE - APPLY IMMEDIATELY
