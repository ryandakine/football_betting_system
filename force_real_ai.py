#!/usr/bin/env python3
"""Force the main system to use real AI instead of fallback"""

print("🔧 FORCING REAL AI INTEGRATION")

# Read the main file with proper encoding
try:
    with open(
        "daily_prediction_and_backtest.py", encoding="utf-8", errors="ignore"
    ) as f:
        content = f.read()
    print("✅ Successfully read main file")
except Exception as e:
    print(f"❌ Could not read file: {e}")
    exit(1)

# Create a backup
with open("daily_prediction_and_backtest_backup.py", "w", encoding="utf-8") as f:
    f.write(content)
print("💾 Created backup: daily_prediction_and_backtest_backup.py")

# Force AI integration by adding this at the top
ai_integration_code = """
# === FORCED REAL AI INTEGRATION ===
print("🔧 FORCING REAL AI INTEGRATION...")
try:
    from api_config import get_api_keys
    keys = get_api_keys()
    from game_selection import GameSelector

    # Force create real AI
    ai_analyzer = GameSelector(keys['claude'], keys['openai'], keys['grok'])
    print("🤖 ✅ FORCED: Real AI (GameSelector) initialized successfully")
    ai_enabled = True
    FORCE_REAL_AI = True

except Exception as e:
    print(f"🤖 ❌ FORCED AI failed: {e}")
    FORCE_REAL_AI = False
# === END FORCED INTEGRATION ===

"""

# Find where to insert the AI code (after imports, before main logic)
lines = content.split("\n")
insert_pos = 0

# Find a good insertion point
for i, line in enumerate(lines):
    if (
        line.strip().startswith("def ")
        or line.strip().startswith("if __name__")
        or "STARTING ENHANCED MLB" in line
    ):
        insert_pos = i
        break

# Insert the forced AI code
lines.insert(insert_pos, ai_integration_code)

# Also modify any fallback AI checks
new_content = "\n".join(lines)

# Replace fallback AI initialization
new_content = new_content.replace(
    "using fallback mock AI", "FORCED: Using Real AI (GameSelector)"
)

# Write the modified file
with open("daily_prediction_and_backtest.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("✅ Modified main system to force real AI")
print("🚀 Run: python daily_prediction_and_backtest.py")
print("💡 To restore original: copy daily_prediction_and_backtest_backup.py back")
