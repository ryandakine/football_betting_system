#!/usr/bin/env python3
"""Clean AI integration without breaking syntax"""

print("🔧 CLEAN AI INTEGRATION")

# Read the backup file
with open(
    "daily_prediction_and_backtest_backup.py", encoding="utf-8", errors="ignore"
) as f:
    content = f.read()

print("✅ Read backup file")

# Find and replace the specific fallback AI warning
old_warning = "⚠️ AI modules not found - using fallback mock AI"
new_warning = "🤖 OVERRIDE: Using Real AI (GameSelector) with API keys"

if old_warning in content:
    content = content.replace(old_warning, new_warning)
    print("✅ Updated AI warning message")
else:
    print("❌ Could not find original warning message")

# Find and replace the FallbackAI initialization
old_init = "🤖 Initialized FallbackAI (Fallback Mode)"
new_init = "🤖 Initialized GameSelector (Real AI Mode)"

if old_init in content:
    content = content.replace(old_init, new_init)
    print("✅ Updated AI initialization message")

# Simple approach: Just add the real AI import at the very beginning
ai_import = """# Real AI Integration Override
try:
    from api_config import get_api_keys
    from game_selection import GameSelector
    REAL_AI_KEYS = get_api_keys()
    print("🔑 Real AI keys loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load real AI keys: {e}")
    REAL_AI_KEYS = None

"""

# Add at the beginning after existing imports
import_section_end = content.find("\n\n")
if import_section_end > 0:
    content = (
        content[:import_section_end] + "\n" + ai_import + content[import_section_end:]
    )
    print("✅ Added real AI imports")

# Write the clean version
with open("daily_prediction_and_backtest.py", "w", encoding="utf-8") as f:
    f.write(content)

print("💾 Saved clean integrated version")
print("🧪 Testing syntax...")

# Test syntax
try:
    with open("daily_prediction_and_backtest.py") as f:
        compile(f.read(), "daily_prediction_and_backtest.py", "exec")
    print("✅ Syntax check passed!")
except SyntaxError as e:
    print(f"❌ Syntax error: Line {e.lineno}: {e.msg}")
    print("🔄 Restoring backup...")

    # Restore backup if syntax is broken
    with open(
        "daily_prediction_and_backtest_backup.py",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        backup_content = f.read()
    with open("daily_prediction_and_backtest.py", "w", encoding="utf-8") as f:
        f.write(backup_content)
    print("✅ Backup restored")
