#!/usr/bin/env python3
"""Complete fix for data_collection.py syntax errors"""

import re

print("🔧 FIXING DATA_COLLECTION.PY SYNTAX ERRORS")

# Read the current file
with open("data_collection.py") as f:
    content = f.read()

print("📝 Original problematic section found")

# Fix the configuration section
old_config = """# --- Configuration ---
DATA_DIR = "data"
HISTORICAL_PLAYER_DATA_PATH = os.path.join
    DATA_DIR, "historical_player_data.parquet"
)
HISTORICAL_TEAM_DATA_PATH = os.path.join
    DATA_DIR, "historical_team_data.parquet"
)
UMPIRE_DATA_PATH == os.path.join(DATA_DIR, "umpire_data.parquet")
# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok is True)"""

new_config = """# --- Configuration ---
DATA_DIR = "data"
HISTORICAL_PLAYER_DATA_PATH = os.path.join(
    DATA_DIR, "historical_player_data.parquet"
)
HISTORICAL_TEAM_DATA_PATH = os.path.join(
    DATA_DIR, "historical_team_data.parquet"
)
UMPIRE_DATA_PATH = os.path.join(DATA_DIR, "umpire_data.parquet")
# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)"""

# Replace the problematic section
if old_config in content:
    content = content.replace(old_config, new_config)
    print("✅ Fixed exact match replacement")
else:
    # Try individual fixes if exact match fails
    print("⚠️ Exact match failed, trying individual fixes...")

    # Fix broken os.path.join calls
    content = re.sub(
        r'HISTORICAL_PLAYER_DATA_PATH = os\.path\.join\s+DATA_DIR, "historical_player_data\.parquet"\s*\)',
        'HISTORICAL_PLAYER_DATA_PATH = os.path.join(\n    DATA_DIR, "historical_player_data.parquet"\n)',
        content,
    )

    content = re.sub(
        r'HISTORICAL_TEAM_DATA_PATH = os\.path\.join\s+DATA_DIR, "historical_team_data\.parquet"\s*\)',
        'HISTORICAL_TEAM_DATA_PATH = os.path.join(\n    DATA_DIR, "historical_team_data.parquet"\n)',
        content,
    )

    # Fix double equals
    content = content.replace("UMPIRE_DATA_PATH ==", "UMPIRE_DATA_PATH =")

    # Fix exist_ok syntax
    content = content.replace("exist_ok is True", "exist_ok=True")

    print("✅ Applied individual regex fixes")

# Write the fixed file
with open("data_collection.py", "w") as f:
    f.write(content)

print("💾 Saved fixed data_collection.py")

# Test the syntax
try:
    with open("data_collection.py") as f:
        compile(f.read(), "data_collection.py", "exec")
    print("✅ Syntax check passed!")
except SyntaxError as e:
    print(f"❌ Still has syntax error: {e}")
    print(f"   Line {e.lineno}: {e.text}")
