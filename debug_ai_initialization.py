#!/usr/bin/env python3
"""Debug why the main system isn't using real AI"""

print("🔍 DEBUGGING AI INITIALIZATION IN MAIN SYSTEM")
print("=" * 60)

# Test 1: Basic imports
print("\n1. 🧪 TESTING BASIC IMPORTS:")
try:
    from game_selection import GameSelector

    print("   ✅ GameSelector import OK")
except Exception as e:
    print(f"   ❌ GameSelector import failed: {e}")

try:
    from api_config import get_api_keys

    keys = get_api_keys()
    print("   ✅ API keys import OK")
    print(f"   📋 Keys available: {list(keys.keys())}")
except Exception as e:
    print(f"   ❌ API keys failed: {e}")

# Test 2: GameSelector creation (what main system does)
print("\n2. 🧪 TESTING GAME SELECTOR CREATION:")
try:
    from api_config import get_api_keys

    keys = get_api_keys()
    from game_selection import GameSelector

    ai = GameSelector(keys["claude"], keys["openai"], keys["grok"])
    print("   ✅ GameSelector created successfully")
    print(f"   📋 Type: {type(ai)}")

    # Test if it has expected methods
    if hasattr(ai, "analyze_games"):
        print("   ✅ Has analyze_games method")
    if hasattr(ai, "get_insights"):
        print("   ✅ Has get_insights method")

except Exception as e:
    print(f"   ❌ GameSelector creation failed: {e}")
    import traceback

    traceback.print_exc()

# Test 3: Find AI initialization in main file
print("\n3. 🔍 FINDING AI INITIALIZATION LOGIC:")
try:
    with open("daily_prediction_and_backtest.py") as f:
        content = f.read()

    lines = content.split("\n")

    # Find the AI initialization section
    ai_init_start = None
    for i, line in enumerate(lines):
        if "AI modules not found" in line or "using fallback" in line:
            ai_init_start = i
            break

    if ai_init_start:
        print(f"   Found AI fallback message at line {ai_init_start + 1}")

        # Find the try block that leads to this
        for i in range(max(0, ai_init_start - 20), ai_init_start):
            line = lines[i].strip()
            if line.startswith("try:"):
                print(f"   Try block starts at line {i + 1}")
                # Show the try/except block
                for j in range(i, min(len(lines), ai_init_start + 10)):
                    indent = "   >>> " if j == ai_init_start else "       "
                    print(f"{indent}{j+1:3d}: {lines[j]}")
                break
        else:
            print("   Could not find try block")
    else:
        print("   Could not find AI fallback message")

except Exception as e:
    print(f"   ❌ Could not analyze main file: {e}")

# Test 4: Check what the main system expects
print("\n4. 🔍 CHECKING EXPECTED AI INTERFACE:")
try:
    # Look for how AI is used in main system
    with open("daily_prediction_and_backtest.py") as f:
        content = f.read()

    # Find AI method calls
    import re

    ai_calls = re.findall(r"ai_analyzer\.(\w+)\(", content)
    if ai_calls:
        print("   Expected AI methods:")
        for method in set(ai_calls):
            print(f"     - {method}()")
    else:
        print("   No AI method calls found")

except Exception as e:
    print(f"   ❌ Could not check AI interface: {e}")

print("\n🎯 DIAGNOSIS COMPLETE!")
print("Run this to see exactly why the real AI isn't loading.")
