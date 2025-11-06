#!/usr/bin/env python3
"""Find and fix the unmatched parenthesis on line 45"""

print("🔍 DEBUGGING LINE 45 SYNTAX ERROR")

# Read the file and show context around line 45
with open("data_collection.py") as f:
    lines = f.readlines()

print(f"📁 File has {len(lines)} lines")

# Show context around line 45
if len(lines) >= 45:
    print("\n📋 Context around line 45:")
    for i in range(max(0, 40), min(len(lines), 50)):
        marker = ">>> " if i == 44 else "    "  # Line 45 is index 44
        print(f"{marker}{i+1:2d}: {lines[i].rstrip()}")

    print(f"\n🎯 Line 45 specifically:")
    line_45 = lines[44]
    print(f"   Content: {repr(line_45)}")

    # Look for function definitions around line 45
    print(f"\n🔍 Looking for function definition pattern:")
    for i in range(max(0, 35), min(len(lines), 50)):
        line = lines[i].strip()
        if line.startswith("def ") or ") -> " in line:
            print(f"   Line {i+1}: {line}")

else:
    print("❌ File has fewer than 45 lines!")

# Count parentheses to find mismatches
print(f"\n🔢 Parentheses analysis:")
open_parens = 0
for i, line in enumerate(lines[:50]):  # Check first 50 lines
    for char in line:
        if char == "(":
            open_parens += 1
        elif char == ")":
            open_parens -= 1
    if open_parens < 0:
        print(f"❌ Unmatched ')' on line {i+1}: {line.strip()}")
        break
    elif "def " in line and open_parens > 0:
        print(
            f"🔍 Function def on line {i+1} with {open_parens} open parens: {line.strip()}"
        )
