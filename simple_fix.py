#!/usr/bin/env python3
"""
Simple fix for the specific indentation issue on lines 274-281
"""


def fix_specific_indentation_issue(filename):
    """Fix the specific try/except indentation mismatch."""

    with open(filename) as f:
        lines = f.readlines()

    print("Fixing specific indentation issue on lines 274-281...")

    # Show the current state
    print("Current problematic section:")
    for i in range(270, 285):
        if i < len(lines):
            line_num = i + 1
            line_content = lines[i].rstrip()
            indent = len(lines[i]) - len(lines[i].lstrip())
            print(f"    {line_num:3} ({indent:2}): {line_content}")

    # The fix: make the except align with the try
    # Line 274 has try: with 12 spaces
    # Line 281 has except: with 8 spaces
    # We need to either:
    # 1. Move try to 8 spaces, OR
    # 2. Move except to 12 spaces

    # Looking at the context, the except should be at 8 spaces (same level as the surrounding code)
    # So we need to move the try block to 8 spaces

    new_lines = []

    for i, line in enumerate(lines):
        line_num = i + 1

        # Fix line 274 (try:) - reduce indentation from 12 to 8 spaces
        if line_num == 274 and "try:" in line:
            print(
                f"Fixing line {line_num}: reducing try indentation from 12 to 8 spaces"
            )
            new_lines.append("        try:\n")

        # Fix lines 275-280 (content of try block) - reduce indentation by 4 spaces
        elif 275 <= line_num <= 280:
            if line.strip():  # Don't modify empty lines
                # Remove 4 spaces from the beginning
                if line.startswith("    "):
                    fixed_line = line[4:]
                    new_lines.append(fixed_line)
                    if line_num == 275:
                        print(f"Fixing line {line_num}: reducing content indentation")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Line 281 (except) is already correct at 8 spaces
        else:
            new_lines.append(line)

    # Write the fixed file
    with open(filename, "w") as f:
        f.writelines(new_lines)

    print("✅ Fixed the try/except indentation alignment")

    # Show the fixed section
    with open(filename) as f:
        fixed_lines = f.readlines()

    print("\nFixed section:")
    for i in range(270, 285):
        if i < len(fixed_lines):
            line_num = i + 1
            line_content = fixed_lines[i].rstrip()
            indent = len(fixed_lines[i]) - len(fixed_lines[i].lstrip())
            marker = ">>> " if line_num in [274, 281] else "    "
            print(f"{marker}{line_num:3} ({indent:2}): {line_content}")

    # Test compilation
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("\n🎉 SUCCESS! File now compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"\n❌ Still has error on line {e.lineno}: {e.msg}")
        return False


if __name__ == "__main__":
    fix_specific_indentation_issue("daily_prediction_and_backtest.py")
