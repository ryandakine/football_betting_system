#!/usr/bin/env python3
"""
Fix the nested try blocks issue
"""


def fix_nested_try_issue(filename):
    """Fix the broken nested try structure."""

    with open(filename) as f:
        lines = f.readlines()

    print("Fixing the nested try blocks issue...")
    print("Problem: Line 244 has try, Line 245 has try, Line 274 has try")
    print("Solution: Remove the extra try on line 274 and fix the structure")

    new_lines = []

    for i, line in enumerate(lines):
        line_num = i + 1

        # Remove the problematic third try on line 274
        if line_num == 274 and "try:" in line:
            print(f"Removing extra try statement on line {line_num}")
            # Instead of the try, just add the content directly
            # The indentation should match the surrounding code
            continue  # Skip this line entirely

        # Fix the indentation of the content that was under the removed try
        elif 275 <= line_num <= 280:
            # These lines were indented for the removed try block
            # Reduce their indentation by 4 spaces to match the structure
            if line.strip():  # Don't modify empty lines
                if line.startswith("        "):  # If it starts with 8+ spaces
                    # Remove 4 spaces to unindent from the removed try
                    fixed_line = line[4:] if line.startswith("    ") else line
                    new_lines.append(fixed_line)
                    if line_num == 275:
                        print(f"Fixed indentation for lines {line_num}-280")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        else:
            new_lines.append(line)

    # Write the fixed file
    with open(filename, "w") as f:
        f.writelines(new_lines)

    print("✅ Fixed the nested try structure")

    # Show the fixed section
    print("\nFixed section around former line 274:")
    with open(filename) as f:
        fixed_lines = f.readlines()

    for i in range(270, min(285, len(fixed_lines))):
        line_num = i + 1
        line_content = fixed_lines[i].rstrip()
        indent = len(fixed_lines[i]) - len(fixed_lines[i].lstrip())
        print(f"    {line_num:3} ({indent:2}): {line_content}")

    # Test compilation
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("\n🎉 SUCCESS! File now compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"\n❌ Still has error on line {e.lineno}: {e.msg}")

        # If there's still an error, it might be a different issue
        if e.lineno != 274:
            print("The error moved to a different line - progress!")
            print("You may need to run the script again for the next issue.")

        return False


if __name__ == "__main__":
    fix_nested_try_issue("daily_prediction_and_backtest.py")
