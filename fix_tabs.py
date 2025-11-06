#!/usr/bin/env python3
"""
Script to fix indentation errors in Python files by converting tabs to spaces
and ensuring consistent indentation.
"""


def fix_indentation(filename):
    """Fix indentation issues in a Python file."""
    try:
        # Read the file
        with open(filename, encoding="utf-8") as f:
            lines = f.readlines()

        print(f"Original file has {len(lines)} lines")

        # Process each line
        fixed_lines = []
        for i, line in enumerate(lines, 1):
            # Convert tabs to 4 spaces
            fixed_line = line.expandtabs(4)
            fixed_lines.append(fixed_line)

        # Write back the fixed content
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(fixed_lines)

        print(f"✅ Successfully fixed indentation in {filename}")
        print("All tabs have been converted to 4 spaces")

        # Try to compile to check for remaining errors
        try:
            with open(filename) as f:
                compile(f.read(), filename, "exec")
            print("✅ File compiles successfully - no syntax errors!")
        except SyntaxError as e:
            print(f"⚠️ Still has syntax error on line {e.lineno}: {e.msg}")

    except Exception as e:
        print(f"❌ Error fixing file: {e}")


if __name__ == "__main__":
    fix_indentation("daily_prediction_and_backtest.py")
