#!/usr/bin/env python3
"""
Advanced script to fix Python indentation errors by analyzing and correcting
indentation levels based on Python syntax.
"""

import ast
import re


def analyze_and_fix_indentation(filename):
    """Analyze and fix indentation issues in a Python file."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Analyzing {len(lines)} lines...")

    # Track indentation levels
    fixed_lines = []
    indent_stack = [0]  # Stack of indentation levels

    for i, line in enumerate(lines, 1):
        original_line = line
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            fixed_lines.append(line)
            continue

        # Calculate current indentation
        current_indent = len(line) - len(line.lstrip())

        # Determine expected indentation based on previous lines and syntax
        expected_indent = indent_stack[-1]

        # Check if this line should increase indentation
        if i > 1:
            prev_line = lines[i - 2].strip()
            if (
                prev_line.endswith(":")
                or prev_line.endswith("(")
                or prev_line.endswith("[")
                or prev_line.endswith("{")
                or "if " in prev_line
                or "for " in prev_line
                or "while " in prev_line
                or "def " in prev_line
                or "class " in prev_line
                or "try:" in prev_line
                or "except" in prev_line
                or "else:" in prev_line
                or "elif " in prev_line
                or "finally:" in prev_line
                or "with " in prev_line
            ):
                expected_indent = indent_stack[-1] + 4
                indent_stack.append(expected_indent)

        # Check if this line should decrease indentation
        if (
            stripped.startswith(
                ("else:", "elif ", "except", "finally:", "return", "break", "continue")
            )
            or stripped.startswith(("def ", "class "))
            or (current_indent < indent_stack[-1] and len(indent_stack) > 1)
        ):

            # Find the right indentation level
            while len(indent_stack) > 1 and indent_stack[-1] > current_indent:
                indent_stack.pop()
            expected_indent = indent_stack[-1]

            # Special handling for else, elif, except, finally
            if stripped.startswith(("else:", "elif ", "except", "finally:")):
                if len(indent_stack) > 1:
                    expected_indent = indent_stack[-2]
                else:
                    expected_indent = 0

        # Apply the expected indentation
        if current_indent != expected_indent:
            print(
                f"Line {i}: Fixing indentation from {current_indent} to {expected_indent}"
            )
            print(f"  Content: {stripped[:50]}...")
            fixed_line = " " * expected_indent + stripped + "\n"
        else:
            fixed_line = line

        fixed_lines.append(fixed_line)

    # Write the fixed file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(fixed_lines)

    print(f"✅ Fixed indentation issues in {filename}")

    # Test compilation
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("✅ File compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"⚠️ Still has syntax error on line {e.lineno}: {e.msg}")
        print(f"Text around error: {e.text}")
        return False


def simple_fix_specific_line(filename, line_num):
    """Fix a specific line's indentation by looking at context."""
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    if line_num > len(lines):
        print(f"Line {line_num} doesn't exist in file")
        return

    problem_line = lines[line_num - 1]
    print(f"Problem line {line_num}: '{problem_line.strip()}'")

    # Look at surrounding lines for context
    for i in range(max(0, line_num - 5), min(len(lines), line_num + 3)):
        prefix = ">>> " if i == line_num - 1 else "    "
        print(f"{prefix}{i+1:3}: {repr(lines[i])}")

    # Try to find the correct indentation by looking at nearby lines
    correct_indent = 0
    for i in range(line_num - 2, -1, -1):  # Look backwards
        if lines[i].strip():  # Non-empty line
            prev_indent = len(lines[i]) - len(lines[i].lstrip())
            # If previous line ends with :, increase indent
            if lines[i].strip().endswith(":"):
                correct_indent = prev_indent + 4
                break
            else:
                correct_indent = prev_indent
                break

    # Apply the fix
    stripped_line = problem_line.strip()
    fixed_line = " " * correct_indent + stripped_line + "\n"
    lines[line_num - 1] = fixed_line

    # Write back
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Fixed line {line_num} with {correct_indent} spaces indentation")


if __name__ == "__main__":
    filename = "daily_prediction_and_backtest.py"

    # First try the advanced fixer
    print("Attempting advanced indentation analysis...")
    if not analyze_and_fix_indentation(filename):
        print("\nAdvanced fixer didn't resolve all issues.")
        print("Trying targeted fix for line 257...")
        simple_fix_specific_line(filename, 257)

        # Test again
        try:
            with open(filename) as f:
                compile(f.read(), filename, "exec")
            print("✅ File now compiles successfully!")
        except SyntaxError as e:
            print(f"⚠️ Still has syntax error on line {e.lineno}: {e.msg}")
            print("You may need to manually fix the remaining issues.")
