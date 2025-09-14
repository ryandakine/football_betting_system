#!/usr/bin/env python3
"""
Fix the try/except issue around line 273
"""


def fix_all_try_except_issues(filename):
    """Find and fix all try/except structural issues in the file."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Checking file with {len(lines)} lines for try/except issues...")

    # Find all try/except issues
    issues = []

    for i, line in enumerate(lines):
        line_num = i + 1
        stripped = line.strip()

        # Look for potential issues
        if stripped.startswith("else:") and line_num > 1:
            # Check if this else belongs to a try block
            prev_lines = lines[max(0, i - 10) : i]
            has_matching_try = False
            has_except_before_else = False

            for prev_line in reversed(prev_lines):
                if "try:" in prev_line:
                    has_matching_try = True
                elif "except" in prev_line and ":" in prev_line:
                    has_except_before_else = True
                    break
                elif prev_line.strip().startswith(("if ", "for ", "while ")):
                    break

            if has_matching_try and not has_except_before_else:
                issues.append((line_num, "orphaned_else", "else without except"))

    print(f"Found {len(issues)} potential issues:")
    for line_num, issue_type, description in issues:
        print(f"  Line {line_num}: {description}")

    # Show context around line 273
    print(f"\nContext around line 273:")
    start_line = max(0, 270)
    end_line = min(len(lines), 280)

    for i in range(start_line, end_line):
        line_num = i + 1
        marker = ">>> " if line_num == 273 else "    "
        print(f"{marker}{line_num:3}: {lines[i].rstrip()}")

    # Look for the try block that should match this else
    print(f"\nLooking for try blocks before line 273:")
    for i in range(max(0, 250), 273):
        if "try:" in lines[i]:
            print(f"  Line {i+1}: {lines[i].rstrip()}")

    # Fix the specific issue
    print(f"\n{'='*50}")
    print("Attempting to fix try/except/else structure...")

    # The issue is likely that there's a try block without a proper except before the else
    # Let's find and fix this pattern

    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        line_num = i + 1

        # If we're at line 273 and it's an orphaned else
        if line_num == 273 and line.strip() == "else:":
            # Look backwards for the matching try
            try_found = False
            except_found = False
            try_indent = 0

            for j in range(i - 1, max(0, i - 20), -1):
                if "try:" in lines[j]:
                    try_found = True
                    try_indent = len(lines[j]) - len(lines[j].lstrip())
                    print(f"Found try on line {j+1} with indent {try_indent}")
                    break
                elif "except" in lines[j] and ":" in lines[j]:
                    except_found = True
                    break

            if try_found and not except_found:
                # Insert a basic except block before the else
                except_indent = try_indent
                new_lines.append(" " * except_indent + "except Exception as e:\n")
                new_lines.append(
                    " " * (except_indent + 4) + 'logger.error(f"Error: {e}")\n'
                )
                new_lines.append(" " * (except_indent + 4) + "pass\n")
                print(f"Added missing except block before else on line {line_num}")

            # Add the else line
            new_lines.append(line)
        else:
            new_lines.append(line)

        i += 1

    # Write the fixed file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"✅ Applied fixes to {filename}")

    # Test compilation
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("✅ File now compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"❌ Still has syntax error on line {e.lineno}: {e.msg}")
        if hasattr(e, "text") and e.text:
            print(f"Text: {e.text.strip()}")

        # If there are still issues, let's try a more comprehensive approach
        if "expected 'except' or 'finally' block" in e.msg or "invalid syntax" in e.msg:
            print(f"\nTrying comprehensive try/except fix...")
            return comprehensive_try_except_fix(filename)

        return False


def comprehensive_try_except_fix(filename):
    """Comprehensive fix for all try/except issues."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Performing comprehensive try/except restructuring...")

    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # If we find a try block
        if "try:" in line:
            try_indent = len(line) - len(line.lstrip())
            new_lines.append(line)  # Add the try line
            i += 1

            # Collect all lines in the try block
            try_block_lines = []
            found_except = False
            found_else = False

            while i < len(lines):
                current_line = lines[i]
                current_indent = len(current_line) - len(current_line.lstrip())
                current_stripped = current_line.strip()

                # If we find except at the same level as try
                if (
                    current_stripped.startswith("except")
                    and ":" in current_stripped
                    and current_indent == try_indent
                ):
                    found_except = True
                    break

                # If we find else at the same level as try
                elif current_stripped == "else:" and current_indent == try_indent:
                    found_else = True
                    break

                # If we find a line at same or lower indentation that's not except/else
                elif (
                    current_indent <= try_indent
                    and current_stripped
                    and not current_stripped.startswith(
                        ("except", "else:", "finally:", "#")
                    )
                ):
                    # We've left the try block
                    break

                try_block_lines.append(current_line)
                i += 1

            # Add the try block content
            new_lines.extend(try_block_lines)

            # Add except block if missing but else is present
            if found_else and not found_except:
                new_lines.append(" " * try_indent + "except Exception as e:\n")
                new_lines.append(
                    " " * (try_indent + 4)
                    + 'logger.error(f"Error in try block: {e}")\n'
                )
                new_lines.append(" " * (try_indent + 4) + "pass\n")
                print(f"Added missing except block before else at line {i+1}")

            # Add except block if completely missing
            elif not found_except and not found_else:
                new_lines.append(" " * try_indent + "except Exception as e:\n")
                new_lines.append(
                    " " * (try_indent + 4)
                    + 'logger.error(f"Error in try block: {e}")\n'
                )
                print(
                    f"Added missing except block for orphaned try at line {len(new_lines) - len(try_block_lines)}"
                )

            # Continue with the except/else block if found
            # (i is already positioned at the except/else line)
        else:
            new_lines.append(line)
            i += 1

    # Write the comprehensive fix
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print("✅ Applied comprehensive try/except fixes")

    # Test compilation
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("✅ File now compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"❌ Still has syntax error on line {e.lineno}: {e.msg}")
        return False


if __name__ == "__main__":
    fix_all_try_except_issues("daily_prediction_and_backtest.py")
