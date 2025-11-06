#!/usr/bin/env python3
"""
Check and fix the syntax issue around line 283
"""


def check_line_283_issue(filename):
    """Check what's causing the syntax error on line 283."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Context around line 283:")
    start_line = max(0, 275)
    end_line = min(len(lines), 295)

    for i in range(start_line, end_line):
        line_num = i + 1
        line_content = lines[i].rstrip()
        marker = ">>> " if line_num == 283 else "    "
        print(f"{marker}{line_num:3}: {repr(line_content)}")

    print("\n" + "=" * 50)
    print("Looking for syntax issues...")

    # Check for common syntax problems
    for i in range(max(0, 275), min(len(lines), 295)):
        line = lines[i]
        line_num = i + 1
        stripped = line.strip()

        # Check for mismatched parentheses, brackets, etc.
        open_parens = line.count("(") - line.count(")")
        open_brackets = line.count("[") - line.count("]")
        open_braces = line.count("{") - line.count("}")

        if open_parens != 0 or open_brackets != 0 or open_braces != 0:
            print(
                f"Line {line_num}: Mismatched delimiters - parens:{open_parens}, brackets:{open_brackets}, braces:{open_braces}"
            )

        # Check for other common issues
        if stripped.endswith(",") and not any(
            next_line.strip().startswith(x)
            for x in [")", "]", "}"]
            for next_line in lines[i + 1 : i + 3]
        ):
            print(f"Line {line_num}: Trailing comma might be problematic")

        if stripped and not stripped.endswith((":",)) and line_num < len(lines) - 1:
            next_line = lines[i + 1].strip()
            if next_line.startswith(("except", "else:", "elif", "finally:")):
                if not stripped.endswith((":",)):
                    print(f"Line {line_num}: Might be missing colon before {next_line}")

    # Try to identify the specific syntax error pattern
    print(f"\nAnalyzing the structure around line 283...")

    # Look at the broader context
    broader_start = max(0, 260)
    broader_end = min(len(lines), 300)

    print(f"Broader context (lines {broader_start+1}-{broader_end}):")
    for i in range(broader_start, broader_end):
        line_num = i + 1
        line_content = lines[i].rstrip()
        indent = len(lines[i]) - len(lines[i].lstrip())

        if line_content.strip():
            marker = ">>> " if line_num == 283 else f"({indent:2})"
            print(f"{marker} {line_num:3}: {line_content}")

    # Try to fix common syntax issues
    print(f"\n{'='*50}")
    print("Attempting to fix syntax issues...")

    fixed_lines = lines.copy()
    changes_made = []

    # Check line 283 specifically
    line_283_idx = 282  # 0-based index
    if line_283_idx < len(fixed_lines):
        line_283 = fixed_lines[line_283_idx]

        # Common fixes for syntax errors
        if line_283.strip() == "except ImportError:":
            # This might be incorrectly placed after else
            # Let's check the structure
            pass

        # Check if there are structural issues with try/except/else
        # Look for patterns like: else: ... except:
        for i in range(
            max(0, line_283_idx - 10), min(len(fixed_lines), line_283_idx + 10)
        ):
            line = fixed_lines[i]
            if "else:" in line and i < len(fixed_lines) - 1:
                # Check if there's an except after this else (which is wrong)
                for j in range(i + 1, min(len(fixed_lines), i + 20)):
                    if "except" in fixed_lines[j] and ":" in fixed_lines[j]:
                        print(
                            f"Found problematic else/except pattern: line {i+1} has else, line {j+1} has except"
                        )
                        # This is wrong - except should come before else
                        # Let's fix it by moving the except before the else

                        # Extract the else block
                        else_line = fixed_lines[i]
                        else_indent = len(else_line) - len(else_line.lstrip())

                        # Extract the except block
                        except_line = fixed_lines[j]
                        except_indent = len(except_line) - len(except_line.lstrip())

                        # If they're at the same indentation level, swap them
                        if else_indent == except_indent:
                            # Move except before else
                            fixed_lines[i] = except_line
                            fixed_lines[j] = else_line
                            changes_made.append(
                                f"Swapped else and except on lines {i+1} and {j+1}"
                            )
                        break

    # Apply fixes if any were made
    if changes_made:
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(fixed_lines)

        print("Changes made:")
        for change in changes_made:
            print(f"  - {change}")
    else:
        print("No automatic fixes applied")

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

        # If it's still line 283, let's try a more targeted fix
        if e.lineno == 283:
            return fix_line_283_targeted(filename)

        return False


def fix_line_283_targeted(filename):
    """Targeted fix for line 283 specifically."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Applying targeted fix for line 283...")

    # The issue might be that except comes after else
    # Let's restructure the try/except/else block properly

    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        line_num = i + 1

        # If we're near the problematic area
        if 270 <= line_num <= 290:
            # Look for the try block and restructure it properly
            if "try:" in line:
                try_indent = len(line) - len(line.lstrip())
                new_lines.append(line)  # try line
                i += 1

                # Collect the try block content
                try_content = []
                except_content = []
                else_content = []

                current_section = "try"

                while i < len(lines) and len(lines) > i:
                    current_line = lines[i]
                    current_indent = len(current_line) - len(current_line.lstrip())
                    current_stripped = current_line.strip()

                    # Check what section we're in
                    if (
                        current_stripped.startswith("except")
                        and ":" in current_stripped
                        and current_indent == try_indent
                    ):
                        current_section = "except"
                        except_content.append(current_line)
                    elif current_stripped == "else:" and current_indent == try_indent:
                        current_section = "else"
                        else_content.append(current_line)
                    elif (
                        current_indent <= try_indent
                        and current_stripped
                        and not current_stripped.startswith(
                            ("#", "except", "else:", "finally:")
                        )
                        and current_section != "try"
                    ):
                        # End of try/except/else block
                        break
                    else:
                        # Add to current section
                        if current_section == "try":
                            try_content.append(current_line)
                        elif current_section == "except":
                            except_content.append(current_line)
                        elif current_section == "else":
                            else_content.append(current_line)

                    i += 1

                # Add sections in correct order: try, except, else
                new_lines.extend(try_content)
                if except_content:
                    new_lines.extend(except_content)
                if else_content:
                    new_lines.extend(else_content)

                print(f"Restructured try/except/else block starting at line {line_num}")

            else:
                new_lines.append(line)
                i += 1
        else:
            new_lines.append(line)
            i += 1

    # Write the fixed file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print("✅ Applied targeted fix for line 283")

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
    check_line_283_issue("daily_prediction_and_backtest.py")
