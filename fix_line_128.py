#!/usr/bin/env python3
"""
Fix the syntax issue around line 128
"""


def fix_line_128_issue(filename):
    """Fix the syntax issue around line 128."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Showing the problematic section around line 128:")
    start_line = max(0, 120)
    end_line = min(len(lines), 140)

    for i in range(start_line, end_line):
        line_num = i + 1
        marker = ">>> " if line_num == 128 else "    "
        print(f"{marker}{line_num:3}: {lines[i].rstrip()}")

    print("\n" + "=" * 50)
    print("The issue is likely that 'elif' has no matching 'if' statement.")
    print("Let's check the if/elif/else structure...")

    # Look for if/elif/else patterns around line 128
    for i in range(max(0, 115), min(len(lines), 135)):
        line_num = i + 1
        line_content = lines[i].strip()

        if line_content.startswith("if "):
            print(f"Found 'if' on line {line_num}")
        elif line_content.startswith("elif "):
            print(f"Found 'elif' on line {line_num}")
        elif line_content.startswith("else:"):
            print(f"Found 'else' on line {line_num}")

    # The issue is likely that the elif on line 128 doesn't have a proper if statement
    # Let's fix this by restructuring the if/elif/else logic

    print(f"\n{'='*50}")
    print("Attempting to fix the if/elif structure...")

    # Find the problematic elif and convert it to if if necessary
    for i in range(len(lines)):
        line = lines[i]
        line_num = i + 1

        if line_num == 128 and "elif isinstance(raw_odds_data, list):" in line:
            # Check if there's a proper if statement before this elif
            found_matching_if = False

            # Look backwards for a matching if
            for j in range(i - 1, max(0, i - 10), -1):
                prev_line = lines[j].strip()
                if prev_line.startswith("if ") and ":" in prev_line:
                    found_matching_if = True
                    print(f"Found matching if on line {j+1}")
                    break
                elif prev_line and not prev_line.startswith("#"):
                    # If we hit a non-empty, non-comment line that's not an if, break
                    if not prev_line.startswith(
                        ("if ", "elif ", "else:", "    ", "\t")
                    ):
                        break

            if not found_matching_if:
                # Convert elif to if
                indent = len(line) - len(line.lstrip())
                new_line = " " * indent + "if isinstance(raw_odds_data, list):\n"
                lines[i] = new_line
                print(f"Converted 'elif' to 'if' on line {line_num}")
            else:
                print(f"Found matching if for elif on line {line_num}")

    # Also check for any structural issues around this area
    print("\nChecking for other structural issues...")

    # Look for lines that might be missing colons or have other issues
    for i in range(max(0, 120), min(len(lines), 135)):
        line = lines[i]
        line_num = i + 1
        stripped = line.strip()

        # Check for common issues
        if stripped.startswith(
            ("if ", "elif ", "else", "for ", "while ", "def ", "class ")
        ):
            if not stripped.endswith(":"):
                print(f"Line {line_num} might be missing a colon: {stripped}")

    # Write the corrected file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n✅ Applied fixes to {filename}")

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

        # If there's still an error, show more context
        if e.lineno:
            print(f"\nMore context around line {e.lineno}:")
            start = max(0, e.lineno - 5)
            end = min(len(lines), e.lineno + 5)
            for i in range(start, end):
                marker = ">>> " if i + 1 == e.lineno else "    "
                print(f"{marker}{i+1:3}: {lines[i].rstrip()}")

        return False


if __name__ == "__main__":
    fix_line_128_issue("daily_prediction_and_backtest.py")
