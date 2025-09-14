#!/usr/bin/env python3
"""
Fix the try/except issue around line 102
"""


def fix_line_102_issue(filename):
    """Fix the try/except alignment issue around line 102."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Showing the problematic section around line 102:")
    start_line = max(0, 95)
    end_line = min(len(lines), 115)

    for i in range(start_line, end_line):
        line_num = i + 1
        marker = ">>> " if line_num == 102 else "    "
        print(f"{marker}{line_num:3}: {lines[i].rstrip()}")

    print("\n" + "=" * 50)
    print("Looking for the try statement that matches the except on line 102...")

    # Find try statements before line 102
    try_lines = []
    for i in range(max(0, 80), 102):
        if "try:" in lines[i]:
            try_lines.append(i + 1)

    print(f"Found try statements on lines: {try_lines}")

    # Let's examine the structure more carefully
    print("\nDetailed analysis of lines 90-110:")
    for i in range(89, min(110, len(lines))):
        line_num = i + 1
        line_content = lines[i].rstrip()
        indent_level = len(lines[i]) - len(lines[i].lstrip())

        if "try:" in line_content:
            marker = f"TRY({indent_level:2})"
        elif "except" in line_content:
            marker = f"EXC({indent_level:2})"
        elif line_num == 102:
            marker = ">>> ERROR"
        else:
            marker = f"   ({indent_level:2})"

        print(f"{marker} {line_num:3}: {line_content}")

    # Try to fix the issue by ensuring proper try/except alignment
    print(f"\n{'='*50}")
    print("Attempting to fix the try/except structure...")

    # Look for the pattern and fix indentation
    for i in range(len(lines)):
        line = lines[i]
        line_num = i + 1

        # If this is line 102 and it's an except statement
        if line_num == 102 and "except" in line:
            # Find the corresponding try statement by looking backwards
            try_indent = None
            for j in range(i - 1, max(0, i - 20), -1):  # Look back up to 20 lines
                if "try:" in lines[j]:
                    try_indent = len(lines[j]) - len(lines[j].lstrip())
                    print(f"Found matching try on line {j+1} with {try_indent} spaces")
                    break

            if try_indent is not None:
                # Align the except with the try
                except_content = line.strip()
                lines[i] = " " * try_indent + except_content + "\n"
                print(
                    f"Fixed line {line_num}: aligned except with try ({try_indent} spaces)"
                )
            else:
                print(f"Could not find matching try for except on line {line_num}")

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

        # If there's still an error, let's try a more aggressive fix
        if e.lineno and "expected 'except' or 'finally' block" in e.msg:
            print(f"\nTrying to fix orphaned try block around line {e.lineno}...")
            return fix_orphaned_try_block(filename, e.lineno)

        return False


def fix_orphaned_try_block(filename, error_line):
    """Fix orphaned try blocks by adding proper except statements."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    # Look for try blocks without proper except blocks
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # If we find a try block
        if "try:" in line:
            try_indent = len(line) - len(line.lstrip())
            new_lines.append(line)
            i += 1

            # Look ahead to see if there's a matching except
            found_except = False
            try_block_lines = []

            # Collect lines that are part of the try block
            while i < len(lines):
                next_line = lines[i]
                next_indent = len(next_line) - len(next_line.lstrip())

                # If we find an except at the same level as try, we're good
                if "except" in next_line and next_indent == try_indent:
                    found_except = True
                    break

                # If we find a line at the same or lower indentation level that's not except
                elif next_indent <= try_indent and next_line.strip() != "":
                    if "except" not in next_line and "finally" not in next_line:
                        # We've left the try block without finding except
                        break

                try_block_lines.append(next_line)
                i += 1

            # Add all the try block lines
            new_lines.extend(try_block_lines)

            # If no except was found, add one
            if not found_except:
                new_lines.append(" " * try_indent + "except Exception as e:\n")
                new_lines.append(
                    " " * (try_indent + 4) + 'logger.error(f"Error: {e}")\n'
                )
                print(
                    f"Added missing except block for try at line {len(new_lines) - len(try_block_lines) - 1}"
                )
        else:
            new_lines.append(line)
            i += 1

    # Write the fixed file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # Test again
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("✅ File now compiles successfully after adding missing except blocks!")
        return True
    except SyntaxError as e:
        print(f"❌ Still has syntax error on line {e.lineno}: {e.msg}")
        return False


if __name__ == "__main__":
    fix_line_102_issue("daily_prediction_and_backtest.py")
