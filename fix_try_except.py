#!/usr/bin/env python3
"""
Script to check and fix the try/except block issue around line 65
"""


def check_and_fix_try_except(filename):
    """Check for try/except block issues around line 65."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Checking lines 55-75 for try/except issues:")
    print("=" * 50)

    # Show lines around 65
    start_line = max(0, 55)
    end_line = min(len(lines), 75)

    for i in range(start_line, end_line):
        line_num = i + 1
        line_content = lines[i].rstrip()

        # Highlight problematic lines
        if line_num == 65:
            prefix = ">>> ERROR >>> "
        elif "try:" in line_content:
            prefix = ">>> TRY   >>> "
        elif "except" in line_content:
            prefix = ">>> EXCEPT>>> "
        elif "finally" in line_content:
            prefix = ">>> FINALLY>>>"
        else:
            prefix = "             "

        print(f"{prefix}{line_num:3}: {line_content}")

    print("\n" + "=" * 50)

    # Look for orphaned try blocks
    try_blocks = []
    except_blocks = []

    for i, line in enumerate(lines):
        if "try:" in line:
            try_blocks.append(i + 1)
        if "except" in line and ":" in line:
            except_blocks.append(i + 1)

    print(f"Found try blocks on lines: {try_blocks}")
    print(f"Found except blocks on lines: {except_blocks}")

    # Try to fix the issue
    print("\nAttempting to fix try/except structure...")

    # Common fix: ensure proper indentation and structure
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # If we find a try block, ensure it has proper except/finally
        if "try:" in line and i < len(lines) - 1:
            # Add the try line
            fixed_lines.append(line)
            i += 1

            # Look ahead for the except block
            found_except = False
            while i < len(lines) and not found_except:
                next_line = lines[i]
                if "except" in next_line:
                    found_except = True
                elif next_line.strip() == "" or next_line.startswith("    "):
                    # Part of try block
                    pass
                else:
                    # Reached end of try block without except
                    if not found_except:
                        # Insert a basic except block
                        fixed_lines.append("    except Exception as e:\n")
                        fixed_lines.append("        logger.error(f'Error: {e}')\n")
                        found_except = True

                fixed_lines.append(next_line)
                i += 1
        else:
            fixed_lines.append(line)
            i += 1

    # Write the fixed version
    with open(filename + ".fixed", "w", encoding="utf-8") as f:
        f.writelines(fixed_lines)

    print(f"Created fixed version: {filename}.fixed")

    # Test the fixed version
    try:
        with open(filename + ".fixed") as f:
            compile(f.read(), filename + ".fixed", "exec")
        print("✅ Fixed version compiles successfully!")

        # Replace original with fixed version
        import shutil

        shutil.move(filename + ".fixed", filename)
        print(f"✅ Replaced original file with fixed version")

    except SyntaxError as e:
        print(f"❌ Fixed version still has error on line {e.lineno}: {e.msg}")
        print(f"Text: {e.text}")


if __name__ == "__main__":
    check_and_fix_try_except("daily_prediction_and_backtest.py")
