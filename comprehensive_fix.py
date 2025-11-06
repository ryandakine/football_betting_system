#!/usr/bin/env python3
"""
Comprehensive fix for all try/except/else/finally structural issues
"""


def comprehensive_try_except_fix(filename):
    """Find and fix ALL try/except structural issues in the entire file."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Analyzing entire file ({len(lines)} lines) for try/except issues...")

    # Parse the entire file and identify all try/except blocks
    try_blocks = []

    for i, line in enumerate(lines):
        if "try:" in line:
            try_blocks.append(i)

    print(f"Found {len(try_blocks)} try blocks at lines: {[i+1 for i in try_blocks]}")

    # For each try block, ensure it has proper structure
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        line_num = i + 1

        # If this is a try block, process it completely
        if "try:" in line:
            print(f"\nProcessing try block at line {line_num}")
            try_indent = len(line) - len(line.lstrip())

            # Collect all parts of this try/except/else/finally block
            try_line = line
            try_content = []
            except_blocks = []
            else_content = []
            finally_content = []

            new_lines.append(try_line)  # Add the try: line
            i += 1

            current_section = "try"
            current_except_content = []

            while i < len(lines):
                current_line = lines[i]
                current_indent = len(current_line) - len(current_line.lstrip())
                current_stripped = current_line.strip()

                # Check if we've left this try block
                if (
                    current_indent <= try_indent
                    and current_stripped
                    and not current_stripped.startswith(
                        ("except", "else:", "finally:", "#")
                    )
                    and current_section != "try"
                ):
                    # We've reached the end of this try block
                    break

                # Detect section changes
                if (
                    current_stripped.startswith("except ")
                    and ":" in current_stripped
                    and current_indent == try_indent
                ):
                    # Save previous except block if any
                    if current_section == "except" and current_except_content:
                        except_blocks.append(current_except_content)

                    # Start new except block
                    current_section = "except"
                    current_except_content = [current_line]

                elif current_stripped == "else:" and current_indent == try_indent:
                    # Save previous except block if any
                    if current_section == "except" and current_except_content:
                        except_blocks.append(current_except_content)
                        current_except_content = []

                    current_section = "else"
                    else_content.append(current_line)

                elif (
                    current_stripped.startswith("finally:")
                    and current_indent == try_indent
                ):
                    # Save previous except block if any
                    if current_section == "except" and current_except_content:
                        except_blocks.append(current_except_content)
                        current_except_content = []

                    current_section = "finally"
                    finally_content.append(current_line)

                else:
                    # Add to current section
                    if current_section == "try":
                        try_content.append(current_line)
                    elif current_section == "except":
                        current_except_content.append(current_line)
                    elif current_section == "else":
                        else_content.append(current_line)
                    elif current_section == "finally":
                        finally_content.append(current_line)

                i += 1

            # Save the last except block if any
            if current_section == "except" and current_except_content:
                except_blocks.append(current_except_content)

            # Now reconstruct the try block in correct order
            print(f"  Try content: {len(try_content)} lines")
            print(f"  Except blocks: {len(except_blocks)}")
            print(f"  Else content: {len(else_content)} lines")
            print(f"  Finally content: {len(finally_content)} lines")

            # Add try content
            new_lines.extend(try_content)

            # Add all except blocks
            for except_block in except_blocks:
                new_lines.extend(except_block)

            # If no except blocks but there's else/finally, add a generic except
            if not except_blocks and (else_content or finally_content):
                print("  Adding missing except block for else/finally")
                new_lines.append(" " * try_indent + "except Exception as e:\n")
                new_lines.append(
                    " " * (try_indent + 4) + "pass  # Generic exception handler\n"
                )

            # Add else content
            new_lines.extend(else_content)

            # Add finally content
            new_lines.extend(finally_content)

            print(f"  Reconstructed try block successfully")

        else:
            # Not a try block, just copy the line
            new_lines.append(line)
            i += 1

    # Write the fixed file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"\n✅ Processed all try blocks in the file")

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

        # Show context around the error
        print(f"\nContext around line {e.lineno}:")
        with open(filename) as f:
            error_lines = f.readlines()

        start = max(0, e.lineno - 5)
        end = min(len(error_lines), e.lineno + 5)

        for i in range(start, end):
            marker = ">>> " if i + 1 == e.lineno else "    "
            print(f"{marker}{i+1:3}: {error_lines[i].rstrip()}")

        return False


if __name__ == "__main__":
    comprehensive_try_except_fix("daily_prediction_and_backtest.py")
