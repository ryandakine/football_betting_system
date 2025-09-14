#!/usr/bin/env python3
"""
Final manual fix for the remaining structural issues
"""


def final_manual_fix(filename):
    """Manually fix the remaining structural issues."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Applying final manual fixes...")

    # The issue is that we have "if True:" statements that were converted from else
    # but they don't have proper blocks or the following code isn't properly indented

    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        line_num = i + 1

        # Check for "if True:" statements that might need fixing
        if "if True:" in line and "# Fixed orphaned else" in line:
            print(f"Found converted else statement at line {line_num}")

            # Add the if True line
            new_lines.append(line)
            i += 1

            # Check what follows - if it's not properly indented, fix it
            if i < len(lines):
                next_line = lines[i]
                if_indent = len(line) - len(line.lstrip())
                next_indent = len(next_line) - len(next_line.lstrip())

                # If the next line isn't indented relative to the if, we need to fix the structure
                if next_line.strip() and next_indent <= if_indent:
                    print(f"  Next line at {i+1} is not properly indented for if block")

                    # If the next line is a try block, we need to indent it
                    if "try:" in next_line:
                        print(f"  Indenting try block that follows if True")

                        # Find all lines that belong to this try block and indent them
                        try_start = i
                        try_indent = next_indent

                        # Indent the try line and everything that follows until we reach
                        # something at the same or lower indentation level
                        while i < len(lines):
                            current_line = lines[i]
                            current_indent = len(current_line) - len(
                                current_line.lstrip()
                            )

                            # If this line is at or below the original try indent and is not empty
                            if (
                                current_indent <= try_indent
                                and current_line.strip()
                                and not current_line.strip().startswith("#")
                            ):
                                # Check if it's part of the try structure (except, else, finally)
                                stripped = current_line.strip()
                                if not (
                                    stripped.startswith(
                                        ("except ", "else:", "finally:")
                                    )
                                    and current_indent == try_indent
                                ):
                                    # This is the end of the try block
                                    break

                            # Indent this line to be inside the if block
                            if current_line.strip():  # Don't indent empty lines
                                new_lines.append("    " + current_line)
                            else:
                                new_lines.append(current_line)

                            i += 1
                    else:
                        # Not a try block, just indent the next few lines appropriately
                        # Collect lines that should be indented under the if
                        lines_to_indent = []
                        base_indent = next_indent

                        while i < len(lines):
                            current_line = lines[i]
                            current_indent = len(current_line) - len(
                                current_line.lstrip()
                            )

                            # If we hit something at lower indentation, stop
                            if (
                                current_indent < base_indent
                                and current_line.strip()
                                and not current_line.strip().startswith("#")
                            ):
                                break

                            lines_to_indent.append(current_line)
                            i += 1

                            # Stop after a reasonable number of lines
                            if len(lines_to_indent) > 20:
                                break

                        # Add the indented lines
                        for line_to_indent in lines_to_indent:
                            if line_to_indent.strip():
                                new_lines.append("    " + line_to_indent)
                            else:
                                new_lines.append(line_to_indent)
                else:
                    # Next line is already properly indented
                    new_lines.append(next_line)
                    i += 1

        # Handle another specific case: orphaned try blocks after if True
        elif "try:" in line and i > 0:
            prev_line = lines[i - 1]
            if "if True:" in prev_line:
                # This try should be indented under the if
                current_indent = len(line) - len(line.lstrip())
                if_indent = len(prev_line) - len(prev_line.lstrip())

                if current_indent <= if_indent:
                    print(f"Indenting orphaned try block at line {line_num}")
                    # Re-indent this try block
                    new_lines.append("    " + line)
                    i += 1

                    # Also indent the content of the try block
                    try_content_indent = current_indent
                    while i < len(lines):
                        current_line = lines[i]
                        current_line_indent = len(current_line) - len(
                            current_line.lstrip()
                        )

                        # If we're still in the try block or its except/else
                        if current_line_indent > try_content_indent or (
                            current_line_indent == try_content_indent
                            and current_line.strip().startswith(
                                ("except ", "else:", "finally:")
                            )
                        ):
                            if current_line.strip():
                                new_lines.append("    " + current_line)
                            else:
                                new_lines.append(current_line)
                            i += 1
                        else:
                            # End of try block
                            break
                else:
                    new_lines.append(line)
                    i += 1
            else:
                new_lines.append(line)
                i += 1
        else:
            new_lines.append(line)
            i += 1

    # Write the fixed file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print("✅ Applied final manual fixes")

    # Test compilation
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("🎉 SUCCESS! File now compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"❌ Still has error on line {e.lineno}: {e.msg}")

        # Show the specific problem area
        with open(filename) as f:
            error_lines = f.readlines()

        print(f"\nProblem area around line {e.lineno}:")
        start = max(0, e.lineno - 8)
        end = min(len(error_lines), e.lineno + 8)

        for i in range(start, end):
            marker = ">>> " if i + 1 == e.lineno else "    "
            print(f"{marker}{i+1:3}: {error_lines[i].rstrip()}")

        return False


if __name__ == "__main__":
    final_manual_fix("daily_prediction_and_backtest.py")
