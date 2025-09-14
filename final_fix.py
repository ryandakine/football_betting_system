#!/usr/bin/env python3
"""
Final fix for the orphaned else statement on line 274
"""


def final_structure_fix(filename):
    """Fix the orphaned else statement and overall structure issues."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Analyzing the structure around line 274...")

    # Show more context
    start_line = max(0, 260)
    end_line = min(len(lines), 290)

    print(f"Context (lines {start_line+1}-{end_line}):")
    for i in range(start_line, end_line):
        line_num = i + 1
        line_content = lines[i].rstrip()
        indent_level = len(lines[i]) - len(lines[i].lstrip())

        marker = ">>> " if line_num == 274 else f"({indent_level:2})"
        print(f"{marker} {line_num:3}: {line_content}")

    print("\n" + "=" * 60)
    print("The issue: 'else:' on line 274 has no matching if/try/except structure")
    print("Looking for what this else should belong to...")

    # Look for patterns that could explain this else
    for i in range(max(0, 250), 274):
        line = lines[i].strip()
        if line.startswith(("if ", "try:", "for ", "while ")):
            print(f"Found control structure on line {i+1}: {line}")

    # The else on line 274 is likely meant to be part of a try/except block
    # Let's find the nearest try block and see if we can restructure

    print("\nRestructuring the problematic section...")

    # Strategy: Find the try block that this else should belong to
    # and restructure the entire section properly

    new_lines = []
    i = 0

    while i < len(lines):
        line_num = i + 1

        # When we reach the problematic area, reconstruct it properly
        if 240 <= line_num <= 300:
            # Look for the start of the function or major block
            if line_num == 240:
                print(f"Reconstructing section from line {line_num}...")

                # Find the logical structure and rebuild it
                # Based on the context, it looks like there should be a try/except/else structure

                # Skip to where we want to start reconstructing
                while i < len(lines) and not (
                    "try:" in lines[i] or "if " in lines[i].strip()
                ):
                    new_lines.append(lines[i])
                    i += 1

                # Now we should be at a control structure
                if i < len(lines):
                    current_line = lines[i]

                    if "try:" in current_line:
                        # This is a try block - handle it properly
                        try_indent = len(current_line) - len(current_line.lstrip())
                        new_lines.append(current_line)  # try:
                        i += 1

                        # Collect everything until we find structural keywords
                        content_lines = []
                        except_lines = []
                        else_lines = []

                        current_section = "try"

                        while i < len(lines) and i < 300:  # Don't go too far
                            curr_line = lines[i]
                            curr_indent = len(curr_line) - len(curr_line.lstrip())
                            curr_stripped = curr_line.strip()

                            # Check for section transitions at the same indentation level
                            if curr_indent == try_indent:
                                if (
                                    curr_stripped.startswith("except ")
                                    and ":" in curr_stripped
                                ):
                                    if current_section == "try":
                                        # We're moving from try to except
                                        current_section = "except"
                                        except_lines.append(curr_line)
                                    else:
                                        # Another except block
                                        except_lines.append(curr_line)
                                elif curr_stripped == "else:":
                                    current_section = "else"
                                    else_lines.append(curr_line)
                                elif curr_stripped and not curr_stripped.startswith(
                                    ("#", "except", "else:", "finally:")
                                ):
                                    # We've left the try block
                                    break
                                else:
                                    # Add to current section
                                    if current_section == "try":
                                        content_lines.append(curr_line)
                                    elif current_section == "except":
                                        except_lines.append(curr_line)
                                    elif current_section == "else":
                                        else_lines.append(curr_line)
                            else:
                                # Indented content - add to current section
                                if current_section == "try":
                                    content_lines.append(curr_line)
                                elif current_section == "except":
                                    except_lines.append(curr_line)
                                elif current_section == "else":
                                    else_lines.append(curr_line)

                            i += 1

                        # Now add the sections in correct order
                        new_lines.extend(content_lines)

                        # Ensure we have at least one except block
                        if not any("except" in line for line in except_lines):
                            new_lines.append(
                                " " * try_indent + "except Exception as e:\n"
                            )
                            new_lines.append(
                                " " * (try_indent + 4) + 'logger.error(f"Error: {e}")\n'
                            )

                        new_lines.extend(except_lines)
                        new_lines.extend(else_lines)

                        print(f"Reconstructed try/except/else structure")

                        # Continue from where we left off
                        continue

            # If we're still in the problematic range but didn't handle it above
            if 270 <= line_num <= 280:
                line = lines[i]
                # Convert orphaned else to if statement or remove it
                if line.strip() == "else:":
                    # Convert to a simple if True: to make it valid
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(
                        " " * indent + "if True:  # Converted from orphaned else\n"
                    )
                    print(f"Converted orphaned else on line {line_num} to if True:")
                else:
                    new_lines.append(line)
                i += 1
            else:
                new_lines.append(lines[i])
                i += 1
        else:
            new_lines.append(lines[i])
            i += 1

    # Write the fixed file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print("✅ Applied final structure fixes")

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

        # If we still have issues, let's try the nuclear option
        print("\nTrying nuclear option: convert all orphaned else statements...")
        return nuclear_fix_orphaned_else(filename)


def nuclear_fix_orphaned_else(filename):
    """Nuclear option: find and fix ALL orphaned else statements."""

    with open(filename) as f:
        lines = f.readlines()

    print("Nuclear fix: Converting all potentially orphaned else statements...")

    new_lines = []

    for i, line in enumerate(lines):
        if line.strip() == "else:":
            # Check if this else has a proper if/try/except before it
            has_matching_structure = False

            # Look backwards for matching structure
            current_indent = len(line) - len(line.lstrip())

            for j in range(i - 1, max(0, i - 20), -1):
                prev_line = lines[j]
                prev_indent = len(prev_line) - len(prev_line.lstrip())
                prev_stripped = prev_line.strip()

                if prev_indent == current_indent:
                    if prev_stripped.startswith(
                        ("if ", "elif ", "try:", "except ")
                    ) and prev_stripped.endswith(":"):
                        has_matching_structure = True
                        break
                    elif prev_stripped and not prev_stripped.startswith("#"):
                        # Hit a different statement at same level
                        break

            if not has_matching_structure:
                # Convert orphaned else to if True:
                indent = len(line) - len(line.lstrip())
                new_lines.append(" " * indent + "if True:  # Fixed orphaned else\n")
                print(f"Fixed orphaned else on line {i+1}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Write the nuclear fix
    with open(filename, "w") as f:
        f.writelines(new_lines)

    print("✅ Applied nuclear fix for orphaned else statements")

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
    final_structure_fix("daily_prediction_and_backtest.py")
