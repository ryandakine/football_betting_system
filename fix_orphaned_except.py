#!/usr/bin/env python3
"""
Targeted fix for orphaned except blocks that lost their try statements
"""


def fix_orphaned_except_blocks(filename):
    """Fix except blocks that lost their try statements when else was converted to if."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Looking for orphaned except blocks...")

    # Find patterns where "if True:" is followed later by "except"
    problem_areas = []

    for i, line in enumerate(lines):
        if "if True:" in line and "# Fixed orphaned else" in line:
            # Look ahead for an except block
            if_indent = len(line) - len(line.lstrip())

            for j in range(i + 1, min(len(lines), i + 50)):  # Look ahead up to 50 lines
                next_line = lines[j]
                next_indent = len(next_line) - len(next_line.lstrip())
                next_stripped = next_line.strip()

                if next_stripped.startswith("except ") and ":" in next_stripped:
                    # Found an except - check if it's at the same level as the if
                    if next_indent <= if_indent:
                        problem_areas.append((i + 1, j + 1, if_indent, next_indent))
                        print(
                            f"Found orphaned except: if True at line {i+1} (indent {if_indent}), except at line {j+1} (indent {next_indent})"
                        )
                        break

    if not problem_areas:
        print("No orphaned except blocks found")
        return False

    print(f"Found {len(problem_areas)} orphaned except blocks to fix")

    # Fix each problem area
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        line_num = i + 1

        # Check if this line is a problematic "if True:"
        is_problem_if = any(area[0] == line_num for area in problem_areas)

        if is_problem_if:
            # Find the corresponding except block
            except_line_num = next(
                area[1] for area in problem_areas if area[0] == line_num
            )
            except_idx = except_line_num - 1

            print(
                f"Fixing if True at line {line_num} with except at line {except_line_num}"
            )

            # Instead of "if True:", convert this section to a proper try/except
            if_indent = len(line) - len(line.lstrip())

            # Replace "if True:" with "try:"
            new_lines.append(" " * if_indent + "try:\n")
            i += 1

            # Copy all content until we reach the except block
            while i < len(lines) and i < except_idx:
                content_line = lines[i]
                new_lines.append(content_line)
                i += 1

            # Now we should be at the except line - add it
            if i < len(lines):
                except_line = lines[i]
                new_lines.append(except_line)
                i += 1

                # Continue with the except block content
                except_indent = len(except_line) - len(except_line.lstrip())

                while i < len(lines):
                    current_line = lines[i]
                    current_indent = len(current_line) - len(current_line.lstrip())
                    current_stripped = current_line.strip()

                    # Continue with except block content
                    if (
                        current_indent > except_indent
                        or (
                            current_indent == except_indent
                            and current_stripped.startswith(
                                ("except ", "else:", "finally:")
                            )
                        )
                        or not current_stripped
                    ):
                        new_lines.append(current_line)
                        i += 1
                    else:
                        # End of except block
                        break

        else:
            # Not a problem line, just copy it
            new_lines.append(line)
            i += 1

    # Write the fixed file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print("✅ Fixed orphaned except blocks")

    # Test compilation
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("🎉 SUCCESS! File now compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"❌ Still has error on line {e.lineno}: {e.msg}")

        # If we still have issues, try a more aggressive approach
        print("\nTrying more aggressive fix...")
        return aggressive_structure_fix(filename)


def aggressive_structure_fix(filename):
    """More aggressive fix - remove all 'if True:' statements and fix indentation."""

    with open(filename) as f:
        lines = f.readlines()

    print("Applying aggressive structure fix...")
    print("Removing all 'if True:' statements and fixing indentation...")

    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # If this is an "if True:" line, remove it and unindent the following block
        if "if True:" in line and "# Fixed orphaned else" in line:
            if_indent = len(line) - len(line.lstrip())
            print(f"Removing if True at line {i+1}")

            # Skip the if True line
            i += 1

            # Unindent the following block
            while i < len(lines):
                current_line = lines[i]
                current_indent = len(current_line) - len(current_line.lstrip())

                # If this line was indented under the if True, unindent it
                if current_indent > if_indent and current_line.strip():
                    # Remove 4 spaces of indentation
                    unindented = (
                        current_line[4:]
                        if current_line.startswith("    ")
                        else current_line
                    )
                    new_lines.append(unindented)
                elif current_indent <= if_indent and current_line.strip():
                    # We've reached the end of the if block
                    new_lines.append(current_line)
                    break
                else:
                    # Empty line or continuation
                    new_lines.append(current_line)

                i += 1
        else:
            new_lines.append(line)
            i += 1

    # Write the aggressively fixed file
    with open(filename, "w") as f:
        f.writelines(new_lines)

    print("✅ Applied aggressive structure fix")

    # Test compilation
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("🎉 SUCCESS! File now compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"❌ Still has error on line {e.lineno}: {e.msg}")
        print(
            "The file may need complete manual reconstruction of the problematic functions."
        )
        return False


if __name__ == "__main__":
    fix_orphaned_except_blocks("daily_prediction_and_backtest.py")
