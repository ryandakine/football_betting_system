#!/usr/bin/env python3
"""
Fix the except-after-else issue by restructuring the try/except blocks properly
"""


def fix_except_after_else_issue(filename):
    """Fix the structural issue where except comes after else."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Fixing the except-after-else structural issue...")
    print("The problem: except ImportError comes after else, which is invalid")
    print("Solution: Restructure as nested try blocks or separate except clauses")

    # Find the problematic try block and restructure it
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        line_num = i + 1

        # Look for the try block that contains this problem
        # Based on the output, it seems to be around line 250-300
        if 240 <= line_num <= 260 and "try:" in line:
            try_indent = len(line) - len(line.lstrip())
            print(
                f"Found problematic try block at line {line_num} with indent {try_indent}"
            )

            # Start the new structure
            new_lines.append(line)  # try:
            i += 1

            # Collect all content until we hit the first except
            try_content = []
            first_except_content = []
            else_content = []
            second_except_content = []

            current_section = "try"

            while i < len(lines):
                current_line = lines[i]
                current_indent = len(current_line) - len(current_line.lstrip())
                current_stripped = current_line.strip()

                # Detect section transitions
                if (
                    current_stripped.startswith("except Exception as e:")
                    and current_indent == try_indent
                ):
                    current_section = "first_except"
                    first_except_content.append(current_line)
                elif current_stripped == "else:" and current_indent == try_indent:
                    current_section = "else"
                    else_content.append(current_line)
                elif (
                    current_stripped.startswith("except ImportError:")
                    and current_indent == try_indent
                ):
                    current_section = "second_except"
                    second_except_content.append(current_line)
                elif (
                    current_indent <= try_indent
                    and current_stripped
                    and not current_stripped.startswith(
                        ("#", "except", "else:", "finally:")
                    )
                    and current_section != "try"
                ):
                    # End of this try block
                    break
                else:
                    # Add to current section
                    if current_section == "try":
                        try_content.append(current_line)
                    elif current_section == "first_except":
                        first_except_content.append(current_line)
                    elif current_section == "else":
                        else_content.append(current_line)
                    elif current_section == "second_except":
                        second_except_content.append(current_line)

                i += 1

            # Now restructure properly
            # The correct structure should be: try -> except ImportError -> except Exception -> else

            # Add the try content (but we need to wrap it in another try for ImportError)
            new_lines.append(" " * (try_indent + 4) + "try:\n")

            # Indent the original try content one more level
            for content_line in try_content:
                if content_line.strip():  # Don't add extra indentation to empty lines
                    new_lines.append("    " + content_line)
                else:
                    new_lines.append(content_line)

            # Add the else content (also indented one more level)
            if else_content:
                # Remove the original else line and add it properly indented
                new_lines.append(" " * (try_indent + 4) + "else:\n")
                for content_line in else_content[1:]:  # Skip the original else: line
                    if content_line.strip():
                        new_lines.append("    " + content_line)
                    else:
                        new_lines.append(content_line)

            # Add the ImportError except (at the inner try level)
            if second_except_content:
                new_lines.append(" " * (try_indent + 4) + "except ImportError:\n")
                for content_line in second_except_content[
                    1:
                ]:  # Skip the original except line
                    if content_line.strip():
                        new_lines.append("    " + content_line)
                    else:
                        new_lines.append(content_line)

            # Add the general Exception except (at the outer try level)
            if first_except_content:
                new_lines.extend(first_except_content)

            print(f"Restructured try block with nested structure")

        else:
            new_lines.append(line)
            i += 1

    # Write the restructured file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print("✅ Restructured the try/except/else blocks properly")

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

        # If restructuring didn't work, try a simpler approach
        return simple_fix_except_after_else(filename)


def simple_fix_except_after_else(filename):
    """Simple fix: remove the problematic except block or move it."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Trying simple fix: moving except ImportError before else")

    # Find and move the except ImportError block
    new_lines = []
    except_import_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        line_num = i + 1

        # If this is the problematic except ImportError line
        if line_num == 283 and "except ImportError:" in line:
            # Collect this except block
            except_indent = len(line) - len(line.lstrip())
            except_import_lines.append(line)
            i += 1

            # Collect the rest of the except block
            while i < len(lines):
                current_line = lines[i]
                current_indent = len(current_line) - len(current_line.lstrip())

                if (
                    current_indent <= except_indent
                    and current_line.strip()
                    and not current_line.strip().startswith("#")
                ):
                    # End of except block
                    break

                except_import_lines.append(current_line)
                i += 1

            print(
                f"Collected except ImportError block ({len(except_import_lines)} lines)"
            )

        # If this is the else line that should come after except
        elif "else:" in line and len(except_import_lines) > 0:
            # Insert the except block before the else
            new_lines.extend(except_import_lines)
            new_lines.append(line)
            except_import_lines = []  # Clear it so we don't add it again
            i += 1
        else:
            new_lines.append(line)
            i += 1

    # Write the fixed file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print("✅ Moved except ImportError before else")

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
    fix_except_after_else_issue("daily_prediction_and_backtest.py")
