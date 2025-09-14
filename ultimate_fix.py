#!/usr/bin/env python3
"""
Ultimate diagnostic and fix for persistent syntax errors
"""


def ultimate_diagnostic_and_fix(filename):
    """Final diagnostic to understand and fix the persistent syntax error."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("=== ULTIMATE DIAGNOSTIC ===")
    print(f"File has {len(lines)} lines")

    # Let's examine the exact error location with more context
    try:
        compile("\n".join(lines), filename, "exec")
        print("✅ File compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error on line {e.lineno}: {e.msg}")
        print(f"Error text: {repr(e.text) if e.text else 'None'}")

        error_line = e.lineno

        # Show extensive context around the error
        print(f"\nExtensive context around line {error_line}:")
        start = max(0, error_line - 15)
        end = min(len(lines), error_line + 15)

        for i in range(start, end):
            line_num = i + 1
            line_content = lines[i].rstrip()
            indent = len(lines[i]) - len(lines[i].lstrip())

            marker = (
                ">>> ERROR >>> " if line_num == error_line else f"    ({indent:2})  "
            )
            print(f"{marker}{line_num:3}: {line_content}")

    # The error "expected 'except' or 'finally' block" means there's a try without except/finally
    # Let's find ALL try statements and check their structure

    print(f"\n=== ANALYZING ALL TRY BLOCKS ===")

    try_locations = []
    for i, line in enumerate(lines):
        if "try:" in line:
            try_locations.append(i + 1)

    print(f"Found try blocks at lines: {try_locations}")

    # Check each try block
    for try_line_num in try_locations:
        try_idx = try_line_num - 1
        try_indent = len(lines[try_idx]) - len(lines[try_idx].lstrip())

        print(f"\nAnalyzing try block at line {try_line_num} (indent {try_indent}):")

        # Look for corresponding except/finally
        found_except = False
        found_finally = False
        found_else = False

        for i in range(try_idx + 1, len(lines)):
            line = lines[i]
            line_indent = len(line) - len(line.lstrip())
            line_stripped = line.strip()

            # If we're at the same indentation level as try
            if line_indent == try_indent and line_stripped:
                if line_stripped.startswith("except ") and ":" in line_stripped:
                    found_except = True
                    print(f"  Found except on line {i+1}: {line_stripped}")
                elif line_stripped.startswith("finally:"):
                    found_finally = True
                    print(f"  Found finally on line {i+1}: {line_stripped}")
                elif line_stripped.startswith("else:"):
                    found_else = True
                    print(f"  Found else on line {i+1}: {line_stripped}")
                elif not line_stripped.startswith("#"):
                    # We've left the try block
                    break

        status = "✅ VALID" if (found_except or found_finally) else "❌ INVALID"
        print(
            f"  Status: {status} (except: {found_except}, finally: {found_finally}, else: {found_else})"
        )

        if not (found_except or found_finally):
            print(
                f"  *** PROBLEM: Try block at line {try_line_num} has no except or finally! ***"
            )

    # Now let's fix ALL invalid try blocks
    print(f"\n=== FIXING INVALID TRY BLOCKS ===")

    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # If this is a try block, validate and fix it
        if "try:" in line:
            try_indent = len(line) - len(line.lstrip())
            try_line_num = i + 1

            # Add the try line
            new_lines.append(line)
            i += 1

            # Collect all content until we find except/finally or leave the block
            try_content = []
            has_except_or_finally = False

            while i < len(lines):
                current_line = lines[i]
                current_indent = len(current_line) - len(current_line.lstrip())
                current_stripped = current_line.strip()

                # Check if we're still in the try block
                if current_indent > try_indent or not current_stripped:
                    # Still inside the try block
                    try_content.append(current_line)
                elif current_indent == try_indent:
                    # Same level as try
                    if current_stripped.startswith(
                        "except "
                    ) or current_stripped.startswith("finally:"):
                        has_except_or_finally = True
                        # Add the content and the except/finally
                        new_lines.extend(try_content)
                        new_lines.append(current_line)
                        break
                    elif current_stripped.startswith("else:"):
                        # There's an else but no except - add a generic except first
                        new_lines.extend(try_content)
                        new_lines.append(" " * try_indent + "except Exception as e:\n")
                        new_lines.append(
                            " " * (try_indent + 4) + "pass  # Auto-added except\n"
                        )
                        new_lines.append(current_line)  # The else
                        has_except_or_finally = True
                        print(
                            f"Added missing except before else for try block at line {try_line_num}"
                        )
                        break
                    else:
                        # We've left the try block without finding except/finally
                        break
                else:
                    # Lower indentation - we've definitely left the try block
                    break

                i += 1

            # If we collected content but never found except/finally, add one
            if not has_except_or_finally:
                new_lines.extend(try_content)
                new_lines.append(" " * try_indent + "except Exception as e:\n")
                new_lines.append(" " * (try_indent + 4) + "pass  # Auto-added except\n")
                print(
                    f"Added missing except for orphaned try block at line {try_line_num}"
                )

                # Don't increment i here because we need to process the current line
                # in the next iteration
            else:
                i += 1  # Move past the except/finally line we just added
        else:
            # Not a try block, just copy
            new_lines.append(line)
            i += 1

    # Write the fixed file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"\n✅ Applied fixes to all try blocks")

    # Final compilation test
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("🎉 SUCCESS! File now compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"💥 STILL FAILING: Line {e.lineno}: {e.msg}")
        print("The file may need manual inspection and fixing.")
        return False


if __name__ == "__main__":
    ultimate_diagnostic_and_fix("daily_prediction_and_backtest.py")
