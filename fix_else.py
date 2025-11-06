#!/usr/bin/env python3
"""
Fix the else syntax issue on line 71
"""


def fix_else_syntax_issue(filename):
    """Fix the invalid else syntax by restructuring the if/try/except logic."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Showing the problematic section (lines 55-80):")
    for i in range(54, min(80, len(lines))):
        line_num = i + 1
        marker = ">>> " if line_num == 71 else "    "
        print(f"{marker}{line_num:3}: {lines[i].rstrip()}")

    print("\n" + "=" * 50)
    print("The issue: 'else:' on line 71 cannot follow a try/except block.")
    print("It should be restructured to proper if/else logic.")
    print("=" * 50)

    # The correct structure should be:
    # if condition:
    #     try:
    #         # do something
    #     except:
    #         # handle error
    # else:  # This else belongs to the if, not the try
    #     # alternative path

    # But currently we have:
    # if condition:
    #     try:
    #         # do something
    # except:  # This except is outside the if block (wrong!)
    #     # handle error
    # else:  # This else has no matching if (wrong!)
    #     # alternative path

    # Let's fix this by restructuring the logic
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        line_num = i + 1

        # When we reach the problematic area (around line 60-75)
        if line_num == 60:  # The if statement
            # Add the if line
            new_lines.append(line)  # if os.path.exists(path):
            i += 1

            # Add the try block (properly indented inside if)
            new_lines.append("        try:\n")  # try:
            new_lines.append("            df = pl.read_parquet(path)\n")
            new_lines.append(
                '            logger.debug(f"Loaded data from {path} (rows: {df.height})")\n'
            )
            new_lines.append("            return df\n")
            new_lines.append("        except Exception as e:\n")  # except (inside if)
            new_lines.append("            logger.error(\n")
            new_lines.append(
                '                f"Error loading data from {path}: {e}. "\n'
            )
            new_lines.append('                f"Returning empty DataFrame."\n')
            new_lines.append("            )\n")
            new_lines.append(
                "            return pl.DataFrame(schema=schema if schema else {})\n"
            )

            # Skip the original problematic lines (61-70)
            while i < len(lines) and i < 70:
                i += 1

            # Add the else block (aligned with if)
            new_lines.append("    else:\n")  # else (aligned with if)
            new_lines.append("        logger.warning(\n")
            new_lines.append(
                '            f"Data file not found: {path}. Returning empty DataFrame."\n'
            )
            new_lines.append("        )\n")
            new_lines.append(
                "        return pl.DataFrame(schema=schema if schema else {})\n"
            )

            # Skip the original else block lines (71-75)
            while i < len(lines) and i < 76:
                i += 1

        else:
            # For all other lines, just copy them
            new_lines.append(line)
            i += 1

    # Write the fixed file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print("\n✅ Restructured the if/try/except/else logic")
    print("New structure:")
    print("  if condition:")
    print("      try:")
    print("          # main logic")
    print("      except:")
    print("          # error handling")
    print("  else:")
    print("      # alternative path")

    # Test compilation
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("\n✅ File now compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"\n❌ Still has syntax error on line {e.lineno}: {e.msg}")
        if hasattr(e, "text") and e.text:
            print(f"Text: {e.text.strip()}")
        return False


if __name__ == "__main__":
    fix_else_syntax_issue("daily_prediction_and_backtest.py")
