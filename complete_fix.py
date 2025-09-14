#!/usr/bin/env python3
"""
Complete the structure fix by properly handling the nested try/except/else
"""


def complete_structure_fix(filename):
    """Complete the fix by adding proper else clause for the nested try."""

    with open(filename) as f:
        lines = f.readlines()

    print("Completing the structure fix...")
    print("Issue: except ImportError has no matching try")
    print("Solution: Convert the code section to an else clause of the inner try")

    new_lines = []

    for i, line in enumerate(lines):
        line_num = i + 1

        # Add an 'else:' clause before the code that was under the removed try
        if (
            line_num == 274
            and "team_predictions = team_predictor.predict_games" in line
        ):
            # This code should be in an else clause of the inner try block
            # The inner try is at line 245, so the else should be at the same level
            # Looking at the indentation, the inner try was at 12 spaces
            # So the else should also be at 12 spaces

            print(f"Adding else clause before line {line_num}")
            new_lines.append("            else:\n")  # 12 spaces to match inner try

            # Now add the original line with proper indentation (16 spaces under else)
            new_lines.append(
                "                team_predictions = team_predictor.predict_games(team_features)\n"
            )

        # Fix indentation for the rest of the else block content
        elif 275 <= line_num <= 279:
            if line.strip():
                # These should be indented under the else (16 spaces)
                content = line.strip()
                if "team_predictions_with_odds = team_predictions.join(" in content:
                    new_lines.append(
                        "                team_predictions_with_odds = team_predictions.join(\n"
                    )
                elif content == "team_features,":
                    new_lines.append("                    team_features,\n")
                elif content.startswith("on="):
                    new_lines.append(
                        '                    on=["game_id", "home_team", "away_team"],\n'
                    )
                elif content.startswith("how="):
                    new_lines.append('                    how="left",\n')
                elif content == ")":
                    new_lines.append("                )\n")
                else:
                    # Generic case - add 16 spaces
                    new_lines.append("                " + content + "\n")
            else:
                new_lines.append(line)

        # Skip the original lines we're replacing
        elif line_num in [274, 275, 276, 277, 278, 279]:
            continue

        else:
            new_lines.append(line)

    # Write the fixed file
    with open(filename, "w") as f:
        f.writelines(new_lines)

    print("✅ Completed the structure fix with proper else clause")

    # Show the fixed section
    print("\nFixed section:")
    with open(filename) as f:
        fixed_lines = f.readlines()

    for i in range(270, min(290, len(fixed_lines))):
        line_num = i + 1
        line_content = fixed_lines[i].rstrip()
        indent = len(fixed_lines[i]) - len(fixed_lines[i].lstrip())
        marker = ">>> " if line_num in [274, 280] else "    "
        print(f"{marker}{line_num:3} ({indent:2}): {line_content}")

    # Test compilation
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("\n🎉 SUCCESS! File now compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"\n❌ Still has error on line {e.lineno}: {e.msg}")

        # Show the new problem area
        print(f"\nProblem area around line {e.lineno}:")
        start = max(0, e.lineno - 5)
        end = min(len(fixed_lines), e.lineno + 5)

        for i in range(start, end):
            marker = ">>> " if i + 1 == e.lineno else "    "
            print(f"{marker}{i+1:3}: {fixed_lines[i].rstrip()}")

        return False


if __name__ == "__main__":
    complete_structure_fix("daily_prediction_and_backtest.py")
