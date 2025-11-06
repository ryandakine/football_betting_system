import re


def fix_long_lines(filename):
    with open(filename, "r") as f:
        content = f.read()

    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        if len(line) > 79:
            # Try to break at common points
            if "logger." in line and "(" in line:
                # Break logger calls
                idx = line.find("(")
                if idx > 0 and idx < 60:
                    indent = len(line) - len(line.lstrip())
                    fixed_lines.append(line[: idx + 1])
                    fixed_lines.append(" " * (indent + 4) + line[idx + 1 :])
                    continue

            # Break at commas
            if "," in line:
                for i in range(75, 60, -1):
                    if i < len(line) and line[i] == ",":
                        fixed_lines.append(line[: i + 1])
                        indent = len(line) - len(line.lstrip()) + 4
                        fixed_lines.append(" " * indent + line[i + 1 :].lstrip())
                        break
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    with open(filename, "w") as f:
        f.write("\n".join(fixed_lines))


fix_long_lines("daily_prediction_and_backtest.py")
print("Fixed!")
