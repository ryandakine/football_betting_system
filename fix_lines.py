import re

with open("daily_prediction_and_backtest.py") as f:
    lines = f.readlines()

fixed_lines = []
for line in lines:
    if len(line.rstrip()) > 79:
        # Simple fix for long lines - break at commas or after opening parens
        if "," in line and len(line) > 79:
            # Find last comma before position 75
            pos = line.rfind(",", 0, 75)
            if pos > 0:
                fixed_lines.append(line[: pos + 1] + "\n")
                fixed_lines.append("    " + line[pos + 1 :])
                continue
    fixed_lines.append(line)

with open("daily_prediction_and_backtest.py", "w") as f:
    f.writelines(fixed_lines)

print("Fixed long lines!")
