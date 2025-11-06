python -c "
import re
with open('daily_prediction_and_backtest.py', 'r') as f:
    content = f.read()

# Find the AI initialization logic
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'AI modules not found' in line or 'using fallback' in line:
        print(f'Found warning at line {i+1}:')
        # Show surrounding context
        start = max(0, i-10)
        end = min(len(lines), i+10)
        for j in range(start, end):
            marker = '>>> ' if j == i else '    '
            print(f'{marker}{j+1:3d}: {lines[j]}')
        break
"
