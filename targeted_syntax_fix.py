#!/usr/bin/env python3
"""
Targeted Syntax Error Fix
Fixes specific patterns that are causing E999 errors.
"""

import re
from pathlib import Path


def fix_unmatched_parentheses_v2(content: str) -> str:
    """Fix unmatched parentheses more carefully."""
    lines = content.split("\n")
    fixed_lines = []

    for line_num, line in enumerate(lines):
        original_line = line

        # Fix specific patterns that were broken by the previous script

        # Pattern: something() followed by parameters -> something(parameters)
        if re.search(r"\w+\(\)\s+\w+", line):
            line = re.sub(r"(\w+)\(\)\s+([^)]+)", r"\1(\2)", line)

        # Pattern: unmatched ) at start of line (continuation from previous)
        if line.strip().startswith(")") and line_num > 0:
            # Check if previous line needs this closing paren
            prev_line = fixed_lines[-1] if fixed_lines else ""
            if "(" in prev_line and prev_line.count("(") > prev_line.count(")"):
                # The ) belongs on the previous line
                fixed_lines[-1] = prev_line + line.strip()
                continue

        # Pattern: function() extra_params: -> function(extra_params):
        line = re.sub(r"(\w+)\(\)\s+([^:)]+):", r"\1(\2):", line)

        # Pattern: import something() -> import something
        line = re.sub(r"(import\s+[^()]+)\(\)", r"\1", line)

        # Pattern: from something import () -> from something import
        line = re.sub(r"(from\s+[^()]+\s+import)\s+\(\)", r"\1", line)

        # Pattern: with something() mode -> with something(mode
        line = re.sub(r"(with\s+\w+)\(\)\s+([^)]+)", r"\1(\2", line)

        # Pattern: def function() params: -> def function(params):
        line = re.sub(r"(def\s+\w+)\(\)\s+([^:)]+):", r"\1(\2):", line)

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_bracket_mismatches(content: str) -> str:
    """Fix bracket mismatches like ) vs } vs ]."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix common bracket mismatches

        # Pattern: closing parenthesis ')' does not match opening parenthesis '{'
        # This usually means ) should be }
        if "{" in line and "}" not in line and ")" in line:
            # Count brackets to see if we need to replace ) with }
            open_curly = line.count("{")
            close_curly = line.count("}")
            open_paren = line.count("(")
            close_paren = line.count(")")

            if open_curly > close_curly and close_paren > open_paren:
                # Replace last ) with }
                line = line[::-1].replace(")", "}", 1)[::-1]

        # Pattern: closing parenthesis ')' does not match opening parenthesis '['
        # This usually means ) should be ]
        if "[" in line and "]" not in line and ")" in line:
            open_square = line.count("[")
            close_square = line.count("]")
            open_paren = line.count("(")
            close_paren = line.count(")")

            if open_square > close_square and close_paren > open_paren:
                # Replace last ) with ]
                line = line[::-1].replace(")", "]", 1)[::-1]

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_fstring_issues(content: str) -> str:
    """Fix f-string issues."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix f-string: unmatched ')' - usually extra parens in f-string
        if 'f"' in line or "f'" in line:
            # Remove extra parentheses that don't belong
            # Pattern: f"text)" -> f"text"
            line = re.sub(r'f"([^"]*)\)"', r'f"\1"', line)
            line = re.sub(r"f'([^']*)\)'", r"f'\1'", line)

        # Fix unterminated string literals
        if line.count('"') % 2 == 1 and not line.strip().endswith("\\"):
            if not '"""' in line:  # Skip docstrings
                line = line + '"'

        if line.count("'") % 2 == 1 and not line.strip().endswith("\\"):
            if not "'''" in line:  # Skip docstrings
                line = line + "'"

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_indentation_issues(content: str) -> str:
    """Fix basic indentation issues."""
    lines = content.split("\n")
    fixed_lines = []
    current_indent = 0

    for i, line in enumerate(lines):
        if not line.strip():
            fixed_lines.append(line)
            continue

        # Get actual indentation
        actual_indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        # Simple rules for expected indentation
        if stripped.startswith(("def ", "class ")):
            current_indent = 0
        elif stripped.startswith(("if ", "for ", "while ", "try:", "with ")):
            # These start new blocks
            pass
        elif stripped.startswith(("elif ", "else:", "except", "finally:")):
            # These should align with the previous if/try
            if current_indent >= 4:
                current_indent -= 4

        # Fix obvious indentation errors
        if actual_indent != current_indent and line.strip():
            fixed_line = " " * current_indent + stripped
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)

        # Update indent for next line
        if line.strip().endswith(":"):
            current_indent += 4

    return "\n".join(fixed_lines)


def fix_file(file_path: Path) -> bool:
    """Fix syntax errors in a specific file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply fixes in order
        content = fix_unmatched_parentheses_v2(content)
        content = fix_bracket_mismatches(content)
        content = fix_fstring_issues(content)

        # Only apply indentation fix to files with indentation errors
        if "IndentationError" in str(file_path) or "unexpected indent" in content:
            content = fix_indentation_issues(content)

        # Only write if content changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

    return False


def main():
    """Main function to fix targeted syntax errors."""
    print("Targeted Syntax Error Fix")
    print("=" * 30)

    # Files with E999 syntax errors from the flake8 output
    problem_files = [
        "backtesting.py",
        "config.py",
        "create_dummy_training_data.py",
        "create_dummy_upcoming_data.py",
        "daily_prediction_and_backtest.py",
        "data_collection.py",
        "data_pipeline.py",
        "debug_import.py",
        "feature_engineer.py",
        "logging_config.py",
        "market_analysis.py",
        "player_performance.py",
        "predictive_modeling.py",
        "retrain_model.py",
        "test_backtesting.py",
        "test_betting_system.py",
        "test_integration_historical_backtest.py",
        "utils.py",
        "validation.py",
        "mlb_feature_engineering/config.py",
        "mlb_feature_engineering/core/cacher.py",
        "mlb_feature_engineering/core/data_preprocessing.py",
        "mlb_feature_engineering/core/feature_engineering.py",
        "mlb_feature_engineering/core/feature_set.py",
        "mlb_feature_engineering/core/game_batch.py",
        "mlb_feature_engineering/core/optimizer.py",
        "mlb_feature_engineering/core/validator.py",
        "mlb_feature_engineering/main.py",
        "mlb_feature_engineering/setup.py",
        "mlb_feature_engineering/tests/test_data_preprocessor.py",
        "mlb_feature_engineering/tests/test_game_batcher.py",
        "tests/test_betting_system.py",
        "tests/test_integration_historical_backtest.py",
    ]

    fixed_count = 0

    for file_rel_path in problem_files:
        file_path = Path(file_rel_path)
        if file_path.exists():
            print(f"Fixing {file_path}...")
            if fix_file(file_path):
                fixed_count += 1
                print(f"  ✓ Fixed {file_path}")
            else:
                print(f"  - No changes needed for {file_path}")
        else:
            print(f"  ! File not found: {file_path}")

    print(f"\nFixed {fixed_count} files.")
    print("\nNext: Run 'flake8 . | findstr E999' to see remaining syntax errors.")


if __name__ == "__main__":
    main()
