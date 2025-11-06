#!/usr/bin/env python3
"""
Emergency File Restoration
Fixes the damage caused by the previous syntax fixer.
"""

import re
from pathlib import Path


def fix_unmatched_parentheses(content: str) -> str:
    """Remove extra parentheses added by the broken fixer."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Remove obvious broken patterns

        # Fix: def function() -> def function(
        line = re.sub(r"def\s+(\w+)\(\)\s+", r"def \1(", line)

        # Fix: import something() -> import something
        line = re.sub(r"import\s+([^()]+)\(\)", r"import \1", line)

        # Fix: from something import () -> from something import
        line = re.sub(r"from\s+([^()]+)\s+import\s+\(\)", r"from \1 import ", line)

        # Fix: variable == value() -> variable = value
        line = re.sub(r"(\w+)\s*==\s*([^()=]+)\(\)", r"\1 = \2", line)

        # Fix: function() something -> function(something)
        line = re.sub(r"(\w+)\(\)\s+([^()]+)", r"\1(\2)", line)

        # Fix: field() default_factory == -> field(default_factory=
        line = re.sub(r"field\(\)\s+([^()=]+)\s*==", r"field(\1=", line)

        # Fix: with something() mode -> with something(mode
        line = re.sub(r"with\s+([^()]+)\(\)\s+([^()]+)", r"with \1(\2", line)

        # Fix obvious syntax errors
        line = re.sub(r"==\s*lambda:", r"=lambda:", line)
        line = re.sub(r"==\s*False", r"is False", line)
        line = re.sub(r"==\s*True", r"is True", line)
        line = re.sub(r"==\s*None", r"is None", line)

        # Fix parameter defaults
        line = re.sub(r"(\w+):\s*(\w+)\s*==\s*([^,)]+)", r"\1: \2 = \3", line)

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_specific_patterns(content: str) -> str:
    """Fix specific broken patterns."""

    # Fix import issues
    content = re.sub(r"import loggingimport", "import logging\nimport", content)

    # Fix class definitions
    content = re.sub(r"class\s+(\w+)\(\)\s*:", r"class \1:", content)

    # Fix function definitions with broken parameters
    content = re.sub(r"def\s+(\w+)\(\)\s+([^:]+):", r"def \1(\2):", content)

    # Fix dataclass field patterns
    content = re.sub(
        r"field\(default_factory\s*==\s*lambda:",
        r"field(default_factory=lambda:",
        content,
    )

    # Fix comparison operators that were wrongly changed
    content = re.sub(r"(\w+)\s*==\s*([\d.]+)", r"\1 = \2", content)

    return content


def fix_file(file_path: Path) -> bool:
    """Fix a damaged file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_unmatched_parentheses(content)
        content = fix_specific_patterns(content)

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
    """Main restoration function."""
    print("Emergency File Restoration")
    print("=" * 30)

    # Get all Python files
    python_files = list(Path(".").rglob("*.py"))

    # Skip our fixer scripts
    skip_files = [
        "emergency_fix.py",
        "emergency_syntax_fixer.py",
        "simple_flake8_fixer.py",
    ]
    python_files = [f for f in python_files if f.name not in skip_files]

    fixed_count = 0

    for file_path in python_files:
        if fix_file(file_path):
            fixed_count += 1
            print(f"Restored: {file_path}")

    print(f"\nRestored {fixed_count} files.")

    # Clean up broken fixer scripts
    cleanup_files = [
        "emergency_syntax_fixer.py",
        "simple_flake8_fixer.py",
        "syntax_fixer.py",
        "mlb_feature_engineering/syntax_fixer.py",
        "mlb_feature_engineering/flake8_autofix.py",
        "mlb_feature_engineering/line_length_fixer.py",
    ]

    for cleanup_file in cleanup_files:
        cleanup_path = Path(cleanup_file)
        if cleanup_path.exists():
            try:
                cleanup_path.unlink()
                print(f"Removed broken script: {cleanup_file}")
            except:
                pass


if __name__ == "__main__":
    main()
