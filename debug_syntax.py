# Run this to check syntax of game_selection.py
import ast
import sys


def check_syntax(filename):
    """Check syntax of a Python file"""
    try:
        with open(filename, encoding="utf-8") as f:
            content = f.read()

        # Try to parse the file
        ast.parse(content)
        print(f"✅ {filename} syntax is valid!")

        # Show first few lines for review
        lines = content.split("\n")[:10]
        print(f"\n📄 First 10 lines of {filename}:")
        for i, line in enumerate(lines, 1):
            print(f"{i:2d}: {line}")

    except SyntaxError as e:
        print(f"❌ Syntax error in {filename}:")
        print(f"   Line {e.lineno}: {e.text}")
        print(f"   Error: {e.msg}")
        print(f"   Position: {' ' * (e.offset-1)}^")

        # Show the problematic line and surrounding context
        try:
            with open(filename, encoding="utf-8") as f:
                lines = f.readlines()

            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)

            print(f"\n📄 Context around line {e.lineno}:")
            for i in range(start, end):
                marker = ">>> " if i + 1 == e.lineno else "    "
                print(f"{marker}{i+1:2d}: {lines[i].rstrip()}")

        except Exception as file_error:
            print(f"Could not read file: {file_error}")

    except FileNotFoundError:
        print(f"❌ File {filename} not found")
    except Exception as e:
        print(f"❌ Error checking {filename}: {e}")


if __name__ == "__main__":
    check_syntax("game_selection.py")
