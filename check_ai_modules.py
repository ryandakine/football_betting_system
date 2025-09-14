# Check what AI modules exist and their status
import os
import sys


def check_ai_modules():
    """Check AI module files and their import status"""

    # Common AI module files in MLB betting systems
    ai_files = [
        "game_selection.py",
        "ai_predictor.py",
        "ml_predictor.py",
        "betting_ai.py",
        "prediction_engine.py",
        "ai_analyzer.py",
    ]

    print("🔍 CHECKING AI MODULE STATUS")
    print("=" * 50)

    # Check which files exist
    existing_files = []
    for file in ai_files:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")

    print(f"\n📊 Found {len(existing_files)} AI module files")

    # Test imports for existing files
    print("\n🧪 TESTING IMPORTS:")
    print("-" * 30)

    for file in existing_files:
        module_name = file.replace(".py", "")
        try:
            # Try to import the module
            spec = __import__(module_name)
            print(f"✅ {file}: Import successful")

            # Check for key classes/functions
            if hasattr(spec, "GameSelector"):
                print(f"   📦 Has GameSelector class")
            if hasattr(spec, "AIPredictor"):
                print(f"   📦 Has AIPredictor class")
            if hasattr(spec, "analyze_games"):
                print(f"   📦 Has analyze_games function")

        except ImportError as e:
            print(f"❌ {file}: Import failed - {e}")
        except SyntaxError as e:
            print(f"❌ {file}: Syntax error - Line {e.lineno}: {e.msg}")
        except Exception as e:
            print(f"❌ {file}: Other error - {e}")

    # Check what your main system expects
    print(f"\n🎯 CHECKING MAIN SYSTEM EXPECTATIONS:")
    print("-" * 40)

    try:
        with open("daily_prediction_and_backtest.py") as f:
            content = f.read()

        # Look for AI import patterns
        if "from game_selection import" in content:
            print("✅ Main system expects: game_selection module")
        if "from ai_predictor import" in content:
            print("✅ Main system expects: ai_predictor module")
        if "import ai_" in content:
            print("✅ Main system expects: ai_ modules")

        # Find the specific imports
        import re

        imports = re.findall(r"from (\w+) import|import (\w+)", content)
        ai_imports = [
            imp
            for sublist in imports
            for imp in sublist
            if imp
            and (
                "ai" in imp.lower() or "game" in imp.lower() or "predict" in imp.lower()
            )
        ]

        if ai_imports:
            print(f"📋 Expected AI modules: {', '.join(set(ai_imports))}")

    except Exception as e:
        print(f"❌ Could not analyze main system: {e}")


if __name__ == "__main__":
    check_ai_modules()
