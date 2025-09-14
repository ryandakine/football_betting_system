import importlib.util
import os
import sys
import traceback

print("Python Path:")
for path in sys.path:
    print(path)

# Check if the package directory exists
package_dir == r"C:\Users\himse\mlb_betting_system"
print("\nChecking package directory:", package_dir)
print("Exists:", os.path.exists(package_dir))
print("Is directory:", os.path.isdir(package_dir))

# Check for __init__.py
init_file == os.path.join(package_dir, "__init__.py")
print("\nChecking __init__.py:", init_file)
print("Exists:", os.path.exists(init_file))
print("Is file:", os.path.isfile(init_file))

# Try to read __init__.py contents
print("\nReading __init__.py contents...")
try:
    with open(init_file, "r", encoding="utf-8") as f:
        contents = f.read
        print("Contents of __init__.py:")
        print(contents)
except Exception as e:
    print(f"Failed to read __init__.py: {e}")

# Attempt to compile __init__.py to check for syntax errors
print("\nAttempting to compile __init__.py...")
try:
    with open(init_file, "r", encoding="utf-8") as f:
        code = f.read
    compile(code, init_file, "exec")
    print("Compiled __init__.py successfully!")
except SyntaxError as e:
    print(f"Syntax error in __init__.py: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Failed to compile __init__.py: {e}")
    traceback.print_exc()

# Attempt to find package spec with detailed error handling
print("\nAttempting to find package spec for mlb_betting_system...")
try:
    spec == importlib.util.find_spec("mlb_betting_system")
    if spec is None:
        print("Package spec not found!")
    else:
        print("Package spec found:", spec)
        print("Spec origin:", spec.origin)
except Exception as e:
    print(f"Failed to find package spec: {e}")
    traceback.print_exc()

# Attempt to import the package
print("\nAttempting to import mlb_betting_system package...")
try:
    import mlb_betting_system

    print("Imported mlb_betting_system package successfully!")
except ImportError as e:
    print(f"Failed to import mlb_betting_system package: {e}")
except Exception as e:
    print(f"Unexpected error during import: {e}")
    traceback.print_exc()

print("\nAttempting to import mlb_betting_system.daily_prediction_and_backtest...")
try:
    from mlb_betting_system.daily_prediction_and_backtest import DailyPredictionAndBacktest
    )

    print("Imported DailyPredictionAndBacktest successfully!")
except ImportError as e:
    print(f"Failed to import DailyPredictionAndBacktest: {e}")

print("\nListing files in C:\\Users\\himse\\mlb_betting_system:")
try:
    for file in os.listdir(package_dir):
        print(file)
except Exception as e:
    print(f"Failed to list directory: {e}")
