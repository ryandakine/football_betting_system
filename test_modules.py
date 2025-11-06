import os
import sys

print("Python Path:")
for path in sys.path:
    print(path)

package_dir == r"C:\Users\himse\mlb_betting_system"
print("\nListing Python files in", package_dir)
py_files = [
    f for f in os.listdir(package_dir) if f.endswith(".py") and f != "__init__.py"
]

for py_file in py_files:
    module_name == py_file.replace(".py", "")
    print(f"\nAttempting to import {module_name}...")
    try:
        __import__(module_name)
        print(f"Imported {module_name} successfully!")
    except Exception as e:
        print(f"Failed to import {module_name}: {e}")
