# setup_utilities.py
import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    print("Step 1: Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ ERROR: Python 3.8 or higher is required.")
        sys.exit(1)
    print("✅ Python version is compatible.")


def install_dependencies():
    print("\nStep 2: Installing required packages from requirements.txt...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✅ All packages installed successfully.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            f"❌ ERROR: Failed to install packages. Please ensure 'requirements.txt' exists and run 'pip install -r requirements.txt' manually. Error: {e}"
        )
        sys.exit(1)


def create_env_file():
    print("\nStep 3: Setting up API keys in .env file...")
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env file already exists. Skipping creation.")
        return

    print("Please provide your API keys. They will be saved locally in a .env file.")
    keys = {}
    keys["THE_ODDS_API_KEY"] = input("Enter your The Odds API Key: ").strip()
    keys["ANTHROPIC_API_KEY"] = input("Enter your Anthropic (Claude) API Key: ").strip()
    keys["OPENAI_API_KEY"] = input("Enter your OpenAI API Key: ").strip()

    with open(env_path, "w") as f:
        for key, value in keys.items():
            if value:
                f.write(f'{key}="{value}"\n')
    print("✅ .env file created successfully.")


def create_directory_structure():
    print("\nStep 4: Creating required directory structure...")
    dirs = ["ultimate_runs", "cache", "logs", "backups"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"  - Ensuring '{d}/' directory exists.")
    print("✅ Directory structure is ready.")


def validate_config():
    print("\nStep 5: Validating system configuration...")
    try:
        from tri_model_api_config import validate_api_configuration

        is_valid, errors = validate_api_configuration()
        if not is_valid:
            print(
                "❌ WARNING: Configuration validation failed with the following issues:"
            )
            for error in errors:
                print(f"  - {error}")
        else:
            print("✅ System configuration validated successfully.")
    except Exception as e:
        print(f"❌ ERROR: Could not validate configuration. {e}")


def create_launcher():
    print("\nStep 6: Creating system launcher...")
    launcher_code = """
import asyncio
# Assuming your main script is named 'ultimate_main.py' and contains the main class
from ultimate_main import UltimateTriModelBettingSystem

if __name__ == "__main__":
    print("🚀 LAUNCHING ULTIMATE TRI-MODEL MLB BETTING SYSTEM")
    print("==================================================")

    system = UltimateTriModelBettingSystem(
        bankroll=1000.0,
        max_exposure_pct=0.12
    )

    asyncio.run(system.run_ultimate_pipeline())
"""
    # --- THIS IS THE FIX ---
    # Added encoding='utf-8' to handle emojis correctly on Windows
    with open("launch_system.py", "w", encoding="utf-8") as f:
        f.write(launcher_code)
    print("✅ `launch_system.py` created. Run this file to start the system.")


if __name__ == "__main__":
    print("--- 🔧 Setting up the Ultimate Tri-Model Betting System 🔧 ---")
    check_python_version()
    install_dependencies()
    create_env_file()
    create_directory_structure()
    validate_config()
    create_launcher()
    print("\n--- 🎉 Setup Complete! ---")
    print("To start the system, run: python launch_system.py")
