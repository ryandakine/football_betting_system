#!/usr/bin/env python3
"""
Environment Setup and Validation for Football Betting System
Handles Python environment, dependency management, and system validation
"""

import sys
import os
import subprocess
import importlib
import pkg_resources
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import platform
import shutil
from dataclasses import dataclass
from enum import Enum


class SetupStatus(Enum):
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationResult:
    status: SetupStatus
    message: str
    details: Optional[str] = None


class EnvironmentManager:
    """Manages Python environment setup and validation."""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.architecture = platform.machine()
        self.results: List[ValidationResult] = []
    
    def validate_python_version(self) -> ValidationResult:
        """Validate Python version meets requirements."""
        required_version = (3, 12)
        current_version = self.python_version[:2]
        
        if current_version >= required_version:
            return ValidationResult(
                SetupStatus.SUCCESS,
                f"Python {self.python_version.major}.{self.python_version.minor} meets requirement (>=3.12)"
            )
        else:
            return ValidationResult(
                SetupStatus.ERROR,
                f"Python {self.python_version.major}.{self.python_version.minor} does not meet requirement (>=3.12)",
                "Please upgrade to Python 3.12 or higher"
            )
    
    def check_virtual_environment(self) -> ValidationResult:
        """Check if running in virtual environment."""
        in_venv = (
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )
        
        if in_venv:
            return ValidationResult(
                SetupStatus.SUCCESS,
                f"Running in virtual environment: {sys.prefix}"
            )
        else:
            return ValidationResult(
                SetupStatus.WARNING,
                "Not running in virtual environment",
                "Consider using venv or conda for isolation"
            )
    
    def validate_dependencies(self) -> ValidationResult:
        """Validate that all required dependencies are available."""
        requirements_file = Path("requirements.txt")
        
        if not requirements_file.exists():
            return ValidationResult(
                SetupStatus.ERROR,
                "requirements.txt not found",
                "Run 'pip freeze > requirements.txt' to create it"
            )
        
        try:
            with open(requirements_file, 'r') as f:
                requirements = [
                    line.strip() for line in f 
                    if line.strip() and not line.startswith('#')
                ]
            
            missing_packages = []
            installed_packages = {pkg.project_name.lower(): pkg.version 
                                for pkg in pkg_resources.working_set}
            
            for req in requirements:
                if '>=' in req:
                    package_name = req.split('>=')[0].strip().lower()
                elif '==' in req:
                    package_name = req.split('==')[0].strip().lower()
                else:
                    package_name = req.strip().lower()
                
                # Handle extras
                if '[' in package_name:
                    package_name = package_name.split('[')[0]
                
                if package_name not in installed_packages:
                    missing_packages.append(req)
            
            if missing_packages:
                return ValidationResult(
                    SetupStatus.WARNING,
                    f"Missing {len(missing_packages)} packages",
                    f"Missing: {', '.join(missing_packages[:5])}"
                )
            else:
                return ValidationResult(
                    SetupStatus.SUCCESS,
                    f"All {len(requirements)} dependencies are installed"
                )
                
        except Exception as e:
            return ValidationResult(
                SetupStatus.ERROR,
                "Failed to validate dependencies",
                str(e)
            )
    
    def install_dependencies(self, upgrade: bool = False) -> ValidationResult:
        """Install dependencies from requirements.txt."""
        requirements_file = Path("requirements.txt")
        
        if not requirements_file.exists():
            return ValidationResult(
                SetupStatus.ERROR,
                "requirements.txt not found for installation"
            )
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            if upgrade:
                cmd.append("--upgrade")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            return ValidationResult(
                SetupStatus.SUCCESS,
                "Dependencies installed successfully",
                result.stdout[-200:] if result.stdout else None
            )
            
        except subprocess.CalledProcessError as e:
            return ValidationResult(
                SetupStatus.ERROR,
                "Failed to install dependencies",
                e.stderr[-200:] if e.stderr else str(e)
            )
    
    def validate_critical_imports(self) -> ValidationResult:
        """Validate that critical packages can be imported."""
        critical_packages = [
            'pandas', 'numpy', 'sklearn', 'torch', 'transformers',
            'fastapi', 'sqlalchemy', 'aiohttp', 'asyncio'
        ]
        
        failed_imports = []
        
        for package in critical_packages:
            try:
                importlib.import_module(package)
            except ImportError as e:
                failed_imports.append(f"{package}: {str(e)}")
        
        if failed_imports:
            return ValidationResult(
                SetupStatus.ERROR,
                f"Failed to import {len(failed_imports)} critical packages",
                "; ".join(failed_imports[:3])
            )
        else:
            return ValidationResult(
                SetupStatus.SUCCESS,
                f"All {len(critical_packages)} critical packages imported successfully"
            )
    
    def check_system_requirements(self) -> ValidationResult:
        """Check system-level requirements."""
        issues = []
        
        # Check available memory
        try:
            if self.platform == "Linux":
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            mem_kb = int(line.split()[1])
                            mem_gb = mem_kb / (1024 * 1024)
                            if mem_gb < 8:
                                issues.append(f"Low memory: {mem_gb:.1f}GB (recommend 16GB+)")
                            break
        except:
            pass
        
        # Check disk space
        try:
            disk_usage = shutil.disk_usage(Path.cwd())
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 10:
                issues.append(f"Low disk space: {free_gb:.1f}GB free")
        except:
            pass
        
        # Check Git
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            issues.append("Git not available")
        
        if issues:
            return ValidationResult(
                SetupStatus.WARNING,
                "System requirement issues detected",
                "; ".join(issues)
            )
        else:
            return ValidationResult(
                SetupStatus.SUCCESS,
                "System requirements check passed"
            )
    
    def setup_development_tools(self) -> ValidationResult:
        """Setup development tools like pre-commit hooks."""
        try:
            # Install pre-commit if available
            if Path(".pre-commit-config.yaml").exists():
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "pre-commit"
                ], capture_output=True, check=True)
                
                subprocess.run([
                    "pre-commit", "install"
                ], capture_output=True, check=True)
                
                return ValidationResult(
                    SetupStatus.SUCCESS,
                    "Development tools configured (pre-commit hooks installed)"
                )
            else:
                return ValidationResult(
                    SetupStatus.WARNING,
                    "No .pre-commit-config.yaml found",
                    "Pre-commit hooks not configured"
                )
                
        except Exception as e:
            return ValidationResult(
                SetupStatus.WARNING,
                "Failed to setup development tools",
                str(e)
            )
    
    def create_env_file(self) -> ValidationResult:
        """Create .env file from template if it doesn't exist."""
        env_file = Path(".env")
        env_example = Path(".env.example")
        
        if env_file.exists():
            return ValidationResult(
                SetupStatus.SUCCESS,
                ".env file already exists"
            )
        
        if env_example.exists():
            try:
                shutil.copy(env_example, env_file)
                return ValidationResult(
                    SetupStatus.SUCCESS,
                    ".env file created from template",
                    "Remember to add your API keys"
                )
            except Exception as e:
                return ValidationResult(
                    SetupStatus.ERROR,
                    "Failed to create .env file",
                    str(e)
                )
        else:
            # Create basic .env file
            env_content = """# Football Betting System Environment Variables
# API Keys
ODDS_API_KEY=your_odds_api_key_here
WEATHER_API_KEY=your_weather_api_key_here
SPORTS_DATA_API_KEY=your_sports_data_api_key_here

# AI Model APIs
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here

# Database
DATABASE_URL=sqlite:///football_betting.db
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here

# Development
DEBUG=True
LOG_LEVEL=INFO
"""
            try:
                with open(env_file, 'w') as f:
                    f.write(env_content)
                
                return ValidationResult(
                    SetupStatus.SUCCESS,
                    ".env file created with template",
                    "Remember to add your actual API keys"
                )
            except Exception as e:
                return ValidationResult(
                    SetupStatus.ERROR,
                    "Failed to create .env file",
                    str(e)
                )
    
    def run_full_setup(self, install_deps: bool = False) -> Dict[str, ValidationResult]:
        """Run complete environment setup and validation."""
        print("🚀 Football Betting System - Environment Setup")
        print("=" * 60)
        
        checks = [
            ("Python Version", self.validate_python_version),
            ("Virtual Environment", self.check_virtual_environment),
            ("System Requirements", self.check_system_requirements),
            ("Dependencies", self.validate_dependencies),
            ("Critical Imports", self.validate_critical_imports),
            ("Development Tools", self.setup_development_tools),
            ("Environment File", self.create_env_file),
        ]
        
        if install_deps:
            checks.insert(4, ("Install Dependencies", lambda: self.install_dependencies()))
        
        results = {}
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}")
            print("-" * 40)
            
            try:
                result = check_func()
                results[check_name] = result
                
                status_icon = {
                    SetupStatus.SUCCESS: "✅",
                    SetupStatus.WARNING: "⚠️",
                    SetupStatus.ERROR: "❌"
                }[result.status]
                
                print(f"{status_icon} {result.message}")
                if result.details:
                    print(f"   Details: {result.details}")
                    
            except Exception as e:
                result = ValidationResult(
                    SetupStatus.ERROR,
                    f"Check failed with exception: {str(e)}"
                )
                results[check_name] = result
                print(f"❌ {result.message}")
        
        # Summary
        print("\n" + "=" * 60)
        success_count = sum(1 for r in results.values() if r.status == SetupStatus.SUCCESS)
        warning_count = sum(1 for r in results.values() if r.status == SetupStatus.WARNING)
        error_count = sum(1 for r in results.values() if r.status == SetupStatus.ERROR)
        
        print(f"📊 Setup Results: {success_count} ✅ | {warning_count} ⚠️ | {error_count} ❌")
        
        if error_count == 0:
            print("🎉 Environment setup completed successfully!")
        elif error_count > 0:
            print("⚠️ Environment setup completed with errors - please review")
        
        return results


def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Football Betting System Environment Setup")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade dependencies")
    parser.add_argument("--check-only", action="store_true", help="Only run checks, no setup")
    
    args = parser.parse_args()
    
    manager = EnvironmentManager()
    
    if args.check_only:
        results = manager.run_full_setup(install_deps=False)
    else:
        results = manager.run_full_setup(install_deps=args.install)
    
    # Exit with error code if there are any errors
    error_count = sum(1 for r in results.values() if r.status == SetupStatus.ERROR)
    sys.exit(error_count)


if __name__ == "__main__":
    main()
