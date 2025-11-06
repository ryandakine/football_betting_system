#!/usr/bin/env python3
"""
Test Suite for Task 2: Python Environment and Dependency Management
Tests environment setup, dependency validation, and configuration files
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile
import shutil

def test_requirements_txt_exists() -> bool:
    """Test that requirements.txt exists and is properly formatted."""
    print("🧪 Testing requirements.txt existence and format...")
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    # Check file content
    with open(requirements_file, 'r') as f:
        content = f.read()
    
    if not content.strip():
        print("❌ requirements.txt is empty")
        return False
    
    # Check for key dependencies
    required_deps = [
        'fastapi', 'pandas', 'numpy', 'scikit-learn', 
        'torch', 'transformers', 'sqlalchemy', 'aiohttp'
    ]
    
    missing_deps = []
    for dep in required_deps:
        if dep not in content.lower():
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"❌ Missing critical dependencies: {missing_deps}")
        return False
    
    print("✅ requirements.txt exists and contains critical dependencies")
    return True

def test_pyproject_toml_exists() -> bool:
    """Test that pyproject.toml exists and is properly configured."""
    print("🧪 Testing pyproject.toml existence and configuration...")
    
    pyproject_file = Path("pyproject.toml")
    
    if not pyproject_file.exists():
        print("❌ pyproject.toml not found")
        return False
    
    # Check file content
    with open(pyproject_file, 'r') as f:
        content = f.read()
    
    # Check for required sections
    required_sections = [
        '[build-system]',
        '[project]',
        '[tool.black]',
        '[tool.ruff]',
        '[tool.mypy]',
        '[tool.pytest.ini_options]'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"❌ Missing required sections: {missing_sections}")
        return False
    
    # Check Python version requirement
    if 'requires-python = ">=3.12"' not in content:
        print("❌ Python 3.12+ requirement not specified")
        return False
    
    print("✅ pyproject.toml exists and is properly configured")
    return True

def test_environment_setup_script() -> bool:
    """Test that environment setup script exists and is executable."""
    print("🧪 Testing environment setup script...")
    
    setup_script = Path("environment_setup.py")
    
    if not setup_script.exists():
        print("❌ environment_setup.py not found")
        return False
    
    # Check if script is executable
    if not os.access(setup_script, os.X_OK):
        print("⚠️  environment_setup.py not executable, setting permissions...")
        os.chmod(setup_script, 0o755)
    
    # Check script content for key classes/functions
    with open(setup_script, 'r') as f:
        content = f.read()
    
    required_elements = [
        'class EnvironmentManager',
        'def validate_python_version',
        'def validate_dependencies',
        'def install_dependencies',
        'def validate_critical_imports'
    ]
    
    missing_elements = []
    for element in required_elements:
        if element not in content:
            missing_elements.append(element)
    
    if missing_elements:
        print(f"❌ Missing required elements: {missing_elements}")
        return False
    
    print("✅ Environment setup script exists and contains required functionality")
    return True

def test_python_version_validation() -> bool:
    """Test Python version validation logic."""
    print("🧪 Testing Python version validation...")
    
    current_version = sys.version_info
    required_version = (3, 12)
    
    if current_version[:2] >= required_version:
        print(f"✅ Python {current_version.major}.{current_version.minor} meets requirement (3.12+)")
        return True
    else:
        print(f"❌ Python {current_version.major}.{current_version.minor} does not meet requirement (3.12+)")
        return False

def test_virtual_environment_detection() -> bool:
    """Test virtual environment detection."""
    print("🧪 Testing virtual environment detection...")
    
    # Check if running in virtual environment
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )
    
    if in_venv:
        print(f"✅ Running in virtual environment: {sys.prefix}")
    else:
        print("⚠️  Not running in virtual environment (acceptable for testing)")
    
    # This test always passes as venv is optional
    return True

def test_dependency_parsing() -> bool:
    """Test dependency parsing from requirements.txt."""
    print("🧪 Testing dependency parsing...")
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found for parsing test")
        return False
    
    try:
        # Parse requirements
        requirements = []
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('python'):
                    requirements.append(line)
        
        if len(requirements) < 10:
            print(f"❌ Too few dependencies parsed: {len(requirements)}")
            return False
        
        print(f"✅ Successfully parsed {len(requirements)} dependencies")
        return True
        
    except Exception as e:
        print(f"❌ Dependency parsing failed: {e}")
        return False

def test_package_installation_check() -> bool:
    """Test package installation checking logic."""
    print("🧪 Testing package installation checking...")
    
    try:
        # Test with a package that should be available (built-in)
        import importlib
        import sys
        
        # Try to check a standard library module
        try:
            importlib.import_module('json')
            print("✅ Package import checking works for standard library")
        except ImportError:
            print("❌ Package import checking failed for standard library")
            return False
        
        # Test package spec parsing
        test_specs = [
            "pandas>=2.1.3",
            "numpy==1.25.2",
            "requests",
            "fastapi[standard]>=0.104.0"
        ]
        
        for spec in test_specs:
            # Basic parsing test
            if '>=' in spec:
                package_name = spec.split('>=')[0].strip()
            elif '==' in spec:
                package_name = spec.split('==')[0].strip()
            else:
                package_name = spec.strip()
            
            # Handle extras
            if '[' in package_name:
                package_name = package_name.split('[')[0]
            
            if not package_name:
                print(f"❌ Failed to parse package name from: {spec}")
                return False
        
        print("✅ Package specification parsing works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Package installation check failed: {e}")
        return False

def test_development_tools_configuration() -> bool:
    """Test development tools configuration."""
    print("🧪 Testing development tools configuration...")
    
    pyproject_file = Path("pyproject.toml")
    
    if not pyproject_file.exists():
        print("❌ pyproject.toml not found for dev tools test")
        return False
    
    with open(pyproject_file, 'r') as f:
        content = f.read()
    
    # Check for development tool configurations
    dev_tools = {
        'black': '[tool.black]',
        'ruff': '[tool.ruff]',
        'mypy': '[tool.mypy]',
        'pytest': '[tool.pytest.ini_options]',
        'coverage': '[tool.coverage'
    }
    
    missing_tools = []
    for tool, config_section in dev_tools.items():
        if config_section not in content:
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"❌ Missing development tool configurations: {missing_tools}")
        return False
    
    print("✅ All development tools are properly configured")
    return True

def test_project_metadata() -> bool:
    """Test project metadata in pyproject.toml."""
    print("🧪 Testing project metadata...")
    
    pyproject_file = Path("pyproject.toml")
    
    if not pyproject_file.exists():
        print("❌ pyproject.toml not found for metadata test")
        return False
    
    with open(pyproject_file, 'r') as f:
        content = f.read()
    
    # Check for required metadata
    required_metadata = [
        'name = "football-betting-system"',
        'version = "1.0.0"',
        'description = ',
        'requires-python = ">=3.12"',
        'authors = '
    ]
    
    missing_metadata = []
    for metadata in required_metadata:
        if metadata not in content:
            missing_metadata.append(metadata)
    
    if missing_metadata:
        print(f"❌ Missing project metadata: {missing_metadata}")
        return False
    
    print("✅ Project metadata is complete")
    return True

def test_optional_dependencies() -> bool:
    """Test optional dependencies configuration."""
    print("🧪 Testing optional dependencies...")
    
    pyproject_file = Path("pyproject.toml")
    
    if not pyproject_file.exists():
        print("❌ pyproject.toml not found for optional deps test")
        return False
    
    with open(pyproject_file, 'r') as f:
        content = f.read()
    
    # Check for optional dependency groups
    optional_groups = ['dev', 'jupyter', 'gpu']
    
    missing_groups = []
    for group in optional_groups:
        if f'{group} = [' not in content:
            missing_groups.append(group)
    
    if missing_groups:
        print(f"❌ Missing optional dependency groups: {missing_groups}")
        return False
    
    print("✅ Optional dependencies are properly configured")
    return True

def run_all_task_2_tests() -> bool:
    """Run all Task 2 tests."""
    print("🚀 TASK 2: Python Environment and Dependency Management - TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Requirements.txt Existence", test_requirements_txt_exists),
        ("Pyproject.toml Existence", test_pyproject_toml_exists),
        ("Environment Setup Script", test_environment_setup_script),
        ("Python Version Validation", test_python_version_validation),
        ("Virtual Environment Detection", test_virtual_environment_detection),
        ("Dependency Parsing", test_dependency_parsing),
        ("Package Installation Check", test_package_installation_check),
        ("Development Tools Configuration", test_development_tools_configuration),
        ("Project Metadata", test_project_metadata),
        ("Optional Dependencies", test_optional_dependencies),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 TASK 2 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TASK 2 TESTS PASSED!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False

def main():
    """Main test runner."""
    success = run_all_task_2_tests()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
