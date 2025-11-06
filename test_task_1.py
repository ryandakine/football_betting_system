#!/usr/bin/env python3
"""
Task 1 validation helpers used by run_task_1_tests.py.
Checks that core repository scaffolding and CI wiring are present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple


def _assert_file_exists(path: Path, description: str) -> bool:
    """Ensure a file exists and is non-empty."""
    if not path.exists():
        print(f"❌ Missing {description}: {path}")
        return False

    # Some generated artifacts may be tiny, but critical files should contain content.
    if path.is_file():
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            print(f"❌ {description} is empty: {path}")
            return False

    print(f"✅ {description} present: {path}")
    return True


def _assert_directory_exists(path: Path, description: str) -> bool:
    """Ensure a directory exists and is not empty."""
    if not path.exists():
        print(f"❌ Missing {description}: {path}")
        return False

    if not path.is_dir():
        print(f"❌ {description} should be a directory: {path}")
        return False

    if not any(path.iterdir()):
        print(f"❌ {description} directory is empty: {path}")
        return False

    print(f"✅ {description} present: {path}")
    return True


def test_readme_exists() -> bool:
    """Verify repository documentation entry point exists."""
    return _assert_file_exists(Path("README.md"), "README file")


def test_requirements_exists() -> bool:
    """Verify base dependency list exists."""
    return _assert_file_exists(Path("requirements.txt"), "requirements.txt")


def test_pyproject_exists() -> bool:
    """Verify project metadata file exists."""
    return _assert_file_exists(Path("pyproject.toml"), "pyproject.toml")


def test_ci_workflow_exists() -> bool:
    """Verify CI workflow configuration is available."""
    return _assert_file_exists(
        Path(".github/workflows/ci.yml"),
        "GitHub Actions CI workflow",
    )


def test_docs_directory_present() -> bool:
    """Ensure project documentation directory exists."""
    return _assert_directory_exists(Path("docs"), "docs directory")


def run_all_task_1_tests() -> bool:
    """Run all Task 1 checks and return success status."""
    tests: List[Tuple[str, Callable[[], bool]]] = [
        ("Repository README", test_readme_exists),
        ("Requirements file", test_requirements_exists),
        ("Pyproject metadata", test_pyproject_exists),
        ("CI workflow", test_ci_workflow_exists),
        ("Documentation directory", test_docs_directory_present),
    ]

    passed = 0
    for name, test_func in tests:
        print(f"\n📋 {name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
                print(f"✅ {name}: PASSED")
            else:
                print(f"❌ {name}: FAILED")
        except Exception as exc:  # pragma: no cover - defensive guardrail
            print(f"❌ {name}: ERROR - {exc}")

    total = len(tests)
    print("\n" + "=" * 60)
    print(f"📊 Task 1 results: {passed}/{total} checks passed")
    success = passed == total
    if success:
        print("🎉 Task 1 scaffolding looks good!")
    else:
        print("⚠️  Task 1 verification failed")
    return success


if __name__ == "__main__":
    raise SystemExit(0 if run_all_task_1_tests() else 1)

