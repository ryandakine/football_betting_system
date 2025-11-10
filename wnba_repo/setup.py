#!/usr/bin/env python3
"""
Setup script for WNBA Betting System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="wnba-betting-system",
    version="1.0.0",
    description="Advanced AI-powered betting intelligence system for WNBA (Women's National Basketball Association)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/wnba-betting-system",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "requests>=2.26.0",
        "aiohttp>=3.8.0",
        "pydantic>=1.9.0",
        "python-dotenv>=0.19.0",
        "pytz>=2021.3",
    ],
    extras_require={
        "ai": [
            "anthropic>=0.3.0",
            "openai>=0.27.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wnba-analyze=main_analyzer:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="sports betting wnba basketball professional womens ai machine-learning",
)
