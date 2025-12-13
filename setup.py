"""
Setup script for NASEA.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().split("\n")
        if line.strip() and not line.startswith("#") and not line.startswith("python")
    ]

setup(
    name="nasea",
    version="0.2.0-alpha",
    author="Alex & Abat",
    description="Natural-Language Autonomous Software-Engineering Agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AbatChan/nasea",  # Update with actual URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Code Generators",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nasea=nasea.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
