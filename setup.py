#!/usr/bin/env python3
"""
Setup for Code Attractor System
pip install attractor-kit
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="attractor-kit",
    version="1.0.0",
    author="Knowledge Attractors Team",
    author_email="team@knowledge-attractors.dev",
    description="Zero-finetune repo-aware code generation for small models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/knowledge-attractors/attractor-kit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "ollama": [
            "requests>=2.28.0",
        ],
        "validation": [
            "ruff>=0.1.0",
            "pylint>=2.15.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "attractor-mine=cli_module:main",
            "attractor-test=qwen_code_test:main",
            "attractor-validate=code_validator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.md"],
    },
    keywords=[
        "code-generation",
        "ai",
        "machine-learning",
        "software-engineering",
        "pattern-mining",
        "code-quality",
        "ollama",
        "qwen",
        "repo-aware",
        "zero-finetune"
    ],
    project_urls={
        "Bug Reports": "https://github.com/knowledge-attractors/attractor-kit/issues",
        "Source": "https://github.com/knowledge-attractors/attractor-kit",
        "Documentation": "https://knowledge-attractors.dev/docs",
    },
)