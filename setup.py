"""
Setup script for PyNeural - Python bindings for Neural Assembly Framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="pyneural",
    version="1.0.0",
    author="Neural Assembly Team",
    description="Python bindings for the Neural Assembly deep learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JayabrataBasu/Neural-Assembly",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[],
    extras_require={
        "numpy": ["numpy>=1.19.0"],
        "dev": [
            "numpy>=1.19.0",
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Assembly",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="deep-learning neural-network assembly x86-64 simd",
    project_urls={
        "Bug Reports": "https://github.com/JayabrataBasu/Neural-Assembly/issues",
        "Source": "https://github.com/JayabrataBasu/Neural-Assembly",
    },
)
