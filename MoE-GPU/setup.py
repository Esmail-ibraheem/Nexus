"""
Setup script for Expert-Sliced GPU Scheduling.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="moe-gpu-scheduling",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Expert-Sliced GPU Scheduling for Mixture of Experts Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/moe-gpu-scheduling",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "moe-benchmark=moe_gpu.benchmark:run_quick_benchmark",
        ],
    },
    include_package_data=True,
    keywords=[
        "mixture-of-experts",
        "gpu-scheduling",
        "cuda",
        "triton",
        "deep-learning",
        "pytorch",
        "optimization",
        "energy-efficiency",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/moe-gpu-scheduling/issues",
        "Source": "https://github.com/yourusername/moe-gpu-scheduling",
        "Documentation": "https://github.com/yourusername/moe-gpu-scheduling/blob/main/README.md",
        "Paper": "https://github.com/yourusername/moe-gpu-scheduling/blob/main/paper/research_paper.md",
    },
)
