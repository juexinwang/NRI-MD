"""
Setup script for NRI-MD package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nri-md",
    version="1.0.0",
    author="Jingxuan Zhu, Juexin Wang, Weiwei Han, Dong Xu",
    author_email="",
    description="Neural relational inference to learn allosteric long-range interactions in proteins from molecular dynamics simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juexinwang/NRI-MD",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0,<2.2.0",
        "numpy>=1.21.0,<1.25.0",
        "scipy>=1.7.0,<1.12.0",
        "matplotlib>=3.5.0,<3.8.0",
        "seaborn>=0.11.0,<0.13.0",
        "pandas>=1.3.0,<2.1.0",
        "networkx>=2.6.0,<3.2.0",
    ],
    extras_require={
        "cuda118": [
            "torch==2.0.1+cu118",
            "torchvision==0.15.2+cu118",
            "torchaudio==2.0.2+cu118",
        ],
        "dev": [
            "pytest>=7.0.0,<8.0.0",
            "pytest-cov>=4.0.0,<5.0.0",
            "black>=22.0.0,<24.0.0",
            "flake8>=4.0.0,<7.0.0",
            "mypy>=0.991,<1.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nri-md-train=nri_md.main:main",
            "nri-md-convert=nri_md.convert_dataset:main",
            "nri-md-visualize=nri_md.postanalysis_visual:main",
            "nri-md-path=nri_md.postanalysis_path:main",
        ],
    },
)