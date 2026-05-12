from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hiersolv",
    version="1.0.0",
    author="Research Team",
    description="HierSolv: Hierarchical Solubility Prediction via Multi-Scale Interaction Graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hiersolv",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.4.0",
        "torch-scatter",
        "torch-sparse",
        "rdkit>=2023.9.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "matplotlib>=3.8.0",
        "scipy>=1.11.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "seaborn>=0.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hiersolv-train=run_experiment:main",
            "hiersolv-predict=predict:main",
            "hiersolv-download=data.download_data:main",
        ],
    },
)
