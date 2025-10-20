# ============================================================================
# ARQUIVO: setup.py
# ============================================================================
"""
Script de instalação do pacote
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="titanic-analysis",
    version="1.0.0",
    author="Lucas",
    author_email="seu.email@exemplo.com",
    description="Análise comparativa de ML para o dataset Titanic",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/titanic-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "titanic-analysis=titanic_analysis.cli:main",
        ],
    },
)