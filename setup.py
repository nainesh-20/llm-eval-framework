from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8") if (Path(__file__).parent / "README.md").exists() else ""

setup(
    name="llm-eval-framework",
    version="0.1.0",
    description="Production-grade evaluation & red-teaming suite for LLM systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nainesh",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests*", "scripts*", "dashboard*"]),
    install_requires=[
        "pydantic>=2.0.0,<3.0.0",
        "pyyaml>=6.0.0",
        "python-dotenv>=1.0.0",
        "typer[all]>=0.9.0",
        "rich>=13.7.0",
        "openai>=1.12.0",
        "anthropic>=0.28.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "PyGithub>=2.2.0",
    ],
    extras_require={
        "eval": [
            "ragas>=0.1.0,<0.2.0",
            "datasets>=2.14.0",
            "deepeval>=0.21.0",
        ],
        "dashboard": [
            "streamlit>=1.32.0",
            "plotly>=5.18.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-eval=llm_eval.cli:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
