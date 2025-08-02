"""
SemEval 2025 Task 11: Multilingual Emotion Detection
Setup script for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#') and not line.startswith('-')
        ]
else:
    requirements = [
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "nltk>=3.6.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "PyYAML>=5.4.0",
    ]

# Development requirements
dev_requirements = [
    "pytest>=6.2.0",
    "pytest-cov>=2.12.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "isort>=5.9.0",
    "jupyter>=1.0.0",
    "notebook>=6.4.0",
]

setup(
    name="semeval2025-emotion-detection",
    version="1.0.0",
    author="Your Name",  # Update with your name
    author_email="your.email@example.com",  # Update with your email
    description="SemEval 2025 Task 11: Multilingual Emotion Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/semeval2025-task11-emotion-detection",  # Update with your repo URL
    project_urls={
        "Bug Reports": "https://github.com/yourusername/semeval2025-task11-emotion-detection/issues",
        "Source": "https://github.com/yourusername/semeval2025-task11-emotion-detection",
        "Documentation": "https://github.com/yourusername/semeval2025-task11-emotion-detection/blob/main/docs/",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "all": dev_requirements + [
            "wandb>=0.12.0",
            "spacy>=3.4.0",
            "langdetect>=1.0.9",
            "fastapi>=0.70.0",
            "uvicorn>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "semeval-pipeline=scripts.run_full_pipeline:main",
            "semeval-explore=scripts.run_data_exploration:main",
            "semeval-preprocess=scripts.run_preprocessing:main",
            "semeval-baseline=scripts.run_baseline_training:main",
            "semeval-transformer=scripts.run_transformer_training:main",
            "semeval-evaluate=scripts.run_evaluation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "nlp",
        "emotion-detection",
        "multilingual",
        "transformers",
        "bert",
        "semeval",
        "machine-learning",
        "deep-learning",
        "pytorch",
        "sentiment-analysis",
    ],
)