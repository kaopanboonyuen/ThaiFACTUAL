#!/usr/bin/env python3
# =============================================================================
# ThaiFACTUAL Setup
# =============================================================================
# Author: Teerapong Panboonyuen (aka Kao Panboonyuen)
# Paper : Debiasing Large Language Models in Thai Political Stance Detection
#         via Counterfactual Calibration
#         EMNLP 2025 – Widening NLP (WiNLP) Workshop, Suzhou, China
# =============================================================================

from setuptools import setup, find_packages

setup(
    name             = "thaifactual",
    version          = "1.0.0",
    description      = "Debiasing LLMs in Thai Political Stance Detection via Counterfactual Calibration",
    long_description = open("README.md", encoding="utf-8").read(),
    long_description_content_type = "text/markdown",
    author           = "Teerapong Panboonyuen",
    author_email     = "teerapong.pa@chula.ac.th",
    url              = "https://github.com/kaopanboonyuen/ThaiFACTUAL",
    packages         = find_packages(),
    python_requires  = ">=3.9",
    install_requires = [
        "openai>=1.30.0",
        "transformers>=4.40.0",
        "torch>=2.2.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scipy>=1.13.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "tqdm>=4.66.0",
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords = [
        "Thai NLP", "stance detection", "political bias",
        "counterfactual", "debiasing", "LLM", "EMNLP 2025"
    ],
)
