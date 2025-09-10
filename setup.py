#!/usr/bin/env python3
"""
AI Scheduler setup file
"""

from setuptools import setup, find_packages

setup(
    name="NNScheduler",
    version="0.1.0",
    description="Neural Network Task Scheduler",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "pytest",
    ],
)