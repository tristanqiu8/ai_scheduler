#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Scheduler Setup Script
Package: ai-scheduler
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# 读取README文件
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "AI Scheduler - Neural Network Task Scheduler with Priority Optimization"

# 读取版本信息
def get_version():
    version_file = Path(__file__).parent / "ai_scheduler" / "__init__.py"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.3"

# 读取依赖
def get_requirements():
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        with open(req_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "plotly>=4.14.0",
        "python-dateutil>=2.8.0",
        "requests>=2.25.0",
        "protobuf>=4.21.0",
    ]

setup(
    name="ai-scheduler",
    version=get_version(),
    author="Tristan.Qiu",
    author_email="tristan.qiu@example.com",  # 请替换为实际邮箱
    description="Neural Network Task Scheduler with Priority Optimization",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ai-scheduler",  # 请替换为实际仓库
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=get_requirements(),

    # 包含数据文件
    include_package_data=True,
    package_data={
        "ai_scheduler": [
            "sample_config/*.json",
        ],
        "NNScheduler": [
            "interface/*.json",
        ],
    },

    # 命令行入口点
    entry_points={
        "console_scripts": [
            "ai-scheduler=ai_scheduler.cli:main",
        ],
    },

    # 项目URLs
    project_urls={
        "Bug Reports": "https://github.com/your-org/ai-scheduler/issues",
        "Source": "https://github.com/your-org/ai-scheduler",
        "Documentation": "https://github.com/your-org/ai-scheduler/wiki",
    },

    # 关键词
    keywords="ai, scheduler, neural-network, optimization, npu, dsp, task-scheduling",

    # ZIP安全标识
    zip_safe=False,
)
