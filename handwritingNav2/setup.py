#!/usr/bin/env python3
"""
HandWriting Navigation - Setup Script
"""

from setuptools import setup, find_packages
import os

# 读取README作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements
def read_requirements():
    """读取requirements.txt"""
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if line and not line.startswith("#"):
                    # 移除注释的包
                    if not line.startswith("# "):
                        requirements.append(line)
    return requirements

setup(
    name="handwriting-nav",
    version="0.1.0",
    author="HandWriting Nav Team",
    description="基于扩散策略的手绘地图导航系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/handwritingNav2",
    packages=find_packages(include=['modeling', 'modeling.*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hwnav-train=scripts.train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

