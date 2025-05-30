from setuptools import setup, find_packages

setup(
    name="riskneutral",
    version="0.1.0",
    description="Risk-Neutral Density Estimation Tools",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Moe-Dada/risk_neutral",
    author="Moses Dada, CQF, Ph.D",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
    ],
)
