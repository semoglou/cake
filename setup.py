from setuptools import setup, find_packages

setup(
    name="cake-ensemble",
    version="0.1.3",
    author="Aggelos Semoglou",
    author_email="a.semoglou@outlook.com",
    description="CAKE: Confidence in Assignments via K-partition Ensembles",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/semoglou/cake",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "scipy>=1.8",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT"
)
