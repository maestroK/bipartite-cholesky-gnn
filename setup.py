from setuptools import setup, find_packages

setup(
    name="bcgnn",
    version="0.1.0",
    description=(
        "Bipartite Cholesky Graph Networks for Many-Body Quantum Chemistry"
    ),
    author="Abdul Samad Khan",
    author_email="24120006@lums.edu.pk",
    url="https://arxiv.org/abs/XXXX.XXXXX",  # update after arXiv submission
    packages=find_packages(exclude=["tests*", "scripts*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=1.5",
        "scipy>=1.10",
        "pyscf>=2.3",
        "torch>=2.0",
        "scikit-learn>=1.2",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": ["pytest>=7.3", "nbformat>=5.7"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
