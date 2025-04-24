from setuptools import setup, find_packages

setup(
    name="recognition",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "recognition=recognition.cli:parse_and_execute",
        ],
    },
    install_requires=[
        "numpy<2.0.0",
        "MDAnalysis==2.4.3",
        "gsd<1.9.3",
        "pandas",
        "scipy",
        "rdkit",
        "networkx",
        "matplotlib",
        "numba<0.60.0"
    ],
)

