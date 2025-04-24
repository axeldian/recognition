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
        "MDAnalysis<2.6.0",
        "pandas",
        "numpy<2.0.0",
        "scipy",
        "rdkit",
        "os",
        "pickle",
        "networkx",
        "matplotlib",
        "itertools",
        "collections",
        "numba"
    ],
)

