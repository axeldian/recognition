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
        "MDAnalysis",
        "pandas",
        "numpy",
    ],
)

