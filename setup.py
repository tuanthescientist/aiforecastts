from setuptools import setup, find_packages

setup(
    name="aiforecastts",
    version="0.1.0",
    packages=find_packages(where="ts_library"),
    package_dir={"": "ts_library"},
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "statsmodels>=0.14",
    ],
)
