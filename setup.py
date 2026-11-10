"""Shim for editable installs with older pip versions."""
from setuptools import find_packages, setup

setup(packages=find_packages(include=["ezero*", "baselines*", "benchmarks*"]))
