from setuptools import find_packages, setup

setup(
    name="tutorial",
    packages=find_packages(exclude=["tutorial_tests"]),
    install_requires=[
        "matplotlib",
        "pandas",
    ],
    extras_require={"dev": ["flake8", "pytest"]},
)
