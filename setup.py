from setuptools import find_packages, setup

setup(
    name="fleetperformancekpi",
    packages=find_packages(exclude=["fleetperformancekpi_tests"]),
    install_requires=[
        "matplotlib",
        "pandas",
    ],
    extras_require={"dev": ["flake8", "pytest"]},
)
