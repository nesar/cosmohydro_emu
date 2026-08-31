"""
Setup script for the cosmohydro_emu package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# Read version from __init__.py
def get_version():
    init_file = os.path.join("cosmohydro_emu", "__init__.py")
    with open(init_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip("'\"")
    return "0.1.0"


setup(
    name="cosmohydro_emu",
    version=get_version(),
    author="Nesar Ramachandra",
    author_email="",
    description="Multi-redshift Gaussian Process emulators for the CRK-HACC CosmoHydro simulation suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nesar/cosmohydro_emu",
    packages=find_packages(exclude=("tests", "tests.*", "scripts", "examples", "docs")),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9,<3.12",
    install_requires=[
        "numpy>=1.21.0,<1.24",
        "scipy>=1.7.0,<1.11",
        "sepia @ git+https://github.com/lanl/SEPIA.git",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "matplotlib>=3.5.0,<3.9",
            "pandas>=1.3.0,<2.0",
        ],
        "plotting": [
            "matplotlib>=3.5.0,<3.9",
            "pandas>=1.3.0,<2.0",
        ],
    },
    package_data={
        "cosmohydro_emu": [
            "models/*/*.pkl",
            "data/*.npz",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="cosmology emulator gaussian-process hydrodynamics simulations baryonic-feedback",
    project_urls={
        "Bug Reports": "https://github.com/nesar/cosmohydro_emu/issues",
        "Source": "https://github.com/nesar/cosmohydro_emu",
    },
)
