from setuptools import find_packages, setup

__version__ = "0.0.1"


with open("README.md", encoding="utf-8") as f:
    _long_description = f.read()

setup(
    name="bayesian_safety_validation",
    version=__version__,
    description="Estimate failure probability for  binary-valued black-box system",
    long_description=_long_description,
    long_description_content_type="text/markdown",
    author="Loris Kong",
    author_email="imloriskong@gmail.com",
    python_requires=">=3.9",  # TODO: 3.10 is perfered.
    entry_points={"console_scripts": ["adt-sim=adt_sim.cli.cli:adt_sim"]},
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.8.4",
        "numpy>=1.26.1",
        "pytest>=7.4.4",
        "typing_extensions>=4.10.0",
        "bayesian-optimization>=1.4.3",
        "scikit-learn>=1.4.0",
        "scipy>=1.13.0",
    ],
    url="https://github.com/loriskong/BayesianSafetyValidation",
    license="MIT License",
)
