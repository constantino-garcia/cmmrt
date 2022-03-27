from setuptools import find_packages, setup

setup(
    name='cmmrt',
    packages=find_packages(),
    version='0.2.0',
    description='Models for prediction and projection of retention times from molecular descriptors and fingerprints',
    author='CEU',
    license='MIT',
    install_requires=[
        "numpy>=1.19.2",
        "optuna>=2.8.0",
        "catboost>=0.25.1",
        "torch>=1.7.1",
        "pandas>=1.2.4",
        "xgboost>=1.3.3",
        "gdown>=3.13.0",
        "gpytorch>=1.4.2",
        "lightgbm>=3.2.1",
        "scikit_learn>=0.24.2",
        "matplotlib>=3.3.3"
    ],
    package_data={"cmmrt": ["data/*.pt"]}
)
