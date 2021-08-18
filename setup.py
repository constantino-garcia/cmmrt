from setuptools import find_packages, setup

setup(
    name='cmmrt',
    package_dir={'':'cmmrt'},
    packages=find_packages('cmmrt'),
    version='0.1.0',
    description='Models for prediction of retention times from molecular properties',
    author='CEU',
    license='MIT',
)
