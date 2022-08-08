from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='multimodel',
    version='0.1',
    packages=['multimodel'],
    description='A package that integrates with scikit-learn and similar models to easily enable multi-tiered model networks in which each model can target different classes and have different hyperparameters and feature inputs.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjkleiman/multimodel",
    project_urls={
        "Issue Tracker": "https://github.com/mjkleiman/multimodel/issues"
    },
    author='Michael J Kleiman',
    author_email='michael@kleiman.me',
    license='BSD-3',
    include_package_data=True
)
