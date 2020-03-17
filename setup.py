from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="model-zoo",
    version="0.0.1",
    author="Samuel Stanton",
    author_email="ss13641@nyu.edu",
    description="Out-of-the-box probabilistic regression models in PyTorch",
    url="https://github.com/samuelstanton/model-zoo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
    install_requires=[
        'gpytorch @ git+https://github.com/cornellius-gp/gpytorch#egg=gpytorch-1.0.1'
    ]
)
