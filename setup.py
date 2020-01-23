import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="model-zoo",
    version="0.0.1",
    author="Samuel Stanton",
    author_email="ss13641@nyu.edu",
    description="Out-of-the-box probabilistic regression models in PyTorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)