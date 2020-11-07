import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mdstuff-mvondomaros",
    version="0.1",
    author="Michael von Domaros",
    author_email="mvondomaros@gmail.com",
    description="A collection of tools needed for the analysis of my MD simulations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mvondomaros/mdstuff",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.0",
)
