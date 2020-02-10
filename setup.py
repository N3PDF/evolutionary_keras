from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="evolutionary_keras",
    version="1.0.1",
    author="S. Carrazza, J. Cruz-Martinez, Roy Stegeman",
    author_email="juan.cruz@mi.infn.it, roy.stegeman@mi.infn.it",
    url="https://github.com/N3PDF/evolutionary_keras",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["numpy", "keras", "sphinx_rtd_theme", "recommonmark",],
    python_requires=">=3.6",
    descriptions="An evolutionary algorithm implementation for Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
