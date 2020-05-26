from setuptools import setup, find_packages
import os
import re


requirements = ['numpy', 'cma']
if os.environ.get('READTHEDOCS') != 'True':
    requirements.append('tensorflow>2.1*')
PACKAGE = 'evolutionary_keras'

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

def get_version():
    """ Gets the version from the package's __init__ file
    if there is some problem, let it happily fail """
    VERSIONFILE = os.path.join('src', PACKAGE, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)

setup(
    name="evolutionary_keras",
    version=get_version(),
    author="S. Carrazza, J. Cruz-Martinez, Roy Stegeman",
    author_email="juan.cruz@mi.infn.it, roy.stegeman@mi.infn.it",
    url="https://github.com/N3PDF/evolutionary_keras",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    install_requires=requirements,
    extras_require={
        'docs' : [
        'sphinx_rtd_theme',
        'recommonmark',
        ],
    },
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
