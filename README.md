![](https://github.com/N3PDF/evolutionary_keras/workflows/pytest/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/evolutionary-keras/badge/?version=latest)](https://evolutionary-keras.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3630399.svg)](https://doi.org/10.5281/zenodo.3630399)



# evolutionary_keras

Keras is one of the most widely used Machine Learning frameworks available in the market. It is a high-level API written in Python and that can run on mulitple backends. Their goal is to be able to build and test new model as fast as possible.

Keras models are trained through the usage of optimizers, all of which are Gradient Descent based. This module deals with that shortcoming of Keras by implementing several Evolutionary Algorithms on top of Keras while keeping the main philosophy of the project: it must be easy to prototype.

The default project library now provides support for:
- Nodal Genetical Algorithm (NGA)
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
