.. title::
   evolutionary_keras documentation

================================================================
Evolutionary Keras: evolutionary strategies for Tensorflow-Keras
================================================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3630339.svg
   :target: https://doi.org/10.5281/zenodo.3630339

.. contents::
   :local:

This is the documentation for the latest release of the ``evolutionary_keras`` python module.
This documentation includes a quick how-to use guide, provides several examples and collects changelogs of new releases.

What is ``evolutionary_keras``
==============================
`Tensorflow <https://www.tensorflow.org/>`_ and its `Keras <https://keras.io/>`_ API are some of the most widely used Machine Learning frameworks available in the market. It is a high-level API written in Python and that can run on multiple backends.
The goal of Keras is to be able to build and test new TensorFlow models as fast as possible.

Keras models are trained through the usage of `optimizers <https://keras.io/optimizers/>`_, all of which are Gradient Descent based.
This module deals with that shortcoming of Keras implementing several Genetic Algorithms on top of Keras while keeping the main philosophy of the project: it must be easy to prototype.

Installing ``evolutionary_keras``
=================================
``evolutionary_keras`` is available in PyPI, conda-forge.

.. code-block:: bash
    
    pip install evolutionary_keras

Furthermore, the code is available under `GPL3.0 <https://github.com/N3PDF/evolutionary_keras/blob/master/LICENSE>`_ in github: `N3PDF/evolutionary_keras <https://github.com/N3PDF/evolutionary_keras>`_.


How to cite ``evolutionary_keras``?
===================================

When using ``evolutionary_keras`` in your research please cite the following zenodo publication.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3630339.svg
   :target: https://doi.org/10.5281/zenodo.3630339
   
.. code-block:: latex

  @software{evolkeras_package,
    author       = {Juan Cruz-Martinez and
                    Roy Stegeman and
                    Stefano Carrazza},
    title        = {evolutionary\_keras: a Genetic Algorithm library},
    month        = jan,
    year         = 2020,
    publisher    = {Zenodo},
    version      = {v0.9b2},
    doi          = {10.5281/zenodo.3630339},
    url          = {https://doi.org/10.5281/zenodo.3630339}
  }







Indices and tables
==================

.. toctree::
   :maxdepth: 4
   :glob:
   :caption: Contents:
   
   optimizers
   howto
   apisrc/evolutionary_keras

   
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
