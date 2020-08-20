evolutionary_keras's documentation!
===================================

This is the documentation for the latest release of the ``evolutionary_keras`` python module.
This documentation includes a quick how-to use guide, provides several examples and collects changelogs of new releases.

==============================
What is ``evolutionary_keras``
==============================
`Tensorflow <https://www.tensorflow.org/>`_ and its `Keras <https://keras.io/>`_ API are some of the most widely used Machine Learning frameworks available in the market. It is a high-level API written in Python and that can run on multiple backends.
The goal of Keras is to be able to build and test new TensorFlow models as fast as possible.

Keras models are trained through the usage of `optimizers <https://keras.io/optimizers/>`_, all of which are Gradient Descent based.
This module deals with that shortcoming of Keras implementing several Genetic Algorithms on top of Keras while keeping the main philosophy of the project: it must be easy to prototype.

==============================
Installing ``evolutionary_keras``
==============================
``evolutionary_keras`` is available in PyPI, conda-forge.

.. code-block:: bash
    
    pip install evolutionary_keras

Furthermore, the code is available under `GPL3.0 <https://github.com/N3PDF/evolutionary_keras/blob/master/LICENSE>`_ in github: `N3PDF/evolutionary_keras <https://github.com/N3PDF/evolutionary_keras>`_.

.. toctree::
   :maxdepth: 4
   :caption: Contents:
   
   howto
   optimizers

.. automodule:: evolutionary_keras
    :members:
    :noindex:




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
