.. _optimizers-label:

==========
Optimizers
==========

This page lists all evolutionary strategies currently implemented

.. _nga-label:

Nodal Genetic Algorithm (NGA)
-----------------------------
Implementation of the Nodal Genetic Algorithm as implemented by the NNPDF collaboration and which is presented in the `NNPDF3.0 release paper <https://link.springer.com/article/10.1007%2FJHEP04%282015%29040>`_.

.. autoclass:: evolutionary_keras.optimizers.NGA

.. _cma-label:

Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
--------------------------------------------------------
Implementation of the Covariance Matrix Adaptation Evolution Strategy as developed by Nokolaus Hansen et al `[ref] <https://zenodo.org/record/3764210>`_ as `pycma <https://github.com/CMA-ES/pycma>`_.

.. autoclass:: evolutionary_keras.optimizers.CMA