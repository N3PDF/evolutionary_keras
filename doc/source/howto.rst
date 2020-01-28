.. _howto-label:

==========
How to use
==========

In order to use the capabilities of ``evolutionary_keras`` in a project is necessary to use the model-classes provided. These classes inherit from the Keras `model class <https://keras.io/models/about-keras-models/>`_ and transparently defer to them whenever a Gradient Descent algorithm is used.

As an example, let us consider a project in which we have some neural network constructed with an input layer ``input_layer`` and an output layer ``output_layer``. The Keras model would usually be constructed as:

.. code-block:: python

    from keras.models import Model
    my_model = Model(input_layer, output_layer)

Using ``evolutionary_keras`` is as easy as doing:

.. code-block:: python

    from evolutionary_keras.models import EvolModel
    my_model = EvolModel(input_layer, output_layer)

From that point onwards ``my_model`` behaves exactly as a normal Keras model implementing the same methods and attributes as well as allowing the usage of Evolutionary :ref:`optimizers-label`.
For instance, the example belows utilizes the Nodal Genetic Algorithm (NGA):

.. code-block:: python

    my_model.compile("nga")

Which will use the default parameters of the :ref:`nga-label`. Subsequent calls to methods such as ``my_model.fit`` will use the NGA algorithm to train.

For a more fine-grained usage we can also import the optimizer and instantiate it ourselves:

.. code-block:: python

    from evolutionary_keras.optimizers import NGA
    my_nga = NGA(population_size = 42, mutation_rate = 0.2)
    my_model.compile(my_nga)
