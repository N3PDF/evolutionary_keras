""" Tests for checking that we can compile and run with all the included optimziers """

import numpy as np
import keras.backend as K
from keras.layers import Input, Dense

from evolutionary_keras.models import EvolModel
from evolutionary_keras.optimizers import NGA

def generate_model(ishape = 2, hshape = 4, oshape = 1):
    """ Generates a model with EvolModel """
    # Prepare some input tensor dimension 2
    input_layer = Input(shape = (ishape,))
    hidden_layer = Dense(units = hshape, activation = "sigmoid")
    output_layer = Dense(units = oshape, activation = "sigmoid")
    modelito = EvolModel(input_layer, output_layer(hidden_layer(input_layer)))
    return modelito

def test_get_shape():
    optimizer = NGA(mutation_rate = 1.0)
    modelito = generate_model(ishape = 2, hshape = 4, oshape = 1)
    modelito.compile(optimizer = optimizer, loss="mse")
    # Now let us create some random input and output and let it run for
    # 100 epochs. Check that it does indeed gets the loss down
    xin = np.ones((100,2))
    yout = np.zeros((100,1))
    start_loss = modelito.evaluate(x = xin, y = yout)
    _ = modelito.fit(x = xin, y = yout, epochs = 50)
    final_loss = modelito.evaluate(x = xin, y = yout)
    assert final_loss < start_loss
