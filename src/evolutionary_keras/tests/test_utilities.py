""" Test the utilities of evolutionary_keras """

import numpy as np
from keras import backend as K
from keras.layers import Dense
from evolutionary_keras.utilities import parse_eval, get_number_nodes


def test_parse_eval():
    """ Test the parse_eval function, which should output a float
    when it gets a list or a float """
    # There are two situations, get a list or get a float
    float_mode = 3.0
    list_mode = [float_mode]
    assert float_mode == parse_eval(list_mode)
    assert float_mode == parse_eval(float_mode)


def test_get_number_nodes():
    """ Get the number of nodes of a layer """
    ii = K.constant(np.random.rand(2, 1))
    nodes = 10
    layer = Dense(units=nodes)
    # Keras won't build the layer until it is called with some input
    _ = layer(ii)
    # Check that indeed the number of nodes is parsed correctly
    assert nodes == get_number_nodes(layer)
