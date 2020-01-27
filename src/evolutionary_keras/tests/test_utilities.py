""" Test the utilities of evolutionary_keras """

import numpy as np
from keras import backend as K
from keras.layers import Dense
from evolutionary_keras.utilities import parse_eval, get_number_nodes

def test_parse_eval():
    # There are two situations, get a list or get a float
    float_mode = 3.0
    list_mode = [float_mode]
    assert float_mode == parse_eval(list_model)
    assert float_mode == parse_eval(float_mode)

def test_get_number_nodes():
    ii = K.constant(np.random.rand(10,1))
    nodes = 15
    layer = Dense(units = nodes)
    _ = layer(ii)
    # Check that indeed the number of nodes is parsed correctly
    assert nodes == get_number_nodes(layer)
