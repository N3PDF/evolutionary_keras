""" Test the utilities of evolutionary_keras """

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input

from evolutionary_keras.utilities import get_number_nodes, parse_eval
from evolutionary_keras.models import EvolModel


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

    nodes = 10

    # Tensorflow won't build the layer until it is called in a model
    input_layer = Input(shape=(1,))
    output_layer = Dense(units=nodes, name="test_layer")
    modelito = EvolModel(input_layer, output_layer(input_layer))
    
    # Check that indeed the number of nodes is parsed correctly
    assert nodes == get_number_nodes(output_layer)
