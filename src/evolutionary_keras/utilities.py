"""
    Module including some useful functions
"""
from tensorflow.keras import backend as K


def get_number_nodes(layer):
    """ Given a keras layer, outputs the number of nodes """
    nodes = 0
    # It is necessary to check whether the layer is trainable
    # AND whether it has any weights which is trainable
    if layer.trainable and layer.trainable_weights:
        # The first dimension is always the batch size so
        # this is a good proxy for the number of nodes
        output_nodes = layer.get_output_shape_at(0)[1:]
        nodes = sum(output_nodes)
    return nodes


def compatibility_numpy(weight):
    """ Wrapper in case the evaluated keras object doesn't have a numpy() method """
    try:
        result = weight.numpy()
    except NotImplementedError:
        result = K.eval(weight.read_value())
    return result
