"""
    Module including some useful functions
"""
from tensorflow.keras import backend as K
from tensorflow.keras import Model


def get_number_nodes(model):
    """ Given a keras layer, outputs the number of nodes """
    nodes = 0
    for layer in model.layers:
        # It is necessary to check whether the layer is trainable
        # AND whether it has any weights which is trainable
        if layer.trainable and layer.trainable_weights:
            if isinstance(layer, Model):
                nodes += get_number_nodes(layer)
            elif hasattr(layer, "bias"):
                nodes += layer.bias.shape[0]
            else:
                raise ValueError(
                    "The NGA optimizer can only be applied to model layers with 'bias' attribute"
                )
    return nodes


def compatibility_numpy(weight):
    """ Wrapper in case the evaluated keras object doesn't have a numpy() method """
    try:
        result = weight.numpy()
    except NotImplementedError:
        result = K.eval(weight.read_value())
    return result
