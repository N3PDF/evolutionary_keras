"""
    Module including some useful functions
"""

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


def parse_eval(loss):
    """ Parses the result from `model.evaluate`, which sometimes
    comes as a list and sometimes comes as one single float
    Returns
    -------
        `loss` : loss as a float
    """
    try:
        loss = loss[0]
    except TypeError as e:
        # If the output was a number then it is ok
        if not isinstance(loss, (float, int)):
            raise e
    return loss
