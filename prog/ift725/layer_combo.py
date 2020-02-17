# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

from ift725.layers import *
from ift725.quick_layers import *


def forward_fully_connected_transform_relu(x, w, b):
    """
    Convenience layer that performs a FC transform followed by a ReLU

    Inputs:
    - x: Input to the FC layer
    - w, b: Weights for the FC layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = forward_fully_connected(x, w, b)
    out, relu_cache = forward_relu(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def backward_fully_connected_transform_relu(dout, cache):
    """
    Backward pass for the FC-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = backward_relu(dout, relu_cache)
    dx, dw, db = backward_fully_connected(da, fc_cache)
    return dx, dw, db


def forward_convolutional_relu(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = forward_relu(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def backward_convolutional_relu(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = backward_relu(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def forward_convolutional_relu_pool(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    out = None
    cache = None

    #############################################################################
    # TODO: Implémentez une opération combo constituée d'un conv-relu-maxpool.  #
    #  Stockez le résultat dans out et cache.                                   #
    #############################################################################

    #############################################################################
    # Fin de votre code                                                         #
    #############################################################################
    return out, cache


def backward_convolutional_relu_pool(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    dx = None
    dw = None
    db = None

    #############################################################################
    # TODO: Implémentez la rétro-propagation d'un conv-relu-maxpool.            #
    #  Stockez le résultat dans dx, dw et db                                    #
    #############################################################################

    #############################################################################
    # Fin de votre code                                                         #
    #############################################################################
    return dx, dw, db
