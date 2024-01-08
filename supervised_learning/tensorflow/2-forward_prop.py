#!/usr/bin/env python3
"""
    Function Forward propagation
"""

import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
        Method to creates the forward propagation graph for NN

        :param x: placeholder for input data
        :param layer_sizes: list number of nodes in each layer of NN
        :param activations: list activation function for each layer

        :return: the prediction of NN in tensor form
    """
    for i in range(len(layer_sizes)):
        new_layer = create_layer(x, layer_sizes[i], activations[i])
        x = new_layer

    return new_layer
