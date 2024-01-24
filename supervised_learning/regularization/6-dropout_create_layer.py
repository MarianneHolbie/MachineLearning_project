#!/usr/bin/env python3
"""
    Create layer with Dropout regularization
"""

import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
        function that creates layer of NN using dropout

        :param prev: tensor, output prev layer
        :param n: number of nodes new layer
        :param activation: activation function on the new layer
        :param keep_prob: proba that node will be kept

        :return: output of the new layer
    """

    # define layer Dropout
    dropout_layer = tf.compat.v1.layers.Dropout(rate=keep_prob)

    # apply dropout
    new_layer = (
        tf.layers.Dense(n,
                        activation=activation,
                        name="layer"))
    output = dropout_layer(new_layer(prev))

    return output
