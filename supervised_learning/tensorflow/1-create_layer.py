#!/usr/bin/env python3
"""
    Function Create Layer
"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
        Method to create layer

        :param prev: tensor output of previous layer
        :param n: number of nodes in the layer to create
        :param activation: activation function layer should use

        :return: tensor output of the layer
    """

    # set initialization to He et. al
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # create layer Dense with paramaters
    new_layer = tf.layers.Dense(n,
                                activation=activation,
                                kernel_initializer=initializer,
                                name="layer")

    # apply layer to input
    output = new_layer(prev)

    return output
