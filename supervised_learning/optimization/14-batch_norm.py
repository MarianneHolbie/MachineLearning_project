#!/usr/bin/env python3
"""
   Batch Normalization upgraded
"""

import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
        Method that creates a batch normalization layer for a
        NN in tf

        :param prev: activated output of the previous layer
        :param n: number of nodes in the layer to be created
        :param activation: activation function for output layer

        :return: tensor of activated output for the layer
    """
    # set initialization to He et. al
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # create layer Dense with parameters
    new_layer = tf.layers.Dense(n,
                                activation=activation,
                                kernel_initializer=initializer,
                                name="layer")

    # apply layer to input
    x = new_layer(prev)

    # beta and gamma : two trainable parameters
    beta = tf.compat.v1.zeros_initializer()
    gamma = tf.compat.v1.ones_initializer()

    epsilon = 1e-8

    x_norm = tf.compat.v1.layers.batch_normalization(
        inputs=x,
        beta_initializer=beta,
        gamma_initializer=gamma,
        epsilon=epsilon)

    return x_norm
