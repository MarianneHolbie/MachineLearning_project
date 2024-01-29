#!/usr/bin/env python3
"""
    Input
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
        function that builds a neural network with the Keras library

        :param nx: number of input features to the network
        :param layers: list, number nodes in each layer
        :param activations: list, activation functions for each layer
        :param lambtha: L2 regularization parameter
        :param keep_prob: proba node kept for dropout

        :return: keras model
    """
    inputs = K.Input(shape=(nx,))

    x = inputs
    for i in range(len(layers)):
        # add Dense layer
        x = K.layers.Dense(layers[i],
                           activation=activations[i],
                           kernel_regularizer=K.regularizers.L2(lambtha))(x)

        # apply Dropout except last layer
        if i != len(layers) - 1 and keep_prob is not None:
            x = K.layers.Dropout(1 - keep_prob)(x)

    # create model
    model = K.Model(inputs, x)

    return model
