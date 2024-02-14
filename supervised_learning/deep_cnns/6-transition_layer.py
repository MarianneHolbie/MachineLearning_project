#!/usr/bin/env python3
"""
    Transition layer
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
        builds a transition layer as described in
        'Densely Connected Convolutional Networks'

        :param X: output from prev layer
        :param nb_filters: int, number of filters in X
        :param compression: factor for the transition layer
        Compression as uses in DenseNet-C
        He normal initialization
        convolutions preceded by BatchNorm and rectified by ReLU

        :return: output transition layer
        and number of filters withing the output
    """
    # initialization
    initializer = K.initializers.HeNormal()

    # define number of filter
    num_filters = nb_filters * compression

    batchN = K.layers.BatchNormalization()(X)
    relu = K.layers.Activation(activation='relu')(batchN)
    conv = K.layers.Conv2D(num_filters,
                           kernel_initializer=initializer,
                           kernel_size=1)(relu)
    output = K.layers.AveragePooling2D(strides=(2, 2))(conv)

    return output, num_filters
