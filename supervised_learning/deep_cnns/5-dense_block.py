#!/usr/bin/env python3
"""
    Dense Block
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
        build a dense block as described in
        'Densely Connected Convolutional Networks'

    :param X: output prev layer
    :param nb_filters: int, number filters in X
    :param growth_rate: growth rate for the dense block
    :param layers: number of layers in dense block

    :return: concatenated output of each layer in Dense Block
            & number of filters within the concatenated outputs
    """
    # define initializer, variable
    initializer = K.initializers.HeNormal()
    input_x = X
    nbr_filter_output = nb_filters

    # loop for Dense block
    for layer in range(layers):
        # bottleneck layer
        batchN1 = K.layers.BatchNormalization()(input_x)
        relu1 = K.layers.Activation(activation='relu')(batchN1)
        conv1 = K.layers.Conv2D(filters=4 * growth_rate,
                                kernel_initializer=initializer,
                                kernel_size=(1, 1))(relu1)

        # conv layer
        batchN2 = K.layers.BatchNormalization()(conv1)
        relu2 = K.layers.Activation(activation='relu')(batchN2)
        conv2 = K.layers.Conv2D(filters=growth_rate,
                                kernel_initializer=initializer,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same')(relu2)

        input_x = K.layers.Concatenate()([input_x, conv2])

        nbr_filter_output += growth_rate

    return input_x, nbr_filter_output
