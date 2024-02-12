#!/usr/bin/env python3
"""
    Projection Block
"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
        build an projection block as described in
        'Deep Residual Learning for Image Recognition' (2015)

    :param A_prev: output from previous layer
    :param filters: tuple or list containing
        * F11 : number of filters in first 1x1 conv
        * F3: number of filters in 3x3 conv
        * F12: number of filters in second 1x1 conv and in shortcut 1x1 conv
    :param s: stride of the first conv in both main path and shortcut connexion
    Each conv layer followed by batch normalization along channels axis
    and ReLu
    He Normal initialization

    :return: activated output of the identity block
    """
    # extract filter
    F11, F3, F12 = filters
    stride = s

    # initializer
    initializer = K.initializers.HeNormal()

    # First layer
    conv1 = K.layers.Conv2D(F11,
                            kernel_size=(1, 1),
                            kernel_initializer=initializer,
                            strides=s,
                            padding='same')(A_prev)
    batchN1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation(activation='relu')(batchN1)

    # second layer
    conv2 = K.layers.Conv2D(F3,
                            kernel_initializer=initializer,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same')(relu1)
    batchN2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation(activation='relu')(batchN2)

    # third layer
    conv3 = K.layers.Conv2D(F12,
                            kernel_initializer=initializer,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same')(relu2)
    batchN3 = K.layers.BatchNormalization(axis=3)(conv3)

    # shortcut way
    shortcut_input = K.layers.Conv2D(F12,
                                     kernel_initializer=initializer,
                                     kernel_size=(1, 1),
                                     strides=s,
                                     padding='same')(A_prev)
    batchN4 = K.layers.BatchNormalization(axis=3)(shortcut_input)

    # add ResNet
    resnet = K.layers.add([batchN3, batchN4])

    # ReLU
    output = K.layers.Activation(activation='relu')(resnet)

    return output
