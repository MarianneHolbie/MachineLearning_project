#!/usr/bin/env python3
"""
    Inception Block
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
        builds an inception block as described in
        'Going Deeper with Convolutions'

        :param A_prev: output from the previons layer
        :param filters: tuple or list containing
            F1 number of filters in the 1x1 convolution
            F3R number of filters in the 1x1 convolution before 3x3 convolution
            F3 number of filters in the 3x3 convolution
            F5R number of filters in the 1x1 convolution before 5x5 convolution
            F5 number of filters in the 5x5 convolution
            FPP number of filters in the 1x1 convolution after the max pooling
        All convolutions inside : linear activation (ReLU)

        :return: concatenated output of the inception block
    """
    # filters extraction
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 conv
    layer1 = K.layers.Conv2D(filters=F1,
                             kernel_size=(1, 1),
                             activation='relu',
                             padding='same')(A_prev)

    # 1x1 + 3x3 conv
    layer2 = K.layers.Conv2D(filters=F3R,
                             kernel_size=(1, 1),
                             activation='relu',
                             padding='same')(A_prev)
    layer2 = K.layers.Conv2D(filters=F3,
                             kernel_size=(3, 3),
                             activation='relu',
                             padding='same')(layer2)

    # 1x1 + 5x5 conv
    layer3 = K.layers.Conv2D(filters=F5R,
                             kernel_size=(1, 1),
                             activation='relu',
                             padding='same')(A_prev)
    layer3 = K.layers.Conv2D(filters=F5,
                             kernel_size=(5, 5),
                             activation='relu',
                             padding='same')(layer3)

    # 1x1 + MaxPooling
    # adjust pool_size and strides to keeps the same dimensions as the entrance
    layer4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                   strides=(1, 1),
                                   padding='same')(A_prev)
    layer4 = K.layers.Conv2D(filters=FPP,
                             kernel_size=(1, 1),
                             activation='relu',
                             padding='same')(layer4)

    # concatenate
    model = K.layers.concatenate([layer1, layer2, layer3, layer4])

    return model
