#!/usr/bin/env python3
"""
    ResNet-50
"""

import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
        Builds the ResNet-50 architecture as described in
        'Deep Residual Learning for Image Recognition' 2015

    :return: Keras Model
    """

    # define input & initializer
    inputs = K.layers.Input(shape=(224, 224, 3))
    initializer = K.initializers.HeNormal()

    # conv1
    conv1 = K.layers.Conv2D(64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=initializer)(inputs)
    batchN1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation(activation='relu')(batchN1)

    # conv2_x
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(relu1)
    p0 = projection_block(pool1, [64, 64, 256], s=1)
    id2 = identity_block(p0, [64, 64, 256])
    id3 = identity_block(id2, [64, 64, 256])

    # conv3_x
    p1 = projection_block(id3, [128, 128, 512], s=2)
    id4 = identity_block(p1, [128, 128, 512])
    id5 = identity_block(id4, [128, 128, 512])
    id6 = identity_block(id5, [128, 128, 512])

    # conv4_x
    p2 = projection_block(id6, [256, 256, 1024], s=2)
    id7 = identity_block(p2, [256, 256, 1024])
    id8 = identity_block(id7, [256, 256, 1024])
    id9 = identity_block(id8, [256, 256, 1024])
    id10 = identity_block(id9, [256, 256, 1024])
    id11 = identity_block(id10, [256, 256, 1024])

    # conv5_x
    p3 = projection_block(id11, [512, 512, 2048], s=2)
    id12 = identity_block(p3, [512, 512, 2048])
    id13 = identity_block(id12, [512, 512, 2048])

    # Avr pool
    pool2 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      strides=(1, 1),
                                      padding='valid')(id13)

    # Fully connected
    output = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer=initializer)(pool2)

    model = K.models.Model(inputs, output)

    return model
