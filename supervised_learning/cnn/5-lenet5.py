#!/usr/bin/env python3
"""
    LeNet-5 (Keras)
"""

import tensorflow.keras as K


def lenet5(X):
    """
        function that builds a modified version of the LeNet-5
        network architecture using keras

        :param X: K.input, shape(m,28,28,1) input images

        Model:
            * Convolutional layer with 6 kernels of shape 5x5 with same padding
            * Max pooling layer with kernels of shape 2x2 with 2x2 strides
            * Convolutional layer with 16 kernels, shape 5x5 with valid padding
            * Max pooling layer with kernels of shape 2x2 with 2x2 strides
            * Fully connected layer with 120 nodes
            * Fully connected layer with 84 nodes
            * Fully connected softmax output layer with 10 nodes
        he_normal initialization
        activation : relu

        :return: K.Model compiled to use Adam optmization and accuracy metrics
    """

    # intializer
    initializer = K.initializers.HeNormal()

    model = K.Sequential([
        K.layers.Conv2D(filters=6,
                        kernel_size=5,
                        padding='same',
                        kernel_initializer=initializer,
                        activation='relu'),
        K.layers.MaxPooling2D(pool_size=2,
                              strides=2),
        K.layers.Conv2D(filters=16,
                        kernel_size=5,
                        padding='valid',
                        kernel_initializer=initializer,
                        activation='relu'),
        K.layers.MaxPooling2D(pool_size=2,
                              strides=2),
        K.layers.Flatten(),
        K.layers.Dense(120,
                       activation='relu',
                       kernel_initializer=initializer),
        K.layers.Dense(84,
                       activation='relu',
                       kernel_initializer=initializer),
        K.layers.Dense(10,
                       kernel_initializer=initializer,
                       activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])

    return model
