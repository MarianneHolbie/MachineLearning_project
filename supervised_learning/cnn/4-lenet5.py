#!/usr/bin/env python3
"""
    LeNet-5 (Tensorflow 1)
"""

import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
        function that builds a modified version of the LeNet-5
        architecture using TensorFlow

        :param x: tf.placeholder, shape(m,28,28,1) input images
        :param y: tf.placeholder, shape(m,10) one-hot labels

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

        :return: * tensor for softmax activated output
                 * training operation that utilizes Adam opt.
                 * tensor for loss of network
                 * tensor for accuracy of the network
    """

    # set initialization to He et. al
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # build layer
    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=5,
                             padding='same',
                             kernel_initializer=initializer,
                             activation='relu')(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=2,
                                   strides=2)(conv1)
    conv2 = tf.layers.Conv2D(filters=16,
                             kernel_size=5,
                             padding='valid',
                             kernel_initializer=initializer,
                             activation='relu')(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=2,
                                   strides=2)(conv2)
    # flatten layers to convert tensor multidim in vector unidirectional
    flat = tf.layers.Flatten()(pool2)
    full1 = tf.layers.Dense(120,
                            activation='relu',
                            kernel_initializer=initializer)(flat)
    full2 = tf.layers.Dense(84,
                            activation='relu',
                            kernel_initializer=initializer)(full1)
    output = tf.layers.Dense(10,
                             activation=None,
                             kernel_initializer=initializer)(full2)
    softmax = tf.nn.softmax(output)

    # calculate loss
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=output)

    # Adam optimizer
    train_Adam = tf.train.AdamOptimizer().minimize(loss)

    # comparison of indice's max value for y and logits
    y_pred = tf.argmax(output, axis=1)
    y_true = tf.argmax(y, axis=1)
    correct_prediction = tf.equal(y_pred, y_true)

    # convert tensor boll in float32
    correct_prediction = tf.cast(correct_prediction, dtype=tf.float32)

    # define accuracy
    accuracy = tf.reduce_mean(correct_prediction)

    return softmax, train_Adam, loss, accuracy
