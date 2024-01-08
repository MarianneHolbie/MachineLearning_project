#!/usr/bin/env python3
"""
    Function training operation
"""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
        Method that creates the training operation for NN
    :param loss: loss of NN's prediction
    :param alpha: learning rate

    :return: operation that trains NN using gradient descent
    """

    # optimizer gradient descent
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=alpha,
        name='GradientDescent'
    )

    # train
    train = optimizer.minimize(loss)

    return train
