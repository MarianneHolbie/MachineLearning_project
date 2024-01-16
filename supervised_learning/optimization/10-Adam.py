#!/usr/bin/env python3
"""
   Adam upgraded
"""

import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
        Method that creates training op for NN
        in tf using Adam optimization algo

        :param loss: loss of NN
        :param alpha: learning rate
        :param beta1: weight used for firs moment
        :param beta2: weight used for second moment
        :param epsilon: small number to avoid division by 0

        :return: Adam optimization operation
    """
    # set optimizer that implement Adam algo in tf
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)

    # train_op to minimize loss with this optimizer
    train_op = optimizer.minimize(loss)

    return train_op
