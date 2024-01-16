#!/usr/bin/env python3
"""
   RMSProp upgraded
"""

import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
        Method tha create the training operation for NN
        in tf using RMSProp optimization algo

        :param loss: loss of NN
        :param alpha: learning rate
        :param beta2: RMSProp weight
        :param epsilon: small number to avoid divsion by zero

        :return: RMSProp optimization operation
    """

    # set optimizer taht implement Momentum algo in tf
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon)

    # train_op to minimize loss with this optimizer
    train_op = optimizer.minimize(loss)

    return train_op
