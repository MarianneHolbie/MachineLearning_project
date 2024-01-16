#!/usr/bin/env python3
"""
   Momentum upgraded
"""

import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
        Method to creates the training operation for a NN
        in tf using gradient descent with momentum opt algo

        :param loss: loss of network
        :param alpha: learning rate
        :param beta1: momentum weight

        :return: momentum optimization operation
    """
    # set optimizer taht implement Momentum algo in tf
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha,
                                           momentum=beta1)

    # train_op to minimize loss with this optimizer
    train_op = optimizer.minimize(loss)

    return train_op
