#!/usr/bin/env python3
"""
    Cost of NN with L2 regularization
"""

import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
        Function that calculate the cost of NN with L2 regularization

        :param cost: tensor containing cost without L2 regularization

        :return: tensor cost WITH L2 regularization
    """
    regularization_L2 = tf.compat.v1.losses.get_regularization_losses()

    cost_L2reg = cost + regularization_L2

    return cost_L2reg
