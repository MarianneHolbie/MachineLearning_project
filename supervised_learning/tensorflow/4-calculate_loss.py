#!/usr/bin/env python3
"""
    Function softmax cross-entropy loss
"""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
        Method to calculate the softmax cross-entropy loss
        of a prediction

        :param y: placeholder for labels input data
        :param y_pred: tensor network's prediction

        :return: tensor loss prediction
    """

    loss = tf.compat.v1.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred)

    return loss
