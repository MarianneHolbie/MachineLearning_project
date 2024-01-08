#!/usr/bin/env python3
"""
    Function Accuracy
"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
        Method to calculates the accuracy of a prediction

        :param y: placeholder for labels of input data
        :param y_pred: tensor containing network's predictions

        :return: tensor containing decimal accuracy of prediction
    """
    # comparison of indice's max value for y and y_pred
    correct_prediction = tf.equal(tf.argmax(y, axis=1),
                                  tf.argmax(y_pred, axis=1))

    # convert tensor bool in float32
    correct_prediction = tf.cast(correct_prediction, dtype=tf.float32)

    # mean of prediction
    accuracy = tf.reduce_mean(correct_prediction)

    return accuracy
