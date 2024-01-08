#!/usr/bin/env python3
"""
    Function Placeholders
"""

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
        Method to create placeholders

        :param nx: number of feature columns in our data
        :param classes: number of classes in our classifier

        :return: Two placeholders x(input data) and y(one-hot labels)
    """

    x = tf.placeholder("float32", [None, nx], name="x")
    y = tf.placeholder("float32", [None, classes], name="y")

    return x, y
