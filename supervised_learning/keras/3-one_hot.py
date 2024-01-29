#!/usr/bin/env python3
"""
    One Hot
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
        function that converts a label vector into a one-hot matrix

        :param labels: labels
        :param classes: nbr of classes

        :return: one-hot matrix, shape(labels,classes)
    """
    return K.utils.to_categorical(labels, classes)
