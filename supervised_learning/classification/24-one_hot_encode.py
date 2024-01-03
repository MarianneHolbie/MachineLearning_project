#!/usr/bin/env python3
"""
    One-Hot Encode
"""


import numpy as np


def one_hot_encode(Y, classes):
    """
        Method that converts a numerical label vector into a
        one-hot vector

        :param Y: ndarray, shape(m), numeric class labels
        :param classes:  maximum number of classes found in Y

        :return: one-hot encoding of Y, shape(classes,m)
                 or None on failure
    """

    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < np.max(Y) + 1:
        return None

    # creating a 2D array filled with 0's
    encoded_array = np.zeros((classes, Y.size), dtype=float)

    # replacing 0 with a 1 at the index of the original array
    encoded_array[Y, np.arange(Y.size)] = 1

    return encoded_array
