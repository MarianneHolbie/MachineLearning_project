#!/usr/bin/env python3
"""
    One-Hot Decode
"""


import numpy as np


def one_hot_decode(one_hot):
    """
        Method that converts a one-hot vector into a
        numerical label vector

        :param one_hot: one-hot encoded ndarray, shape(classes,m)

        :return: ndarray shape(m) containing numeric labels
                 or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    else:
        return np.argmax(one_hot, axis=0)
