#!/usr/bin/env python3
"""
    Function shuffles data
"""

import numpy as np

def shuffle_data(X, Y):
    """
        Function that shuffles the data points in two matrices the same way

        :param X: ndarray, shape(m, nx) to shuffle
        :param Y: ndarray, shape(m, ny) to shuffle

        :return: shuffled X and Y atrices
    """
    X_shuffled = np.random.permutation(X)
    Y_shuffled = np.random.permutation(Y)
    return X_shuffled, Y_shuffled
