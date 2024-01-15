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

        :return: shuffled X and Y matrices
    """
    m = X.shape[0]
    permutted_index = np.random.permutation(m)
    X_shuffled = X[permutted_index]
    Y_shuffled = Y[permutted_index]
    return X_shuffled, Y_shuffled
