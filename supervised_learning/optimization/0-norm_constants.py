#!/usr/bin/env python3
"""
    Function standardization constants
"""

import numpy as np


def normalization_constants(X):
    """
        Method to calculates the normalization constants

        :param X: ndarray, shape(m,nx) to normalize
                m : number of data points
                nx: number of features

        :return: mean and standard deviation of each feature, respectively
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
