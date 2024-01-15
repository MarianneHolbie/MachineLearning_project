#!/usr/bin/env python3
"""
    Function standardization matrix
"""

import numpy as np


def normalize(X, m, s):
    """
        Method to calculates the normalization of a matrix

        :param X: ndarray, shape(d,nx) to normalize
                d : number of data points
                nx: number of features
        :param m: ndarray, shape(nx,) mean of all features of X
        :param s: ndarray, shape(nx,) std of all features of X

        :return: noramlized X matrix
    """
    # formule of normalisation Z = (X - mean) / std
    Z = (X - m) / s
    return Z
