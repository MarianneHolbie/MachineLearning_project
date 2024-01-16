#!/usr/bin/env python3
"""
   Batch Normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
        Method that normalizes an unactivated output of a NN
        using batch normalization

        :param Z: ndarray, shape(m,n) to be normalized
        :param gamma: ndarray, shape(1,n), scales used for batch norm
        :param beta: ndarray, shape(1,n) offsets for batch norm
        :param epsilon: small number used to avoid division by zero

        :return: normalized z matrix
    """
    # compute mean and variance
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    std_dev = np.sqrt(variance + epsilon)

    # normalize : - mean / root square variance
    Z_norm = (Z - mean) / std_dev

    # scale by gamma and shifted by beta
    scaled = gamma * Z_norm + beta

    return scaled
