#!/usr/bin/env python3
"""
    Mean and Covariance
"""
import numpy as np


def mean_cov(X):
    """
        function that calculates the mean and covariance of a data set

        :param X: ndarray,shape(n,d)
                    n: number of data points
                    d: number of dimensions in each data point

        :return: mean, cov
                mean: ndarray, shape(1,d), containing mean of data set
                cov: ndarray, shape(d,d), cov matrix of data set
    """
    if not isinstance(X, np.ndarray) or not len(X.shape) == 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Mean for each line
    mean = np.mean(X, axis=0, keepdims=True)

    # calculate difference between each element of X and corresponding mean
    X_mean = X - mean
    # calculate matrix of covariance
    cov = np.matmul(X_mean.T, X_mean) / (n - 1)

    return mean, cov
