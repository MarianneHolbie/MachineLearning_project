#!/usr/bin/env python3
"""
    Class MultiNormal that represents a Multivariate Normal distribution
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


class MultiNormal:
    """
        Multivariate Normal distribution
    """

    def __init__(self, data):
        """
            class constructor

        :param data: ndarray, shape(d,n) containing data set
                n: number of data points
                d: number of dimensions in each data point
        """

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        # data shape(d,n) but in mean_cov X shape(n,d)
        mean, cov = mean_cov(data.T)
        self.mean = mean.T
        self.cov = cov

    def pdf(self, x):
        """
            calculates the PDF at a data point

            :param x: ndarray, shape(d,1) data point
                    d: number of dimensions of the Multinomial instance

            :return: value of the PDF
            f(x) = (1 / (sqrt((2 * π)^k * det(Σ)))) *
              exp(-0.5 * (x - μ)^T * Σ^(-1) * (x - μ))
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.ndim != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".
                             format(d))

        # calculate PDF
        cov_det = np.linalg.det(self.cov)
        cov_inv = np.linalg.inv(self.cov)

        exponent = -0.5 * np.matmul(np.matmul((x - self.mean).T, cov_inv),
                                    (x - self.mean))
        coefficient = (1 / np.sqrt((2 * np.pi) ** d * cov_det))
        pdf_value = coefficient * np.exp(exponent)

        return pdf_value[0][0]
