#!/usr/bin/env python3
"""
    L2 regularization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
        Function that calculates the cost of a network with L2 Regularization

        FORMULE = loss + lamda/2m * sum||w||**2

        :param cost: cost of network without L2 regularization
        :param lambtha: regularization parameter
        :param weights: dict of weights and biases (ndarrays)
        :param L: number of layers in the NN
        :param m: number of data points used

        :return: cost of network accounting for L2 regularization
    """
    reg_term = 0

    for i in range(1, L + 1):
        # construct key
        weights_key = 'W' + str(i)

        # extract weight matrix
        W_i = weights[weights_key]

        reg_term += np.sum(np.square(W_i))

    cost_L2 = cost + (lambtha / (2 * m)) * reg_term

    return cost_L2
