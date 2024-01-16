#!/usr/bin/env python3
"""
   Adam upgraded
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
        method that updates variable in place using
        Adam optimizer algo

        :param alpha: learning rate
        :param beta1: weight for first moment
        :param beta2: weight for second moment
        :param epsilon: small number to avoid division by zero
        :param var: ndarray, variable to be updated
        :param grad: ndarray, gradient of var
        :param v: previous first moment of var
        :param s: previous second moment of var
        :param t: time step used for bias correction

        :return: update var, new first moment, new second moment
    """

    # update rules
    new_v = beta1 * v + (1 - beta1) * grad
    new_s = beta2 * s + (1 - beta2) * grad**2

    # bias correction
    v_corrected = new_v / (1 - beta1**t)
    s_corrected = new_s / (1 - beta2**t)

    # update var
    var = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var, new_v, new_s
