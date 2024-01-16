#!/usr/bin/env python3
"""
   RMSProp
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
        function that updates a variable
        using the RMSProp optimization algo

        :param alpha: learning rate
        :param beta2: RMSProp weight
        :param epsilon: small number to avoid division by zero
        :param var: ndarray, variable to be updated
        :param grad: ndarray, gradient of var
        :param s: previous second moment of var

        :return: updated variable and new moment
    """

    # squared gradient
    squared_gradient = beta2 * s + (1 - beta2) * grad**2

    # update the variable using RMSprop update rule
    update_var = var - alpha * grad / (np.sqrt(squared_gradient) + epsilon)

    return update_var, squared_gradient
