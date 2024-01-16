#!/usr/bin/env python3
"""
   Momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
        method that updates a variable using the gradient descent
        with momentum optimization algorithm

        :param alpha: learning rate
        :param beta1: momentum weight
        :param var: ndarray, variable to be updated
        :param grad: ndarray, gradient of var
        :param v: previous first moment of var

        :return: updated variable and the new moment
    """
    # formule of momentum
    dW = beta1 * v + (1 - beta1) * grad

    # update var
    var_new = var - dW * alpha

    return var_new, dW
