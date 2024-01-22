#!/usr/bin/env python3
"""
    Gradient Descent with L2 regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
        function that updates the weights and biases of NN
        using gradient descent with L2 regularization

        Formula:
        W = W - alpha * (dW + (lambtha/m) * W)

        :param Y: ndarray, shape(classes,m) correct labels
        :param weights: dict of weights and biases of NN
        :param cache: dict of outputs of each layer of NN
        :param alpha: leaning rate
        :param lambtha: L2 regularization parameter
        :param L: number of layers of the network

    """
    # store m number training examples
    m = Y.shape[1]

    # derivative of final layer
    # starting point for backpropagation
    dZ = cache['A' + str(L)] - Y

    # backpropagation loop
    for layer in range(L, 0, -1):
        L2_regularization = lambtha / m * weights['W' + str(layer)]

        # activation of previous layer
        A_prev = cache['A' + str(layer - 1)]

        # Gradient of Weight and biases
        dW = np.matmul(dZ, A_prev.T) / m + L2_regularization
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.matmul(weights['W' + str(layer)].T, dZ)

        # derivative of tanh activation
        if layer != 1:
            dZ = dA * (1 - A_prev ** 2)
        else:  # Apply softmax derivative for the last layer
            dZ = dA

        # update weights
        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
