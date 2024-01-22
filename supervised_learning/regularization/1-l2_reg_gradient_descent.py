#!/usr/bin/env python3
"""
    Gradient Descent with L2 regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
        function that updates the weights and biases of NN
        using gradient descent with L2 regularization

        :param Y: ndarray, shape(classes,m) correct labels
        :param weights: dict of weights and biases of NN
        :param cache: dict of outputs of each layer of NN
        :param alpha: leaning rate
        :param lambtha: L2 regularization parameter
        :param L: number of layers of the network

    """
    # store m
    m = Y.shape[1]

    # derivative of final layer (output=self.L)
    dZ = cache['A' + str(L)] - Y
    dW = np.matmul(dZ, cache['A' + str(L - 1)].T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    W_prev = np.copy(weights['W' + str(L)])
    weights['W' + str(L)] -= alpha * dW
    weights['b' + str(L)] -= alpha * db

    for l in range(L - 1, 0, -1):
        L2_regularization = lambtha / m * weights['W' + str(l)]
        dA = np.matmul(W_prev.T, dZ)
        A = cache['A' + str(l)]

        # derivative of tanh activation
        if l != 1:
            dZ = dA * (1 - (np.tanh(A))**2)
        else:  # Apply softmax derivative for the last layer
            dZ = dA

        dW = np.matmul(dZ, cache['A' + str(l - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        W_prev = np.copy(weights['W' + str(l)])

        # update weights
        weights['W' + str(l)] -= alpha * (dW + L2_regularization)
        weights['b' + str(l)] -= alpha * db
