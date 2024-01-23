#!/usr/bin/env python3
"""
    Gradient descent with L2 regularization
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
        function that updates weights of NN with Dropout reg
        using gradient descent

        :param Y: ndarray, shape(classes,m) correct labels
        :param weights: dict, weights and biases of NN
        :param cache: dict, output and dropout mask of each layer
        :param alpha: learning rate
        :param keep_prob: proba a node will be kept
        :param L: number of layer of network
    """

    # store m
    m = Y.shape[1]

    # derivative of final layer
    dZ = cache['A' + str(L)] - Y
    dW = np.matmul(dZ, cache['A' + str(L - 1)].T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m

    W_prev = np.copy(weights['W' + str(L)])
    weights['W' + str(L)] -= alpha * dW
    weights['b' + str(L)] -= alpha * db

    for layer in range(L - 1, 0, -1):
        dA = np.matmul(W_prev.T, dZ)
        # apply mask dropout & normalize
        dA = np.multiply(dA, cache['D' + str(layer)])
        dA = dA * 1 / keep_prob

        A = cache['A' + str(layer)]
        dZ = np.multiply(dA, np.int64(A > 0))
        dW = np.matmul(dZ, cache['A' + str(layer - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        W_prev = np.copy(weights['W' + str(layer)])
        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
