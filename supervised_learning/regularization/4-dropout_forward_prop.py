#!/usr/bin/env python3
"""
    Forward Prop with L2 regularization
"""

import tensorflow.compat.v1 as tf
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
        Function that conduct forward prop using dropout

        :param X: ndarray, shape(nx,m), input data
        :param weights: dict, weights and biases of NN
        :param L: number of layers in the network
        :param keep_prob: proba that a node will be kepts

        :return: dict output each layer and dropout mask used on each layer
    """
    cache = {'A0': X}

    for l in range(1, L):
        Z = (np.matmul(weights["W" + str(l)],
                       cache['A' + str(l - 1)]) +
             weights['b' + str(l)])
        A = np.tanh(Z)
        dropout = np.random.binomial(1, keep_prob, size=A.shape)
        cache["D" + str(l)] = dropout
        A = np.multiply(A, dropout)
        cache['A' + str(l)] = A

    # last layer with softmax activation
    Z = (np.matmul(weights["W" + str(L)],
                   cache['A' + str(L - 1)]) +
         weights['b' + str(L)])
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    cache['A' + str(L)] = A

    return cache
