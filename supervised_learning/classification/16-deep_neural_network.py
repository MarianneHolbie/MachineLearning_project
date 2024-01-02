#!/usr/bin/env python3
"""
    Class DeepNeuralNetwork : deep NN performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
        class DeepNeuralNetwork
    """

    def __init__(self, nx, layers):
        """
            class constructor

            :param nx: number of input features
            :param layers: number of nodes in each layer

        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if not isinstance(layers, list) or not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # initialize parameters with He method
        for l in range(1, self.L):
            self.weights["W" + str(l)] = np.random.randn(layers[l], layers[l-1]) * np.sqrt(2./layers[l-1])
            self.weights["b" + str(l)] = np.zeros((layers[l], 1))
