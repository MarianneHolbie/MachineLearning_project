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
        if (not isinstance(layers, list) or
                not all(map(lambda x: isinstance(x, int) and x > 0, layers))):
            raise TypeError("layers must be a list of positive integers")

        # private attribute
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # initialize parameters with He method
        for i in range(self.__L):
            if i == 0:
                self.__weights["W" + str(i+1)] = (np.random.randn(layers[i],
                                                                  nx)
                                                  * np.sqrt(2 / nx))
            else:
                self.__weights["W" + str(i+1)] = \
                    (np.random.randn(layers[i],
                                     layers[i - 1])
                     * np.sqrt(2 / layers[i - 1]))
            self.__weights["b" + str(i+1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
            The number of layers in neural network

            :return: value for private attribute __L
        """
        return self.__L

    @property
    def cache(self):
        """
            Dictionary to hold all intermediary value
            Upon instantiation, empty

            :return: value for private attribute __cache
        """
        return self.__cache

    @property
    def weights(self):
        """
            Dictionary hold all weights and biased of network

            :return: value for private attribute __weights
        """
        return self.__weights
