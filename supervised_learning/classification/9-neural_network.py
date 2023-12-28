#!/usr/bin/env python3
"""
    Class NeuralNetwork : NN with one hidden layer
                          performing binary classification
"""

import numpy as np


class NeuralNetwork:
    """
        class NeuralNetwork
    """

    def __init__(self, nx, nodes):
        """
            class constructor

            :param nx: number of input features
            :param nodes: number of nodes in the hidden layer
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Private public instance attribute
        # W1 & W2 normal distribution
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    # Getter function

    @property
    def W1(self):
        """
            The weights vector of hidden layer

            :return: private value of W1
        """
        return self.__W1

    @property
    def b1(self):
        """
            The bias of hidden layer

            :return: private value of b1
        """
        return self.__b1

    @property
    def A1(self):
        """
            The activated output of hidden layer

            :return: private value of A1
        """
        return self.__A1

    @property
    def W2(self):
        """
            The weights vector of output neuron

            :return: private value of W2
        """
        return self.__W2

    @property
    def b2(self):
        """
            The bias of output neuron

            :return: private value of b2
        """
        return self.__b2

    @property
    def A2(self):
        """
            The activated output neuron (prediction)

            :return: private value of A2
        """
        return self.__A2
