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

        # Initialize public instance attribute
        # W1 & W2 normal distribution
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
