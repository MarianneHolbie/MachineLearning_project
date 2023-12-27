#!/usr/bin/env python3
"""
    Class Neuron
"""

import numpy as np


class Neuron:
    """
        Class Neuron : define single neuron performing binary classification
    """

    def __init__(self, nx):
        """
            Class constructor

            :param nx: number of input features to the neuron
        """
        # Manage exceptions
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # public instance attributes
        # W: Weights vector : initialized using random normal distribution
        self.W = np.random.normal(loc=0, scale=1, size=(1, nx))
        # b : bias & A activated output, both initialized to 0
        self.b = 0
        self.A = 0
