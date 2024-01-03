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

    def forward_prop(self, X):
        """
            method calculate forward propagation of neural network

            :param X: ndarray, shape(nx,m) input data

            :return: output neural network and cache
        """

        # store X in A0
        if 'A0' not in self.__cache:
            self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            # first layer
            if i == 1:
                W = self.__weights["W{}".format(i)]
                b = self.__weights["b{}".format(i)]
                # multiplication of weight and add bias
                Z = np.matmul(W, X) + b
            else:  # next layers
                W = self.__weights["W{}".format(i)]
                b = self.__weights["b{}".format(i)]
                X = self.__cache['A{}'.format(i-1)]
                Z = np.matmul(W, X) + b

            # activation function
            self.__cache["A{}".format(i)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A{}".format(i)], self.__cache

    def cost(self, Y, A):
        """
            Calculate cost of the model using logistic regression

            :param Y: ndarray, shape(1,m) correct labels
            :param A: ndarray, shape(1,m) activated output

            :return: cost
        """

        # store m value
        m = Y.shape[1]

        # calculate log loss function
        log_loss = -(1 / m) * np.sum((Y * np.log(A) + (1 - Y) *
                                      np.log(1.0000001 - A)))

        return log_loss

    def evaluate(self, X, Y):
        """
                Method to evaluate the network's prediction

                :param X: ndarray shape(nx,m) contains input data
                :param Y: ndarray shape (1,m) correct labels

                :return: network's prediction and cost of the network
                """

        # run forward propagation
        output, cache = self.forward_prop(X)

        # calculate cost
        cost = self.cost(Y, output)

        # label values
        result = np.where(output >= 0.5, 1, 0)

        return result, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
            Method calculate one pass of gradient descent
            on neural network

            :param Y: ndarray, shape(1,m), correct labels
            :param cache: dictionary containing all intermediary value of
             network
            :param alpha: learning rate

        """

        # store m
        m = Y.shape[1]

        # derivative of final layer (output=self.L)
        dZ_f = cache["A{}".format(self.L)] - Y

        # back loop to calculate previous
        for layer in range(self.L, 0, -1):

            # activation previous layer
            A_p = cache["A{}".format(layer - 1)]

            # derivate
            dW = (1 / m) * np.matmul(dZ_f, A_p.T)
            db = (1 / m) * np.sum(dZ_f, axis=1, keepdims=True)

            # weight of current layer
            A = self.weights['W{}'.format(layer)]
            # derivate current layer
            dZ = np.matmul(A.T, dZ_f) * A_p * (1 - A_p)

            # update parameters W and b : new position
            self.__weights["W{}".format(layer)] -= alpha * dW
            self.__weights["b{}".format(layer)] -= alpha * db

            # update dz_f with new value found
            dZ_f = dZ
