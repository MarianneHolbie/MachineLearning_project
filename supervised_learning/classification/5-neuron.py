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
        self.__W = np.random.normal(loc=0, scale=1, size=(1, nx))
        # b : bias & A activated output, both initialized to 0
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
            The weights vector for the neuron

            :return: value for private attribute __W
        """
        return self.__W

    @property
    def b(self):
        """
            The bias for the neuron

            :return: value for private attribute __b
        """
        return self.__b

    @property
    def A(self):
        """
            The activated output of the neuron (prediction)

            :return: value for private attribute __A
        """
        return self.__A

    def forward_prop(self, X):
        """
            method to calculate the forward propagation of the neuron

            :param X: ndarray (shape (nx, m)) contains input data

            :return: forward propagation
        """
        # multiplication of weight and add bias
        Z = np.matmul(self.__W, X) + self.__b

        # activation function
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A

    def cost(self, Y, A):
        """
            Method to calculate cost (using logistic regression)

            :param Y: ndarray shape(1,m) correct labels
            :param A: ndarray shape(1,m) activated output

            :return: the cost
        """
        # store m value
        m = Y.shape[1]

        # calculate log loss function
        log_loss = -(1 / m) * np.sum((Y * np.log(A) + (1-Y) *
                                      np.log(1.0000001 - A)))

        return log_loss

    def evaluate(self, X, Y):
        """
        Method to evaluate the neuron's prediction

        :param X: ndarray shape(nx,m) contains input data
        :param Y: ndarray shape (1,m) correct labels

        :return: neuron's prediction and cost of the network
        """

        # run forward propagation
        A = self.forward_prop(X)

        # calculate cost
        cost = self.cost(Y, A)

        # label values
        result = np.where(A >= 0.5, 1, 0)

        return result, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
            Method calculate one pass of gradient descent on the neuron

            :param X: ndarray, shape(nx,m) contains input data
            :param Y: ndarray, shape(1, m) correct labels
            :param A: ndarray, shape(1,m) activated output
            :param alpha: learning rate

            :return: one pass of gradient descent on the neuron
        """

        # store m
        m = X.shape[1]

        # calculate weight gradient with X transpose
        grad_w = 1 / m * np.matmul((A-Y), X.T)

        # calculate bias gradient
        grad_b = 1/m * np.sum((A-Y))

        # update parameters W and b
        self.__W = self.__W - alpha * grad_w
        self.__b = self.__b - alpha * grad_b
