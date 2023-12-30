#!/usr/bin/env python3
"""
    Class NeuralNetwork : NN with one hidden layer
                          performing binary classification
"""

import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """
            method to calculate the forward propagation of NN

            :param X: ndarray (shape (nx, m)) contains input data

            :return: private attribute __A1 and __ A2
        """
        # multiplication of weight and add bias
        Z1 = np.matmul(self.__W1, X) + self.__b1

        # activation function
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
            public method calculate cost function using logistic regression

            :param Y: ndarray shape(1,m), correct labels
            :param A: ndarray shape(1,m), activated output

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
        A1, A2 = self.forward_prop(X)

        # calculate cost
        cost = self.cost(Y, A2)

        # label values
        result = np.where(A2 >= 0.5, 1, 0)

        return result, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
            Method calculate one pass of gradient descent
            on neural network

            :param X:  ndarray, shape(nx, m) input data
            :param Y: ndarray, shape(1,m), correct labels
            :param A1: output hidden layer
            :param A2: predicted output
            :param alpha: learning rate

        """

        # store m
        m = X.shape[1]

        # derivative 2
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # derivative 1
        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # update parameters W and b
        self.__W1 = self.__W1 - alpha * dW1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b1 = self.__b1 - alpha * db1
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
            Method to train neural network

            :param X: ndarray, shape(nx,m) input data
            :param Y: ndarray, shape(1,m) correct labels
            :param iterations: number of iterations to train
            :param alpha: learning rate
            :param verbose: boolean print or not information
            :param graph: boolean print or not graph
            :param step: int

            :return: evaluation of the training data
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 1 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        # list to store cost /iter
        costs = []
        count = []

        for i in range(iterations + 1):
            # run forward propagation
            self.forward_prop(X)
            # run gradient descent
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

            # verbose TRUE, every step + first and last iteration
            if verbose and (i % step == 0 or i == 0 or i == iterations):
                # run evaluate
                cost = self.cost(Y, self.__A2)
                print(f"Cost after {i} iterations: {cost}")

                # store cost for graph
                costs.append(cost)
                count.append(i)

            # graph TRUE after training complete
        if graph:
            plt.plot(count, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
