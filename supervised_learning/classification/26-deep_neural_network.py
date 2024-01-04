#!/usr/bin/env python3
"""
    Class DeepNeuralNetwork : deep NN performing binary classification
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


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
                self.__weights["W" + str(i + 1)] = (np.random.randn(layers[i],
                                                                    nx)
                                                    * np.sqrt(2 / nx))
            else:
                self.__weights["W" + str(i + 1)] = \
                    (np.random.randn(layers[i],
                                     layers[i - 1])
                     * np.sqrt(2 / layers[i - 1]))
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

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
                X = self.__cache['A{}'.format(i - 1)]
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

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
            Method to train deep neural network

            :param X: ndarray, shape(nx,m), input data
            :param Y: ndarray, shapte(1,m), correct labels
            :param iterations: number of iterations to train over
            :param alpha: learning rate
            :param verbose: boolean print or not information
            :param graph: boolean print or not graph
            :param step: int

            :return: evaluation of training after iterations
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # list to store cost /iter
        costs = []
        count = []

        for i in range(iterations + 1):
            # run forward propagation
            A, cache = self.forward_prop(X)

            # run gradient descent for all iterations except the last one
            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)

            cost = self.cost(Y, A)

            # store cost for graph
            costs.append(cost)
            count.append(i)

            # verbose TRUE, every step + first and last iteration
            if verbose and (i % step == 0 or i == 0 or i == iterations):
                # run evaluate
                print("Cost after {} iterations: {}".format(i, cost))

        # graph TRUE after training complete
        if graph:
            plt.plot(count, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
            Method to saves instance object to a file in pickle format

            :param filename: file which the object should be saved

        """
        # test extention
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        # open file in binary write mode
        with open(filename, 'wb') as file:
            # use pickel to dump the object into the file
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
            method to load a pickled DeepNeuralNetwork object

            :param filename: file from which object should be loaded

            :return: loaded object
                    or None if filename doesn't exist
        """

        try:
            # open file in binary mode
            with open(filename, 'rb') as file:
                # use pickle to load
                loaded_object = pickle.load(file)
            return loaded_object

        except FileNotFoundError:
            return None
