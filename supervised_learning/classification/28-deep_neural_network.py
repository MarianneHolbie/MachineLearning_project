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

    def __init__(self, nx, layers, activation='sig'):
        """
            class constructor

            :param nx: number of input features
            :param layers: number of nodes in each layer
            :param activation: type of activation in hidden layer

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
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        # private attribute
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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

    @property
    def activation(self):
        """
            activation function used in the hidden layer

            :return: value for private attribute __activation
        """
        return self.__activation

    def forward_prop(self, X):
        """
            method calculate forward propagation of neural network

            :param X: ndarray, shape(nx,m) input data

            :return: output neural network and cache
        """

        # store X in A0

        self.__cache['A0'] = X
        L = self.__L

        for l in range(1, L):
            Z = (np.matmul(self.__weights["W" + str(l)],
                           self.__cache['A' + str(l - 1)]) +
                 self.__weights['b' + str(l)])
            if self.__activation == 'sig':
                A = 1 / (1 + np.exp(-Z))
            else:
                A = np.tanh(Z)
            self.__cache['A' + str(l)] = A

        Z = (np.matmul(self.__weights["W" + str(L)],
                       self.__cache['A' + str(L - 1)]) +
             self.__weights['b' + str(L)])
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        self.__cache['A' + str(L)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
            Calculate cross-entropy cost for multiclass

            :param Y: ndarray, shape(1,m) correct labels
            :param A: ndarray, shape(1,m) activated output

            :return: cost
        """

        # store m value
        m = Y.shape[1]

        # calculate log loss function
        log_loss = -(1 / m) * np.sum(Y * np.log(A))

        return log_loss

    def evaluate(self, X, Y):
        """
            Method to evaluate the network's prediction

            :param X: ndarray shape(nx,m) contains input data
            :param Y: one-hot ndarray shape (classes,m)

            :return: network's prediction and cost of the network
        """

        # run forward propagation
        A, _ = self.forward_prop(X)

        # calculate cost
        cost = self.cost(Y, A)

        return np.where(A == np.max(A, axis=0), 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
            Method calculate one pass of gradient descent
            on neural network

            :param Y: ndarray, shape(1,m), correct labels
            :param cache: dictionary containing all intermediary value of
             network
            :param alpha: learning rate

        """
        L = self.__L

        # store m
        m = Y.shape[1]

        # derivative of final layer (output=self.L)
        dZ = cache['A' + str(L)] - Y
        dW = np.matmul(dZ, cache['A' + str(L - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        W_prev = np.copy(self.__weights['W' + str(L)])
        self.__weights['W' + str(L)] -= alpha * dW
        self.__weights['b' + str(L)] -= alpha * db

        for l in range(L - 1, 0, -1):
            dA = np.matmul(W_prev.T, dZ)
            A = cache['A' + str(l)]
            if self.__activation == 'sig':
                dZ = dA * A * (1 - A)
            else:
                dZ = dA * (1 - (A ** 2))
            dW = np.matmul(dZ, cache['A' + str(l - 1)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            W_prev = np.copy(self.__weights['W' + str(l)])
            self.__weights['W' + str(l)] -= alpha * dW
            self.__weights['b' + str(l)] -= alpha * db

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
        if graph:
            costs = []
            count = []

        for i in range(iterations + 1):
            # run forward propagation
            A, _ = self.forward_prop(X)

            # run gradient descent for all iterations except the last one
            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)

            if i % step == 0 or i == iterations:
                current_cost = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".
                          format(i, current_cost))
                if graph:
                    costs.append(current_cost)
                    count.append(i)

            # verbose TRUE, every step + first and last iteration
            if verbose and (i % step == 0 or i == 0 or i == iterations):
                # run evaluate
                print("Cost after {} iterations: {}".format(i, current_cost))

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
