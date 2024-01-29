#!/usr/bin/env python3
"""
    Save and load weight function
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
        function that saves a model's weights

        :param network: model whose weights should be saved
        :param filename: path file where saved weights
        :param save_format: format in which saved weights

        :return: None
    """
    network.save_weights(filepath=filename,
                         save_format=save_format)


def load_weights(network, filename):
    """
        function that loads a model's weights

        :param network: model to which the weights should be loaded
        :param filename: path of the file where weights to be loaded

        :return: None
    """
    network.load_weights(filepath=filename)
