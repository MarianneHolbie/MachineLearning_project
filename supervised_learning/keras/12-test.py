#!/usr/bin/env python3
"""
    Test neural network
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
        function that tests a neural network

        :param network: model to test
        :param data: input data to test model with
        :param labels: correct one-hot labels of data
        :param verbose: boolean, output printed or not

        :return: loss, accuracy of model with testing data
    """
    return network.evaluate(x=data,
                            y=labels,
                            verbose=verbose)
