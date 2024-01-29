#!/usr/bin/env python3
"""
    Make prediction using neural network
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
        makes a prediction using a neural network

        :param network: NN to make prediction with
        :param data: input data
        :param verbose: boolean, output printed or not

        :return: prediction of data
    """
    return network.predict(x=data,
                           verbose=verbose)
