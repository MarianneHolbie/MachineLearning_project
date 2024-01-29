#!/usr/bin/env python3
"""
    Save and load model function
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
        function that saves an entire model

        :param network: model to save
        :param filename: path where model saved to

        :return: None
    """
    network.save(filename)


def load_model(filename):
    """
        function that loads an entire model

        :param filename: path where model loaded from

        :return: loaded model
    """
    return K.models.load_model(filename)
