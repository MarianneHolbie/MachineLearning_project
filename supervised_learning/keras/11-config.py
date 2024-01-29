#!/usr/bin/env python3
"""
    Save and load configuration
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
        saves a model's configuration in JSON format

        :param network: model whose configuration should be saved
        :param filename: path where be saved to

        :return: None
    """
    network_config = network.to_json()
    with open(filename, "w") as f:
        f.write(network_config)


def load_config(filename):
    """
        loads a model with a specific configuration

        :param filename: path containing the model's configuration

        :return: loaded model
    """
    with open(filename, "r") as f:
        load_network_config = f.read()
    return K.models.model_from_json(load_network_config)
