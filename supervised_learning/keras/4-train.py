#!/usr/bin/env python3
"""
    Train
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
        Function that trains a model using mini-batch gradient descent

        :param network: model to train
        :param data: ndarray, shape(m, nx), input data
        :param labels: ndarray, shape(m,classes), labels
        :param batch_size: size of the batch
        :param epochs: number of passes through data for mini-bath
        :param verbose: boolean, print or not during training
        :param shuffle: boolean, shuffle or not every epoch

        :return: History
    """
    history = network.fit(x=data,
                          y=labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=verbose,
                          shuffle=shuffle)

    return history
