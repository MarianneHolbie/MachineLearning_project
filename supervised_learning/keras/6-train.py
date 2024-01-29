#!/usr/bin/env python3
"""
    Train
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
        Function that trains a model using mini-batch gradient descent

        :param network: model to train
        :param data: ndarray, shape(m, nx), input data
        :param labels: ndarray, shape(m,classes), labels
        :param batch_size: size of the batch
        :param epochs: number of passes through data for mini-bath
        :param validation_data: data to validate the model
        :param early_stopping: boolean, use or not early stopping
        :param patience: patience for early stopping
        :param verbose: boolean, print or not during training
        :param shuffle: boolean, shuffle or not every epoch

        :return: History
    """
    callback = []
    if early_stopping is True and validation_data is not None:
        callback = K.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=patience)

    history = network.fit(x=data,
                          y=labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=validation_data,
                          callbacks=[callback],
                          verbose=verbose,
                          shuffle=shuffle)

    return history
