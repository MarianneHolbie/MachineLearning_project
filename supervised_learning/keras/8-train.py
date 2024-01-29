#!/usr/bin/env python3
"""
    SAve only the Best
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
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
        :param learning_rate_decay: boolean, use or not learning rate decay
        :param alpha: initial learning rate
        :param decay_rate: decay rate
        :param save_best: boolean, save model after epoch if the best
        :param filepath: where model should be saved
        :param verbose: boolean, print or not during training
        :param shuffle: boolean, shuffle or not every epoch

        :return: History
    """
    callback = []
    if early_stopping is True and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)

        # add to callback list
        callback.append(early_stop)

    if learning_rate_decay and validation_data:
        # function calculate new learning rate
        def scheduler(epochs):
            lr = alpha / (1 + decay_rate * epochs)
            return lr

        inv_time_decay = K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1)

        # add to callback list
        callback.append(inv_time_decay)

    # save best model
    if save_best:
        save_best_model = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True
        )

        callback.append(save_best_model)

    history = network.fit(x=data,
                          y=labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=validation_data,
                          callbacks=[callback],
                          verbose=verbose,
                          shuffle=shuffle)

    return history
