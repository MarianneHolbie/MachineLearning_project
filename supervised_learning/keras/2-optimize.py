#!/usr/bin/env python3
"""
    Optimize
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
        function that sets up Adam optimization for a keras model
        with categorical crossentropy loss and accuracy metrics

        :param network: model to optimize
        :param alpha: learning rate
        :param beta1: first Adam optimization param
        :param beta2: second Adam optimization param

        :return: None
    """
    Adam_optimizer = K.optimizers.Adam(learning_rate=alpha,
                                       beta_1=beta1,
                                       beta_2=beta2)

    network.compile(optimizer=Adam_optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
