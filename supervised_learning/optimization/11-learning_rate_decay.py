#!/usr/bin/env python3
"""
   Learning Rate Decay
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
        Method to updates the learning rate using
        inverse time decay in numpy

        :param alpha: original learning rate
        :param decay_rate: weight used to determine the rate at
            which alpha will decay
        :param global_step: number of passes of gradient descent
            that have elapsed
        :param decay_step: number of passes of gradient descent
            that should occur before alpha is decayed further

        :return: updates value for alpha
    """
    # calculate factor that increases over time
    factor = (1 + decay_rate * (global_step//decay_step))
    # scale original learning rate by inverse of the factor
    alpha = alpha / factor
    return alpha
