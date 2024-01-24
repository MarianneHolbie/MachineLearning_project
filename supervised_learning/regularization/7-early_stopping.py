#!/usr/bin/env python3
"""
    Early Stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
        function that determines if should stop gradient descent early

        :param cost: current validation cost of NN
        :param opt_cost: lowest recorded validation cost of NN
        :param threshold: threshold used for early stopping
        :param patience: count used for early stopping
        :param count: count of how long the threshold has not been met

        :return: boolean (be stopped early), updated count
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return count == patience, count
