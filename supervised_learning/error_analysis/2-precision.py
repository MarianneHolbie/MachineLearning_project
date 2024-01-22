#!/usr/bin/env python3
"""
    Precision
"""

import numpy as np


def precision(confusion):
    """
        function that calculates the precision for each class
        in a confusion matrix

        :param confusion: ndarray, shape(classes,classes), confusion matrix

        :return: ndarray, shape(classes,), precision for each class
    """
    # number of classes
    classes = confusion.shape[0]
    # initialize sensitivity
    precision_matrix = np.zeros((classes,))

    for i in range(classes):
        true_positif = confusion[i, i]
        falses_positif = np.sum(confusion[:, i]) - true_positif
        precision_matrix[i] = true_positif / (true_positif + falses_positif)

    return precision_matrix
