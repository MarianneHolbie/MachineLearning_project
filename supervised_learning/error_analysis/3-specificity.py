#!/usr/bin/env python3
"""
    Specificity
"""

import numpy as np


def specificity(confusion):
    """
        Function to calculates specificity of each class
        in a confusion matrix

        :param confusion: ndarray, shape(classes,classes) confusion matrix

        :return: ndarray, shape(classes,), specificity of each class
    """
    # number of classes
    classes = confusion.shape[0]
    # initialize specificity
    specificity_matrix = np.zeros((classes,))

    # formule specificity : true negativ / (true negativ + falses positif)
    for i in range(classes):
        true_positif = confusion[i, i]
        falses_positif = np.sum(confusion[:, i]) - true_positif
        falses_negatifs = np.sum(confusion[i, :]) - true_positif

        # true negatifs = Total sample - (true_positif + falses_positifs + falses_negatifs)
        true_negatif = np.sum(confusion) - (true_positif + falses_positif + falses_negatifs)

        specificity_matrix[i] = true_negatif / (true_negatif + falses_positif)

    return specificity_matrix
