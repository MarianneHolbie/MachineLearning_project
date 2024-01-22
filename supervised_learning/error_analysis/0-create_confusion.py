#!/usr/bin/env python3
"""
    Matrix confusion
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
        function that creates a confusion matrix

        :param labels: ndarray, shape(m,classes), correct labels
        :param logits: ndarray, shape(m,classes), predicted labels
        :return: ndarray, shape(classes,classes),
                row indices = correct labels,
                columns indices = predicted labels
    """
    # number of example
    m = labels.shape[0]
    # classes
    classes = labels.shape[1]
    # initialize confusion matrix
    conf_matrix = np.zeros((classes, classes))

    # loop across the examples
    for i in range(m):
        # determine true class & predicted class
        true_class = np.argmax(labels[i])
        predicted_class = np.argmax(logits[i])

        # sum in confusion matrix in the corresponding cell
        conf_matrix[true_class, predicted_class] += 1

    return conf_matrix
