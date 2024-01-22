#!/usr/bin/env python3
"""
    F1 score
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
        function that calculates F1 score of a confusion matrix

        :param confusion: ndarray, shape(classes, classes)

        :return: ndarray, shape(classes,) F1 score of each class
    """
    # number of classes
    classes = confusion.shape[0]
    # initialize f1 score matrix
    f_one = np.zeros((classes,))

    calc_precision = precision(confusion)
    calc_sensitivity = sensitivity(confusion)

    for i in range(classes):
        f_one[i] = (2 * (calc_precision[i] * calc_sensitivity[i]) /
                    (calc_precision[i] + calc_sensitivity[i]))

    return f_one
