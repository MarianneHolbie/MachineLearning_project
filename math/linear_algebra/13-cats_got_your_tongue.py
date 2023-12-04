#!/usr/bin/env python3
""" function  that concatenates two matrices along
    a specific axis
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ function : concatenate two matrix along specific axis

    Arguments:
        mat1 : first matrix
        mat2 : second matrix
        axis: specific axis for concatenation (0 vertical, 1 horizontal)

    Returns:
       new numpy array
    """
    return np.concatenate((mat1, mat2), axis=axis)
