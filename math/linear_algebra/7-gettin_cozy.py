#!/usr/bin/env python3
""" function to concatenate two matrix
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ function : concatenate two matrix

    Arguments:
        mat1 : first matrix
        mat2 : second matrix
        axis: specific axis for concatenation (0 vertical, 1 horizontal)

    Returns:
        new concatened matrix
    """

    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    elif axis == 1 and len(mat1) != len(mat2):
        return None
    elif not mat1 or not mat2:
        return None

    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        # zip :  take 2 sequences return tuples of 2 elements
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
