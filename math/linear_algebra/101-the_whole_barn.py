#!/usr/bin/env python3
""" function that add two matrix
"""


def add_matrices(mat1, mat2):
    """ function : add matrix1 and matrix2

    Arguments:
        mat1 : matrix 1
        mat2 : matrix 2

    Returns:
        new matrix
    """

    # compare type of mat1 and mat2
    if type(mat1) != type(mat2):
        return None

    # compare if mat1 and mat2 are list
    if isinstance(mat1, list):
        # if lengths are different or mat1 and mat2 are empty lists
        if len(mat1) != len(mat2) or not mat1 or not mat2:
            return None

        # zip :  take 2 sequences return tuples of 2 elements
        result = [add_matrices(m1, m2) for m1, m2 in zip(mat1, mat2)]

        # check if any element in result is None
        if any(element is None for element in result):
            return None

        # if result is a list of list with one element, return the element
        if len(result) == 1:
            return result[0]
        else:
            return result

    else:
        try:
            return mat1 + mat2
        except TypeError:
            # In case mat1 and mat2 are not compatible for addition
            return None
