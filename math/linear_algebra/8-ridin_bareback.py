#!/usr/bin/env python3
""" function to multipy matrix
"""


def mat_mul(mat1, mat2):
    """ function : matrix multiplication

    Arguments:
        mat1 : first matrix
        mat2 : second matrix

    Returns:
        new matrix
    """

    if len(mat1[0]) != len(mat2):
        return None

    # initialise matrix avec des 0
    new_matrix = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # add number in newMatrix
    # i number of line of mat1
    for i in range(len(mat1)):
        # j number of column of mat2
        for j in range(len(mat2[0])):
            # k number of line of mat2
            for k in range(len(mat2)):
                new_matrix[i][j] += mat1[i][k] * mat2[k][j]

    return new_matrix
