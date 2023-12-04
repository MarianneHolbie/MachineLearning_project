#!/usr/bin/env python3
""" function to transpose a matrix"""


def matrix_transpose(matrix):
    """ function :transpose a matrix

    Arguments:
        matrix: matrix to calculate

    Returns:
        transposed matrix
    """

    result = [[matrix[j][i] for j in range(len(matrix))]
              for i in range(len(matrix[0]))]
    return result
