#!/usr/bin/env python3
""" function to add two matrix
    use previous function to calculate shape
"""


def matrix_shape(matrix):
    """ function : calculate the shape of a matrix

    Arguments:
        matrix: matrix to calculate

    Returns:
        a list of integers
    """
    if not matrix or not isinstance(matrix, list):
        return []

    depth = matrix_shape(matrix[0])

    return [len(matrix)] + depth


def add_matrices2D(matrix1, matrix2):
    """ function : add two matrix

    Arguments:
        matrix1 : first matrix
        matrix2 : second matrix

    Returns:
        new matrix
    """
    shape1 = matrix_shape(matrix1)
    shape2 = matrix_shape(matrix2)

    if shape1 != shape2:
        return None
    elif matrix1 is None or matrix2 is None:
        return None

    new_matrix = []

    for i in range(shape1[0]):
        row_result = []

        for j in range(shape1[1]):
            sum_element = matrix1[i][j] + matrix2[i][j]

            row_result.append(sum_element)

        new_matrix.append(row_result)

    return new_matrix
