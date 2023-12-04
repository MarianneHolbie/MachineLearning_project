#!/usr/bin/env python3
""" function to add two matrix
    use previous function to calculate shape
"""


matrix_shape = __import__('2-size_me_please').matrix_shape


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
    elif not matrix1 or not matrix2:
        return []

    new_matrix = []

    for i in range(len(matrix1)):
        row_result = []

        for j in range(len(matrix1[0])):
            sum_element = matrix1[i][j] + matrix2[i][j]

            row_result.append(sum_element)

        new_matrix.append(row_result)

    return new_matrix
