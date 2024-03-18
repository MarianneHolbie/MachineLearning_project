#!/usr/bin/env python3
"""
    Minor
"""


def sub_matrix(matrix, i):
    """
        function to create submatrix 2x2

        :param matrix: initial matrix
        :param i: row, column to pop

        :return: submatrix 2x2
    """
    if not matrix:
        return []

    matrix2 = []
    for row in matrix[1:]:
        matrix2.append(row[:i] + row[i + 1:])

    return matrix2


def determinant(matrix):
    """
        function that calculates the determinant of a matrix

        :param matrix: list of lists whose determinant should be calculated

        :return: determinant of matrix
    """
    # Test list of list format
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    d = 0

    # special case
    if len(matrix[0]) == 0:
        return 1

    # test square format of matrix
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    # special case only one element
    if len(matrix) == 1:
        d = matrix[0][0]
        return d
    # simple matrix of 2x2
    elif len(matrix) == 2:
        d = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        return d
    # recursiv action to sum all sub_matrix
    else:
        for i in range(len(matrix[0])):
            d += (((-1) ** i) * matrix[0][i] *
                  determinant(sub_matrix(matrix, i)))
        return d


def minor(matrix):
    """
        function that calculates the minor of a matrix

        :param matrix: list of lists whose minor should be calculated

        :return: minor of matrix
    """
    # Test list of list format
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    # test square matrix
    if len(matrix) != len(matrix[0]) or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    # special case
    if len(matrix) == 1:
        return [[1]]

    minor_matrix = []

    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix[i])):
            # create sub matrix removing i and j column
            sub_matrix_value = [row[:j] + row[j + 1:] for row_idx, row
                                in enumerate(matrix) if row_idx != i]
            # ensure sub_matrix is a list of list
            sub_matrix_value = [row for row in sub_matrix_value if row]
            # check if sublist is empty
            if not sub_matrix_value:
                det_sub_matrix = 0
            else:
                det_sub_matrix = determinant(sub_matrix_value)
            minor_row.append(det_sub_matrix)
        minor_matrix.append(minor_row)

    return minor_matrix


def cofactor(matrix):
    """
        function that calculates the cofactor matrix of a matrix

        :param matrix: list of lists whose cofactor should be calculated

        :return: cofactors matrix
    """
    # Test list of list format
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    # test square matrix
    if len(matrix) != len(matrix[0]) or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = minor(matrix)
    cofactor_matrix = []

    for i in range(len(minor_matrix)):
        cofactor_row = []
        # change the sign of alternate cells
        for j in range(len(minor_matrix[i])):
            cofactor_value = (-1) ** (i + j) * minor_matrix[i][j]
            cofactor_row.append(cofactor_value)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


def adjugate(matrix):
    """
        function that calculates the adjugate matrix of matrix

        :param matrix: list of lists whose adjugate matrix should be calculated

        :return: adjugate matrix
    """
    # Test list of list format
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    # test square matrix
    if len(matrix) != len(matrix[0]) or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = []

    # swap positions of cofactor matrix over the diagonal
    if len(matrix) == 1:
        return [[1]]
    else:
        for i in range(len(matrix)):
            adjugate_row = []
            for j in range(len(matrix[0])):
                adjugate_row.append(cofactor_matrix[j][i])
            adjugate_matrix.append(adjugate_row)

    return adjugate_matrix


def inverse(matrix):
    """
        function that calculates the inverse of a matrix

        :param matrix: list of list whose inverse should be calculated

        :return: inverse of matrix, or None if matrix is singular
    """
    # Test list of list format
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    # test square matrix
    if len(matrix) != len(matrix[0]) or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None
    else:
        inverse_matrix = []
        adj = adjugate(matrix)
        # calculate inverse by dividing each element by determinant
        for row in adj:
            inverse_row = []
            for element in row:
                inverse_row.append(element / det)
            inverse_matrix.append(inverse_row)
        return inverse_matrix
