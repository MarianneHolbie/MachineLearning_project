#!/usr/bin/env python3
""" function  that performs element-wise addition,
    subtraction, multiplication, and division
"""


def np_elementwise(mat1, mat2):
    """ function : perform element-wise addition, substraction,
    multiplication, division

    Arguments:
        mat1 : first matrix
        mat2 : second matrix

    Returns:
        tuple containing : sum, difference, product and quotient
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
