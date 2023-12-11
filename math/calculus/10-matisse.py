#!/usr/bin/env python3
""" Function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """ function : calculate derivative of a polynomial

        Arguments:
            poly: list of coefficients representing a polynomial
                index : represent the power of x that the coefficient
                belongs to

        Returns:
            new list of coefficients representing the derivative of
            the polynomial
    """
    if not poly or not isinstance(poly, list) \
            or not all(isinstance(cuff, (int, float)) for cuff in poly):
        return None
    else:
        # calculate derivative of polynom
        derivative_cuff = [n * cuff for n, cuff in enumerate(poly[1:],
                                                             start=1)]

        # special case derivative is 0
        if not derivative_cuff:
            return [0]

        return derivative_cuff
