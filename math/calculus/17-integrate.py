#!/usr/bin/env python3
""" Function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """ function : calculate integral of a polynomial

        Arguments:
            poly: list of coefficients representing a polynomial
                index : represent the power of x that the coefficient
                belongs to
            C: integration constant

        Returns:
            new list of coefficients representing the integral of
            the polynomial
            None if poly or C are not valid
    """
    # test poly : list or element are number or C is int
    if not poly or not isinstance(poly, list) \
            or not all(isinstance(cuff, (int, float)) for cuff in poly) \
            or not isinstance(C, int):
        return None
    # special case
    elif poly == [0]:
        return[C]
    else:
        # calculate integral of polynom
        # ∫xn dx = x^(n+1)/(n+1) + C
        integral_cuff = [poly[idx] / (idx + 1) for idx in range(0, len(poly))]

        # insert C at the beginning
        integral_cuff.insert(0, float(C))

        # return whole number if integer, else float
        return [float(cuff) if not cuff.is_integer() else int(cuff)
                for cuff in integral_cuff]
