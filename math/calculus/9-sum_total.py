#!/usr/bin/env python3
""" Function that calculates sum of all i squared"""


def summation_i_squared(n):
    """ function : calculate sum of i squared

        Arguments:
            n: stopping condition

        Returns:
            integer value
    """
    if n < 1 or not isinstance(n, int):
        return None
    else:
        # somme k^2 = n * (n+1) * (2*n+1)
        return(n * (n + 1) * (2 * n + 1)) // 6
