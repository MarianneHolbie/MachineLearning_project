#!/usr/bin/env python3
""" Class Poisson that represents a poisson distribution"""


class Poisson:
    """
        Class Poisson that represents a poisson distribution
    """
    def __init__(self, data=None, lambtha=1):
        """
            Class constructor

            :param data: list of values used to estimate the distribution
            :param lambtha: lambda value of the distribution
        """

        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
