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

        # Euler's number
        self.e = 2.7182818285

    def factorial(self, n):
        """
            Accessory function to calculate factorial of a number
        """
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    def pmf(self, k):
        """
            Calculates the value of the PMF for a given number of “successes”

            :param k: number of “successes”
            :return: PMF value for k
        """

        # convert k to int in case it is a float
        if not isinstance(k, int):
            k = int(k)
        # check k is positive
        if k < 0:
            return 0
        else:
            # calculate factorial of k
            fk = self.factorial(k)
            # calculate pmf
            pmf = ((self.lambtha ** k) * (self.e ** (-self.lambtha))) / fk
            return pmf

    def cdf(self, k):
        """
            Calculates the value of the CDF for a given number of “successes”

            :param k: number of “successes”
            :return: CDF value for k
        """

        # convert k to int in case it is a float
        if not isinstance(k, int):
            k = int(k)
        # check k is positive
        if k < 0:
            return 0
        else:
            # calculate cdf
            cdf = 0
            for i in range(0, k + 1):
                cdf += self.pmf(i)
            return cdf
