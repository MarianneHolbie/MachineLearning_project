#!/usr/bin/env python3
"""
    Class to represent a binomial distribution
"""


class Binomial:
    """
        Class to represent a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
            Class constructor

            :param data: List of the data to estimate the distribution
            :param n: number of Bernoulli trials
            :param p: probability of a “success”
        """

        if data is None:
            # Use the given n and p
            self.n = int(n)
            self.p = float(p)
            # check n is positive
            if self.n <= 0:
                raise ValueError("n must be a positive value")
            # check p is between 0 and 1
            if self.p <= 0 or self.p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            # calculate n and p from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # mean is the sum of the data divided by the number of data points
            mean = sum(data) / len(data)

            # calculate variance
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)

            # calculate p from the mean
            self.p = 1 - (variance / mean)

            # calculate n from p and mean
            self.n = round(mean / self.p)

            # recalculate p from n and mean
            self.p = mean / self.n

        # Euler's number
        self.e = 2.7182818285
        # Pi's number
        self.pi = 3.1415926536
