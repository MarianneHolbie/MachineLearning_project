#!/usr/bin/env python3
"""
    Class to represent an exponential distribution
"""


class Exponential:
    """
        Class to represent an exponential distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
            Class constructor

            :data (list): List of the data to be used to estimate the distribution
            :lambtha (float): Expected number of occurrences in a given time frame

        """

        if data is None:
            # Use the given lambtha
            self.lambtha = float(lambtha)
        else:
            # calculate lambtha from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # lambtha is the inverse of the mean of the data
            self.lambtha = 1 / (sum(data) / len(data))

            # check lambtha is positive
            if self.lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
