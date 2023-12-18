#!/usr/bin/env python3
"""
    Class to represent a normal distribution
"""


class Normal:
    """
        Class to represent a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
            Class constructor

            :param data : List of the data to estimate the distribution
            :param mean : Mean of the distribution
            :param stddev : Standard deviation of the distribution
        """

        if data is None:
            # Use the given mean and stddev
            self.mean = float(mean)
            self.stddev = float(stddev)
            # check stddev is positive
            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            # calculate mean and stddev from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # mean is the sum of the data divided by the number of data points
            self.mean = sum(data) / len(data)

            # stddev is the square root of the variance
            variance = 0
            for i in range(len(data)):
                variance += (data[i] - self.mean) ** 2
            self.stddev = (variance / len(data)) ** (1 / 2)
