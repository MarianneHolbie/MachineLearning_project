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

        # Euler's number
        self.e = 2.7182818285
        # Pi's number
        self.pi = 3.1415926536

    def z_score(self, x):
        """
            Calculates the z-score of a given x-value

            :param x: given value
            :return: z-score to normalize
        """
        z_score = (x - self.mean) / self.stddev
        return z_score

    def x_value(self, z):
        """
            Calculate the x_value of a given z_score

        :param z: z_score
        :return: x_value
        """
        x_value = self.mean + z * self.stddev
        return x_value
