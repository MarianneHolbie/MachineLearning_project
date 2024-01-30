#!/usr/bin/env python3
"""
    Valid Convolution
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
        Function that performs a valid convolution on grayscale images

        :param images: ndarray, shape(m, h, w), multiple grayscale images
        :param kernel: ndarray, shape(kh,kw), kernel for convolution

        :return: ndarray containing convolved images
    """
    # size images and kernel
    m, h, w = images.shape
    kh, kw = kernel.shape

    # output size
    output_height = h - kh + 1
    output_width = w - kw + 1

    # initialize output
    convolved_images = np.zeros((m, output_height, output_width))

    # convolution
    for i in range(output_height):
        for j in range(output_width):
            # extract region from each image
            image_zone = images[:, i:i+kh, j:j+kw]

            # element wize multiplication
            convolved_images[:, i, j] = np.sum(image_zone * kernel,
                                               axis=(1, 2))

    return convolved_images
