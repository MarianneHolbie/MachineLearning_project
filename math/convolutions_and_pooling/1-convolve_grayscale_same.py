#!/usr/bin/env python3
"""
    Same Convolution
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
        Function that performs a same convolution on grayscale images

        :param images: ndarray, shape(m, h, w), multiple grayscale images
        :param kernel: ndarray, shape(kh,kw), kernel for convolution

        :return: ndarray containing convolved images
    """
    # size images and kernel
    m, h, w = images.shape
    kh, kw = kernel.shape

    # output size
    output_height = h
    output_width = w

    # initialize output
    convolved_images = np.zeros((m, output_height, output_width))

    # calcul padding (size odd or even)
    padding_width = int(kw / 2)
    padding_height = int(kh / 2)

    # add zero padding to the input images
    image_pad = np.pad(images,
                       ((0, 0), (padding_height, padding_height),
                        (padding_width, padding_width)), mode='constant')

    # convolution
    for i in range(output_height):
        for j in range(output_width):
            # extract region from each image
            image_zone = image_pad[:, i:i+kh, j:j+kw]

            # element wize multiplication
            convolved_images[:, i, j] = np.sum(image_zone * kernel,
                                               axis=(1, 2))

    return convolved_images
