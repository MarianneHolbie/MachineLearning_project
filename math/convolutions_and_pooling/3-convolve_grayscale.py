#!/usr/bin/env python3
"""
    Strided convolution
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
        Function that performs a convolution on grayscale images

        :param images: ndarray, shape(m, h, w), multiple grayscale images
        :param kernel: ndarray, shape(kh,kw), kernel for convolution
        :param padding: tuple (ph,pw) and 'same" or "valid'
        :param stride: tuple (sh, sw)

        :return: ndarray containing convolved images
    """
    # size images, kernel, padding, stride
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # output size and padding
    if padding == 'valid':
        # no padding
        ph, pw = 0, 0
    elif padding == 'same':
        """
            (h - 1) * sh : total distance traveled by the filter
             when moving vertically over the image
            (+ kh - h): size of the filter (kh) subtract the height image (h)
             to ensure the filter stays entirely within the image.
            /2 : get the distance from the center of the image
            to the top or bottom edge
            + 1 : add 1 to ensure that even if the distance is a decimal,
            we round up to ensure the filter stays entirely within the image.
            """
        ph = int((((h - 1) * sh + kh - h)/2) + 1)
        pw = int((((w - 1) * sw + kw - w)/2) + 1)

    elif isinstance(padding, tuple):
        ph, pw = padding

    # generalize output calcul
    output_height = int((h - kh + 2 * ph) / sh + 1)
    output_width = int((w - kw + 2 * pw) / sw + 1)

    # initialize output
    convolved_images = np.zeros((m, output_height, output_width))

    # pad image
    image_pad = np.pad(images,
                       ((0, 0), (ph, ph),
                        (pw, pw)), mode='constant')

    # convolution
    for i in range(output_height):
        for j in range(output_width):
            # extract region from each image
            image_zone = image_pad[:, i * sh:i * sh + kh, j * sw:j * sw + kw]

            # element wize multiplication
            convolved_images[:, i, j] = np.sum(image_zone * kernel,
                                               axis=(1, 2))

    return convolved_images
