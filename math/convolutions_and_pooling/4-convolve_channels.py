#!/usr/bin/env python3
"""
    Convolution with channels
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
        Function that performs a convolution on images with channels

        :param images: ndarray, shape(m, h, w, c), multiple images
        :param kernel: ndarray, shape(kh,kw, c), kernel for convolution
        :param padding: tuple (ph,pw) or 'same" or "valid'
        :param stride: tuple (sh, sw)

        :return: ndarray containing convolved images
    """
    # size images, kernel, padding, stride
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    # output size and padding
    if padding == 'valid':
        # no padding
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)

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
                        (pw, pw), (0, 0)), mode='constant')

    # convolution
    for i in range(output_height):
        for j in range(output_width):
            # extract region from each image
            image_zone = image_pad[:, i * sh:i * sh + kh,
                                   j * sw:j * sw + kw, :]

            # element wize multiplication
            convolved_images[:, i, j] = np.sum(image_zone * kernel,
                                               axis=(1, 2, 3))

    return convolved_images
