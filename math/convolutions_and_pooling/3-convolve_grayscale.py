#!/usr/bin/env python3
"""
    Strided convolution
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
        Function that performs a convolution on grayscale images
        with custom padding

        :param images: ndarray, shape(m, h, w), multiple grayscale images
        :param kernel: ndarray, shape(kh,kw), kernel for convolution
        :param padding: tuple (ph,pw) and 'same" or "valid'
        :param stride: tuple (sh, sw)

        :return: ndarray containing convolved images
    """
    # size images, kernel, padding, stride
    m, h, w = images.shape
    kh, kw = kernel.shape
    if isinstance(padding, tuple):
        ph, pw = padding[0]
    else:
        ph, pw = None, None
    sh, sw = stride

    # output size and padding
    if ph is not None and pw is not None:
        output_height = int((h - kh + 2 * ph) / sh + 1)
        output_width = int((w - kw + 2 * pw) / sw + 1)
        image_pad = np.pad(images,
                           ((0, 0), (ph, ph),
                            (pw, pw)), mode='constant')
    elif padding == 'valid':
        output_height = int((h - kh + 1) / sh)
        output_width = int((w - kw + 1) / sw)
        image_pad = images
    else:
        output_height = int(h / sh)
        output_width = int(w / sw)
        padding_width = int(kw / 2)
        padding_height = int(kh / 2)
        image_pad = np.pad(images,
                           ((0, 0), (padding_height, padding_height),
                            (padding_width, padding_width)), mode='constant')

    # initialize output
    convolved_images = np.zeros((m, output_height, output_width))

    # convolution
    for i in range(output_height):
        for j in range(output_width):
            # extract region from each image
            image_zone = image_pad[:, i:i + kh, j:j + kw]

            # element wize multiplication
            convolved_images[:, i, j] = np.sum(image_zone * kernel,
                                               axis=(1, 2))

    return convolved_images
