#!/usr/bin/env python3
"""
    Pooling
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
            Function that performs pooling on images

        :param images: ndarray, shape(m, h, w, c), multiple images
        :param kernel_shape: ndarray, shape(kh,kw), kernel shape for pooling
        :param stride: tuple (sh, sw)
        :param mode: type of pooling 'max' or 'avg'

        :return: ndarray containing pooled images

    """
    # size images, kernel, padding, stride
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # output size
    output_height = int((h - kh) / sh + 1)
    output_width = int((w - kw) / sw + 1)

    # initialize output
    pooled_images = np.zeros((m, output_height, output_width, c))

    for i in range(output_height):
        for j in range(output_width):
            image_zone = images[:, i * sh:i * sh + kh,
                                j * sw:j * sw + kw, :]
            if mode == 'max':
                pooled_images[:, i, j, :] = np.max(image_zone, axis=(1, 2))
            elif mode == 'avg':
                pooled_images[:, i, j, :] = np.average(image_zone, axis=(1, 2))

    return pooled_images
