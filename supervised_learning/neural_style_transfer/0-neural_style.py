#!/usr/bin/env python3
"""
    Neural style transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """
        Class that performs tasks for neural style transfer
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
            Class constructor neural style transfer

            :param style_image: ndarray, image used as style reference
            :param content_image: ndarray, image used as content reference
            :param alpha: weight for content cost
            :param beta: weight for style cost
        """

        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.style_image = style_image
        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError("content_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.content_image = content_image
        if int(alpha) < 0:
            raise TypeError("alpha must be a non-negative number")
        else:
            self.alpha = alpha
        if int(beta) < 0:
            raise TypeError("beta must be a non-negative number")
        else:
            self.beta = beta

    @staticmethod
    def scale_image(image):
        """
            rescales an image such that its pixels values are between 0 and 1
            and its largest side is 512 px

            :param image: ndarray, shape(h,w,3) image to be scaled

            :return:scaled image
        """
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise (TypeError
                   ("image must be a numpy.ndarray with shape (h, w, 3)"))

        h, w, _ = image.shape

        if w > h:
            w_new = 512
            h_new = (h * 512) // w
        else:
            h_new = 512
            w_new = (w * 512) // h

        image = image / 255

        resized_image = tf.image.resize(image,
                                        size=[h_new, w_new],
                                        method='bicubic')

        tf_resize_image = tf.expand_dims(resized_image, 0)

        return tf_resize_image
