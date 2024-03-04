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

        self.style_image = style_image

        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError("style_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.style_image = self.scale_image(style_image)
        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError("content_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.content_image = self.scale_image(content_image)
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        else:
            self.alpha = alpha
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        else:
            self.beta = beta

        self.model = None
        self.load_model()
        self.gram_style_features, self.content_feature = (
            self.generate_features())

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
            h_new = int((h * 512) / w)
        else:
            h_new = 512
            w_new = int((w * 512) / h)

        resized_image = tf.image.resize(image,
                                        size=[h_new, w_new],
                                        method='bicubic')

        # Normalize
        resized_image = resized_image / 255

        # limit pixel value between 0 and 1
        resized_image = tf.clip_by_value(resized_image, 0, 1)

        tf_resize_image = tf.expand_dims(resized_image, 0)

        return tf_resize_image

    def load_model(self):
        """
            create the model used to calculate cost
            VGG19
            :return:
        """
        # Keras API
        modelVGG19 = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        modelVGG19.trainable = False

        # selected layers
        selected_layers = self.style_layers + [self.content_layer]

        outputs = [modelVGG19.get_layer(name).output for name
                   in selected_layers]

        # construct model
        model = tf.keras.Model([modelVGG19.input], outputs)

        # for replace MaxPooling layer by AveragePooling layer
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        tf.keras.models.save_model(model, 'vgg_base.h5')
        model_avg = tf.keras.models.load_model('vgg_base.h5',
                                               custom_objects=custom_objects)

        self.model = model_avg

    @staticmethod
    def gram_matrix(input_layer):
        """
            Calculate Gram Matrix

            :param input_layer: instance of tf.Tensor or tf.Variable
                shape(1,h,w,c), layer output whose gram matrix should
                be calculated
            :return: tf.tensor, shape(1,c,c) containing gram matrix
        """

        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        # sum of product
        # b: num of batch, i&j spatial coordinate, c channel
        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        # form of input tensor
        input_shape = tf.shape(input_layer)

        # nbr spatial position in each feature card : h*w
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

        # normalisation of result
        norm_result = result / num_locations

        return norm_result

    def generate_features(self):
        """
            method extract the features used to calculate neural style cost
        :return:
        """
        # preprocess style and content image
        preprocess_style = (tf.keras.applications.vgg19.
                            preprocess_input(self.style_image * 255))
        preprocess_content = (
            tf.keras.applications.vgg19.
            preprocess_input(self.content_image * 255))

        # get style and content outputs from VGG19 model
        style_output = self.model(preprocess_style)
        content_output = self.model(preprocess_content)

        # compute Gram matrices for style features
        self.gram_style_features = [self.gram_matrix(style_layer) for
                                    style_layer in style_output]

        # excluding the last element considered more suitable for capturing
        # the style of image
        self.gram_style_features = self.gram_style_features[:-1]

        # select only last network layer
        self.content_feature = content_output[-1]

        return self.gram_style_features, self.content_feature
