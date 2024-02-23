#!/usr/bin/env python3
"""
    Transfer Learning with Keras
    Model to use:  Inception_Resnet_V2
"""

import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
        trains a convolutional neural network to classify the CIFAR 10 dataset

        :param X: ndarray, shape(m, 32, 32, 3) containing CIFAR 10 images
        :param Y: ndarray, shape(m, ) containing CIFAR 10 labels for X

        :return: X_p, Y_p
            X_p: ndarray containing preprocessed X
            Y_p: ndarray containing preprocessed Y
    """
    X = K.applications.inception_resnet_v2.preprocess_input(X)
    y = K.utils.to_categorical(Y, 10)
    return X, y


if __name__ == "__main__":
    # load data
    (X_train, y_train), (X_test, y_test) = K.datasets.cifar10.load_data()

    # preprocessing
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    # create model
    base_model = K.applications.InceptionResNetV2(weights='imagenet',
                                                  include_top=False,
                                                  input_shape=(299, 299, 3))

    # input resizing
    inputs = K.Input(shape=(32, 32, 3))
    input = K.layers.Lambda(lambda image: tf.image.resize(image, (299, 299)))(inputs)

    # Base model
    x = base_model(input, training=False)
    # freeze some layer (before 633)
    for layer in base_model.layers[:633]:
        layer.trainable = False

    for layer in base_model.layers[633:]:
        layer.trainable = True

    # Add layers
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(288, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    # construct model
    model = K.Model(inputs, outputs)

    # architecture final model
    model.summary()

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # fit model
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        batch_size=300,
                        epochs=10,
                        verbose=1)

    # save model
    model.save("cifar10.h5")

    # evaluate model
    results = model.evaluate(X_test, y_test)
    print("test loss, test acc:", results)
