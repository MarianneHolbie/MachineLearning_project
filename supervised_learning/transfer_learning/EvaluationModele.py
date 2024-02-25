#!/usr/bin/env python3
"""
    Transfer Learning with Keras
    Model to use: Inception_Resnet_V2
"""

import tensorflow.keras as K
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools


# Preprocess function
def preprocess_data(X, Y):
    X = tf.keras.applications.inception_resnet_v2.preprocess_input(X)
    y = tf.keras.utils.to_categorical(Y, 10)
    return X, y


if __name__ == "__main__":
    # load CIFAR-10
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # preprocess data
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    # load saved model
    model = K.models.load_model("cifar10.h5")

    # predic for test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # true classes
    y_true_classes = np.argmax(y_test, axis=1)

    # classification report
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes))

    # confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    # Let's prettify it
    fig, ax = plt.subplots(figsize=(10, 10))
    # create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # set lables to be classes
    if class_names is not None:
        labels = class_names
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted Label",
           ylabel="True Label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # adjust label size
    ax.yaxis.label.set_size(15)
    ax.xaxis.label.set_size(15)
    ax.title.set_size(15)

    # set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=8)
    plt.show()