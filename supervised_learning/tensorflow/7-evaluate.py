#!/usr/bin/env python3
"""
    Function evaluates the output of NN
"""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
        Method to evaluates the output of a NN

        :param X: ndarray, input data to evaluate
        :param Y: ndarray, one-hot labels
        :param save_path: location to load the model from

        :return: network prediction, accuracy, and loss
    """

    # open Session
    with tf.Session() as sess:

        # Import and continue training without building the model from scratch.
        new_saver = tf.train.import_meta_graph(save_path + ".meta")
        new_saver.restore(sess, save_path)

        # get variable saved model
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        # calculate new prediction, accuracy and loss
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})

        return prediction, accuracy, loss
