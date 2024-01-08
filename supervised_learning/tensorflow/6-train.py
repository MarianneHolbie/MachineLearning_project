#!/usr/bin/env python3
"""
    Function training operation
"""

import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
        Method that build, trains and save NN classifier

        :param X_train: ndarray, training input data
        :param Y_train: ndarray, training labels
        :param X_valid: ndarray, validation input data
        :param Y_valid: ndarray, validation labels
        :param layer_sizes: list number of nodes in each layer NN
        :param activations: list activation functions for each layer NN
        :param alpha: learning rate
        :param iterations: number of iterations to train over
        :param save_path: where save the model

        :return: path where the model was saved
    """
    # number of training examples (m) and
    # number of features (nx) from input data
    m, nx = X_train.shape
    # number of classes
    classes = Y_train.shape[1]

    # create placeholder
    x, y = create_placeholders(nx, classes)

    # prediction
    y_pred = forward_prop(x, layer_sizes, activations)

    # loss function
    loss = calculate_loss(y, y_pred)

    # accuracy
    accuracy = calculate_accuracy(y, y_pred)

    # train_op
    train_op = create_train_op(loss, alpha)

    # initialize variables
    init_op = tf.global_variables_initializer()

    # op to save and restore all the variables
    saver = tf.train.Saver()

    # make more readable
    # launch model, initialize variable .. save
    with tf.Session() as sess:
        sess.run(init_op)

        # add to collection
        tf.add_to_collection("x", x)
        tf.add_to_collection("y", y)
        tf.add_to_collection("y_pred", y_pred)
        tf.add_to_collection("loss", loss)
        tf.add_to_collection("accuracy", accuracy)
        tf.add_to_collection("train_op", train_op)

        for i in range(iterations + 1):
            training_loss, training_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})

            valid_loss, valid_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(training_loss))
                print("\tTraining Accuracy: {}".format(training_acc))
                print("\tValidation Cost: {}".format(valid_loss))
                print("\tValidation Accuracy: {}".format(valid_acc))

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        save_path = saver.save(sess, save_path)

    return save_path
