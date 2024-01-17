#!/usr/bin/env python3
"""
   Mini-batch
"""

import tensorflow.compat.v1 as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
        Function trains a loaded neural network model using
        mini-batch gradient descent

        :param X_train: ndarray, shape(m,784), training data
        :param Y_train: ndarray, shape(m,10), training labels
        :param X_valid: ndarray, shape(m,784), validation data
        :param Y_valid: ndarray, shape(m,10), validation labels
        :param batch_size: number of data points in batch
        :param epochs: number of times the training should pass
               through the whole dataset
        :param load_path: path from which to load the model
        :param save_path: path to where save model after training

        :return: path where model was saved
    """

    # metagraph and restore session
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(load_path + ".meta")
        new_saver.restore(sess, load_path)

        # get following tensors and op from collection restored
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        # store m
        m = X_train.shape[0]

        # loop over epochs
        for epoch in range(epochs + 1):

            # Print training and validation metrics after each epoch
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})

            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                X_train_shuffled, Y_train_shuffled = \
                    shuffle_data(X_train, Y_train)

                # calculate number of batches
                nbr_batch = m // batch_size + (m % batch_size != 0)

                for step_number in range(nbr_batch):
                    first_index = step_number * batch_size
                    last_index = min(first_index + batch_size, m)

                    x_batch = X_train_shuffled[first_index: last_index]
                    y_batch = Y_train_shuffled[first_index: last_index]

                    # run training step with mini-batch
                    sess.run(train_op, feed_dict={x: x_batch, y: y_batch})

                    # print mini-batch result every 100 batches
                    if step_number > 0 and (step_number + 1) % 100 == 0:
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: x_batch, y: y_batch})

                        print("\tStep {}:".format(step_number + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        # save trained model
        saved_model = new_saver.save(sess, save_path)

        return saved_model
