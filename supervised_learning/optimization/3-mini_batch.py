#!/usr/bin/env python3
"""
   Mini-batch
"""

import tensorflow.compat.v1 as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
        Function trains a loaded neural network model using mini-batch gradient descent

        :param X_train: ndarray, shape(m,784), training data
        :param Y_train: ndarray, shape(m,10), training labels
        :param X_valid: ndarray, shape(m,784), validation data
        :param Y_valid: ndarray, shape(m,10), validation labels
        :param batch_size: number of data points in batch
        :param epochs: number of times the training should pass through the whole dataset
        :param load_path: path from which to load the model
        :param save_path: path to where save model after training

        :return: path where model was saved
    """
    # import metagraphe and restore session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph('graph.ckpt.meta')
        new_saver.restore(sess, load_path)

        # get following tensors and op from collection restored
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')

        # loop over epochs

        for epoch in range(epochs + 1):
            train_dataset = shuffle_data(X_train, Y_train)
            X_train = train_dataset[0]
            Y_train = train_dataset[1]

            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})

            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost:: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            len_dataset = len(train_dataset[0])

            for step_number in range(len_dataset//32):
                firs_index = step_number * batch_size
                last_index = (step_number + 1) * batch_size
                x_batch = train_dataset[0][firs_index: last_index]
                y_batch = train_dataset[1][firs_index: last_index]

                step_cost, step_accuracy = sess.run(
                    [loss, accuracy], feed_dict={x: x_batch, y: y_batch})

                if step_number != 0 and step_number % 100 == 0:

                    print("\tStep {}:".format(step_number))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))

        # save session
        saved_model = new_saver.save(sess, save_path)

        return saved_model
