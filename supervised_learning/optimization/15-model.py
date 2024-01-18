#!/usr/bin/env python3
"""
   Builds, trains, saves NN model
   using Adam optimization, mini-batch gradient descent,
   learning rate decay, and batch normalization in TF
"""

import tensorflow.compat.v1 as tf
import numpy as np


def create_layer(prev, n, activation):
    """
        Method to create layer

        :param prev: tensor output of previous layer
        :param n: number of nodes in the layer to create
        :param activation: activation function layer should use

        :return: tensor output of the layer
    """

    # set initialization to He et. al
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # create layer Dense with paramaters
    new_layer = tf.layers.Dense(n,
                                activation=activation,
                                kernel_initializer=initializer,
                                name="layer")

    # apply layer to input
    output = new_layer(prev)

    return output


def create_batch_norm_layer(prev, n, activation):
    """
        Method that creates a batch normalization layer for a
        NN in tf

        :param prev: activated output of the previous layer
        :param n: number of nodes in the layer to be created
        :param activation: activation function for output layer

        :return: tensor of activated output for the layer
    """
    if activation is None:
        return create_layer(prev, n, activation)

    # set initialization to He et. al
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # create layer Dense with parameters
    new_layer = tf.layers.Dense(n,
                                activation=None,
                                kernel_initializer=initializer,
                                name="layer")

    # apply layer to input
    x = new_layer(prev)
    mean, variance = tf.nn.moments(x, axes=[0])

    # beta and gamma : two trainable parameters
    gamma = tf.Variable(tf.ones([n]), name='gamma')
    beta = tf.Variable(tf.zeros([n]), name='beta')

    epsilon = 1e-8

    # apply batch normalization
    x_norm = tf.nn.batch_normalization(
        x=x,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon)

    return activation(x_norm)


def forward_prop(prev, layers, activations, epsilon):
    """

        :param prev: input data
        :param layers: list number of nodes in each layer of NN
        :param activations: list activation function for each layer
        :param epsilon: small number to avoid division by 0

        :return: the prediction of NN in tensor form
    """
    # all layers get batch_normalization but the last one, that stays without
    # any activation or normalization

    # first layer
    layers_norm = create_batch_norm_layer(prev, layers[0], activations[0])

    for i in range(1, len(layers)):
        layers_norm = create_batch_norm_layer(layers_norm,
                                              layers[i], activations[i])

    return layers_norm


def calculate_accuracy(y, y_pred):
    """
        Method to calculates the accuracy of a prediction

        :param y: placeholder for labels of input data
        :param y_pred: tensor containing network's predictions

        :return: tensor containing decimal accuracy of prediction
    """
    # comparison of indice's max value for y and y_pred
    correct_prediction = tf.equal(tf.argmax(y, axis=1),
                                  tf.argmax(y_pred, axis=1))

    # convert tensor bool in float32
    correct_prediction = tf.cast(correct_prediction, dtype=tf.float32)

    # mean of prediction
    accuracy = tf.reduce_mean(correct_prediction)

    return accuracy


def calculate_loss(y, y_pred):
    """
        Method to calculate the softmax cross-entropy loss
        of a prediction

        :param y: placeholder for labels input data
        :param y_pred: tensor network's prediction

        :return: tensor loss prediction
    """

    loss = tf.compat.v1.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred)

    return loss


def create_train_op(loss, alpha, beta1, beta2, epsilon):
    """
        Method that creates the training operation for NN
        using Adam optimization algo

        :param loss: loss of NN's prediction
        :param alpha: learning rate
        :param beta1: weight used 1st moment
        :param beta2: weight used 2nd moment
        :param epsilon: small number to avoid division by 0

        :return: Adam optimizer
    """

    # optimizer gradient descent
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        name='Adam'
    )

    # train
    train = optimizer.minimize(loss)

    return train


def shuffle_data(X, Y):
    """
        Function that shuffles the data points in two matrices the same way

        :param X: ndarray, shape(m, nx) to shuffle
        :param Y: ndarray, shape(m, ny) to shuffle

        :return: shuffled X and Y matrices
    """
    m = X.shape[0]
    permutted_index = np.random.permutation(m)
    X_shuffled = X[permutted_index]
    Y_shuffled = Y[permutted_index]
    return X_shuffled, Y_shuffled


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
        Function that builds, trains, saves a NN model in TF
        using Adam optimizer, mini-batch GD,
              Learning Rate decay, batch normalization

        :param Data_train: tuple, training inputs, training labels
        :param Data_valid: tuple, validation inputs, validation labels
        :param layers: number of nodes in each layer
        :param activations: list activation functions
        :param alpha: learning rate
        :param beta1: weight first moment Adam Opt
        :param beta2: weight second moment Adam Opt
        :param epsilon: small number
        :param decay_rate: decay rate for inverse time of learning rate
        :param batch_size: number of data points in mini-batch
        :param epochs: number times training
        :param save_path: path where model was saved

        :return: saved model path
    """
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # number of training examples (m) and
    # number of features (nx) from input data
    m, nx = X_train.shape
    # number of classes
    classes = Y_train.shape[1]

    # initialize x, y and add them to collection
    x = tf.placeholder(tf.float32, shape=(None, nx))
    y = tf.placeholder(tf.float32, shape=(None, classes))
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)

    # initialize loss and add it to collection
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    # initialize accuracy and add it to collection
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    # initialize global_step variable
    # hint: not trainable
    global_step = tf.Variable(initial_value=0,
                              trainable=False,
                              dtype=tf.int32,
                              name='global_step')

    # create "alpha" the learning rate decay operation in tensorflow
    alpha = tf.compat.v1.train.inverse_time_decay(
        learning_rate=alpha,
        decay_rate=decay_rate,
        decay_steps=1,
        global_step=global_step,
        staircase=True)

    # initizalize train_op and add it to collection
    # hint: don't forget to add global_step parameter in optimizer().minimize()
    train_op = create_train_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs + 1):
            train_cost, train_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid})

            # print training and validation cost and accuracy
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if i < epochs:
                # learning rate decay
                sess.run(global_step.assign(i))

                # update learning rate
                sess.run(alpha)

                # shuffle data
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                nbr_batch = m // batch_size + (m % batch_size != 0)

                for j in range(nbr_batch):
                    start = j * batch_size
                    stop = min(start + batch_size, m)

                    # get X_batch and Y_batch from X_train shuffled
                    # and Y_train shuffled
                    X_batch = X_shuffled[start:stop]
                    Y_batch = Y_shuffled[start:stop]

                    # run training operation
                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    if j > 0 and (j + 1) % 100 == 0:
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X_batch, y: Y_batch}
                        )
                        # print batch cost and accuracy
                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        # save and return the path to where the model was saved
        save_path = saver.save(sess, save_path)

    return save_path
