# Optimization

## Learning Objectives

* What is a hyperparameter?
* How and why do you normalize your input data?
* What is a saddle point?
* What is stochastic gradient descent?
* What is mini-batch gradient descent?
* What is a moving average? How do you implement it?
* What is gradient descent with momentum? How do you implement it?
* What is RMSProp? How do you implement it?
* What is Adam optimization? How do you implement it?
* What is learning rate decay? How do you implement it?
* What is batch normalization? How do you implement it?

## Tasks

| Filename                   | Description                                                                                       |
|----------------------------|---------------------------------------------------------------------------------------------------|
| `0-norm_constants.py`      | Calculates the normalization (standardization) constants of a matrix.                            |
| `1-normalize.py`           | Normalizes (standardizes) a matrix.                                                               |
| `2-shuffle_data.py`        | Shuffles the data points in two matrices the same way.                                            |
| `3-mini_batch.py`          | Trains a loaded neural network model using mini-batch gradient descent.                           |
| `4-moving_average.py`      | Calculates the weighted moving average of a data set.                                             |
| `5-momentum.py`            | Updates a variable using the gradient descent with momentum optimization algorithm.               |
| `6-momentum.py`            | Trains a loaded neural network model using gradient descent with momentum.                        |
| `7-RMSProp.py`             | Updates a variable using the RMSProp optimization algorithm.                                      |
| `8-RMSProp.py`             | Trains a loaded neural network model using RMSProp optimization.                                  |
| `9-Adam.py`                | Updates a variable in place using the Adam optimization algorithm.                                |
| `10-Adam.py`               | Trains a loaded neural network model using Adam optimization.                                     |
| `11-learning_rate_decay.py`| Updates the learning rate using inverse time decay in numpy.                                     |
| `12-learning_rate_decay.py`| Trains a loaded neural network model using mini-batch gradient descent with learning rate decay.  |
| `13-batch_norm.py`         | Normalizes an unactivated output of a neural network using batch normalization.                   |
| `14-batch_norm.py`         | Trains a loaded neural network model using mini-batch gradient descent with batch normalization.  |
| `15-model.py`              | Creates, trains, and validates a neural network model in tensorflow using Adam optimization.      |
