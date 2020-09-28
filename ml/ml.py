import numpy as np


def train(x_m, y):
    pass


def linear_regression_cost(x_m, y, theta):
    """
    :param x_m: matrix of size m x n
    :param y: vector of length m (m x 1)
    :param theta: vector of length n (n x 1)
    :return: cost - scalar value
    """
    (m, n) = x_m.shape

    assert theta.shape == (n, 1)

    h_x = x_m @ theta  # h_x -> m x 1
    h_x_minus_y = h_x - y
    return (h_x_minus_y.T @ h_x_minus_y) / (2 * m)


def linear_regression_cost_derivative(x_m, y, theta):
    (m, n) = x_m.shape

    assert theta.shape == (n, 1)

    h_x = x_m @ theta
    return ((h_x - y).T @ x_m).T / m


def logistic_regression_hypothesis(x_m, theta):
    """

    :param x_m: matrix of size m x n
    :param theta: vector of length n (n x 1)
    :return: vector of length m (m x 1)
    """
    (m, n) = x_m.shape

    assert theta.shape == (n, 1)

    z = x_m @ theta

    assert z.shape == (m, 1)

    return sigmoid(z)


def sigmoid(z):
    return 1 / (1 + (np.e ** -z))


def logistic_regression_cost(x_m, y, theta, regularization=0):
    (m, n) = x_m.shape

    assert theta.shape == (n, 1)

    h_x = logistic_regression_hypothesis(x_m, theta)
    costs = -y * np.log(h_x) - (1 - y) * np.log(1 - h_x)
    cost = costs.sum() / m
    if regularization:
        cost = cost + ((theta[1:, :] ** 2).sum() * regularization) / (2 * m)
    return cost


def logistic_regression_cost_derivative(x_m, y, theta, regularization=0):
    (m, n) = x_m.shape

    assert theta.shape == (n, 1)

    h_x = logistic_regression_hypothesis(x_m, theta)
    derivate = ((h_x - y).T @ x_m).T / m

    assert derivate.shape == (n, 1)

    if regularization:
        reg = np.vstack((np.array([[0]]), theta[1:, :]))
        reg = (regularization / m) * reg
        derivate = derivate + reg

    return derivate


def logistic_regression_cost_gradient(theta, x_m, y, regularization=0):
    """
    Combines functions logistic_regression_cost and logistic_regression_cost_derivative.
    :param theta:
    :param x_m:
    :param y:
    :return: tuple with function results of logistic_regression_cost and logistic_regression_cost_derivative
    """
    theta = theta.reshape((len(theta), 1))
    return logistic_regression_cost(x_m, y, theta, regularization), logistic_regression_cost_derivative(x_m, y, theta,
                                                                                                        regularization)


def gradient_descent(x_m, y, cost_func, cost_func_derivative, alpha=0.01, num_iter=1000, regularization=0):
    (m, n) = x_m.shape
    theta = np.zeros((n, 1))
    costs = np.empty((num_iter, 1))
    for i in range(num_iter):
        costs[i, 0] = cost_func(x_m, y, theta)
        der = cost_func_derivative(x_m, y, theta, regularization)
        theta = theta - alpha * der
    return theta, costs


def feed_forward(X, Theta_matrices):
    """
    Feeds the inputs X through neural network represented by Theta matrices.
    :param X: inputs: m x n (m examples, n features)
    :param Theta_matrices: Theta matrix for each layer
    :return: matrix of shape m x #neurons_in_output_layer
    """
    (m, n) = X.shape

    A = X.T
    for Theta in Theta_matrices:
        A = np.vstack((np.ones((1, m)), A))
        Z = Theta @ A
        A = sigmoid(Z)

    return A.T