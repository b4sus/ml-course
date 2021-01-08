import numpy as np

from ml.utils import sigmoid


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
    theta = theta.reshape((-1, 1))
    cost = logistic_regression_cost(x_m, y, theta, regularization)
    gradient = logistic_regression_cost_derivative(x_m, y, theta, regularization).reshape((-1))
    return cost, gradient