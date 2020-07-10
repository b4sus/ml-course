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
    return ((h_x - y).T @ x_m).T


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

    return 1 / (1 + (np.e ** -z))


def logistic_regression_cost(x_m, y, theta, h_func=logistic_regression_hypothesis):
    h_x = h_func(x_m, theta)
    costs = -y * np.log(h_x) - (1 - y) * np.log(1 - h_x)
    return costs.sum() / x_m.shape[0]


def logistic_regression_cost_derivative(x_m, y, theta, h_func=logistic_regression_hypothesis):
    h_x = h_func(x_m, theta)
    return ((h_x - y).T @ x_m).T


def gradient_descent(x_m, y, cost_func, cost_func_derivative, alpha=0.01, num_iter=1000):
    (m, n) = x_m.shape
    theta = np.zeros((n, 1))
    costs = np.empty((num_iter, 1))
    for i in range(num_iter):
        costs[i, 0] = cost_func(x_m, y, theta)
        der = (cost_func_derivative(x_m, y, theta) / m)
        theta = theta - alpha * der
    return theta, costs
