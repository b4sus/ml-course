import numpy as np

def linear_regression_cost(x_m, y, theta, regularization_lambda=0):
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
    cost = (h_x_minus_y.T @ h_x_minus_y) / (2 * m)

    if regularization_lambda:
        cost += (regularization_lambda / (2 * m)) * (theta[1:, :] ** 2).sum()

    return cost


def linear_regression_cost_derivative(x_m, y, theta, regularization_lambda=0):
    (m, n) = x_m.shape

    assert theta.shape == (n, 1)

    h_x = x_m @ theta
    gradient = ((h_x - y).T @ x_m).T / m

    if regularization_lambda:
        regularization = (regularization_lambda / m) * theta[1:, :]
        assert regularization.shape == (n - 1, 1)
        gradient[1:, :] += regularization

    return gradient


def linear_regression_cost_gradient(theta, X, y, regularization_lambda=0):
    theta = theta.reshape((-1, 1))
    cost = linear_regression_cost(X, y, theta, regularization_lambda)
    gradient = linear_regression_cost_derivative(X, y, theta, regularization_lambda)
    return cost, gradient.reshape((-1))

def rmse(theta, X, y):
    h_x = X @ theta
    return np.sqrt(((y - h_x) ** 2).sum() / len(y))
