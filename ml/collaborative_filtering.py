import numpy as np


def cost_function(X, Y, R, Theta):
    """

    :param X: num_movies x num_features
    :param Y: num_movies x num_users
    :param R: num_movies x num_users
    :param Theta: num_users x num_features
    :return:
    """
    (num_movies, num_features) = X.shape
    num_users = Y.shape[1]
    assert Y.shape == R.shape
    assert (num_users, num_features) == Theta.shape

    Cost = ((X @ Theta.T) - Y) ** 2
    return (Cost * R).sum() / 2


def cost_function_derivative(X, Y, R, Theta):
    """

    :param X: num_movies x num_features
    :param Y: num_movies x num_users
    :param R: num_movies x num_users
    :param Theta: num_users x num_features
    :return:
    """
    (num_movies, num_features) = X.shape
    num_users = Y.shape[1]
    assert Y.shape == R.shape
    assert (num_users, num_features) == Theta.shape

    Diff = (X @ Theta.T) - Y
    Diff = Diff * R
    # Diff is a num_movies x num_users matrix
    X_grad = Diff @ (Theta * R)

    return X_grad