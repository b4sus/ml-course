import numpy as np

import ml.utils as utils


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

    X_grad = np.empty((num_movies, num_features))
    for i in range(num_movies):
        users_who_rated_movie_i = np.nonzero(R[i, :] == 1)[0]
        Y_temp = Y[i, users_who_rated_movie_i]
        Theta_temp = Theta[users_who_rated_movie_i, :]
        X_grad[i, :] = (X[i, :] @ Theta_temp.T - Y_temp) @ Theta_temp

    Theta_grad = np.empty((num_users, num_features))
    for j in range(num_users):
        movies_rated_by_user_j = np.nonzero(R[:, j] == 1)[0]
        Y_temp = Y[movies_rated_by_user_j, j].reshape((-1, 1))  # movies_rated_by_user_j x 1
        X_temp = X[movies_rated_by_user_j, :]  # movies_rated_by_user_j x num_features
        theta = Theta[j, :].T.reshape((-1, 1))  # num_features x 1
        Theta_grad[j, :] = (X_temp @ theta - Y_temp).T @ X_temp

    return X_grad, Theta_grad


def cost_function_gradient(params, shapes, Y, R):
    assert len(shapes) == 2
    Matrices = utils.roll(params, shapes)
    X =Matrices[0]
    Theta =Matrices[1]

    (X_grad, Theta_grad) = cost_function_derivative(X, Y, R, Theta)
    return cost_function(X, Y, R, Theta), utils.flatten_and_stack([X_grad, Theta_grad])