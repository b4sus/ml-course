import numpy as np
import scipy.optimize as op

import ml.utils as utils


def cost_function(X, Y, R, Theta, regularization_lambda=0):
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

    regularization = 0
    if regularization_lambda:
        theta_regularization = np.sum(Theta ** 2)
        x_regularization = np.sum(X ** 2)

        regularization = (regularization_lambda / 2) * (theta_regularization + x_regularization)

    return (Cost * R).sum() / 2 + regularization


def cost_function_derivative(X, Y, R, Theta, regularization_lambda=0):
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
        regularization = np.zeros((1, num_features))
        if regularization_lambda:
            regularization = regularization_lambda * X[i, :]
        X_grad[i, :] = (X[i, :] @ Theta_temp.T - Y_temp) @ Theta_temp + regularization

    Theta_grad = np.empty((num_users, num_features))
    for j in range(num_users):
        movies_rated_by_user_j = np.nonzero(R[:, j] == 1)[0]
        Y_temp = Y[movies_rated_by_user_j, j].reshape((-1, 1))  # movies_rated_by_user_j x 1
        X_temp = X[movies_rated_by_user_j, :]  # movies_rated_by_user_j x num_features
        theta = Theta[j, :].T.reshape((-1, 1))  # num_features x 1
        regularization = np.zeros((1, num_features))
        if regularization_lambda:
            regularization = regularization_lambda * Theta[j, :]
        Theta_grad[j, :] = (X_temp @ theta - Y_temp).T @ X_temp + regularization

    return X_grad, Theta_grad


def cost_function_gradient(params, shapes, Y, R, regularization_lambda=0):
    assert len(shapes) == 2
    Matrices = utils.roll(params, shapes)
    X = Matrices[0]
    Theta = Matrices[1]

    (X_grad, Theta_grad) = cost_function_derivative(X, Y, R, Theta, regularization_lambda)
    return cost_function(X, Y, R, Theta, regularization_lambda), utils.flatten_and_stack([X_grad, Theta_grad])[0].reshape((-1))


def learn(Y, R, num_features, regularization_lambda=0):
    """

    :param Y: num_movies x num_users
    :param R: num_movies x num_users
    :param num_features: how many features should be generated
    :param regularization_lambda:
    :return:
    """
    assert Y.shape == R.shape

    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    X = random_initialize((num_movies, num_features))
    Theta = random_initialize((num_users, num_features))

    optimize_result = op.minimize(fun=cost_function_gradient,
                                  x0=utils.flatten_and_stack([X, Theta])[0],
                                  args=([X.shape, Theta.shape], Y, R, regularization_lambda),
                                  method="CG",
                                  jac=True)
    Matrices = utils.roll(optimize_result.x, [X.shape, Theta.shape])

    X = Matrices[0]
    Theta = Matrices[1]

    return X, Theta


def random_initialize(shape, rng=np.random.default_rng()):
    return rng.standard_normal(shape)
