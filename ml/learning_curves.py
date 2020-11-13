import numpy as np
from sklearn.preprocessing import PolynomialFeatures

import ml.feature as feature


def learning_curves_of_different_training_set_size(X_train, y_train, X_cv, y_cv, minimize_fun, cost_fun):
    """
    Trains theta with X_train and y_train using minimize_fun with different training size (1 to full).
    Calculates and return training set and cross-validation error.
    :param X_train:
    :param y_train:
    :param X_cv:
    :param y_cv:
    :param minimize_fun: expected signature minimize_fun(X, y) -> theta
    :param cost_fun: expected signature cost_fun(X, y, theta) -> float
    :return: training errors, cross-validation errors
    """
    j_train = []
    j_cv = []

    X_cv_with_bias = np.hstack((np.ones((X_cv.shape[0], 1)), X_cv))

    for nr_of_training_examples in range(1, X_train.shape[0]):
        X_sub_train = X_train[:nr_of_training_examples, :]
        X_sub_train = np.hstack((np.ones((X_sub_train.shape[0], 1)), X_sub_train))

        y_sub_train = y_train[:X_sub_train.shape[0]]
        theta = minimize_fun(X_sub_train, y_sub_train)

        j_train.append(cost_fun(X_sub_train, y_sub_train, theta))
        j_cv.append(cost_fun(X_cv_with_bias, y_cv, theta))

    return j_train, j_cv


def learning_curves_of_different_polynomial_degree(X_train, y_train, X_cv, y_cv, minimize_fun, cost_fun,
                                                   max_polynomial_degree):
    """
    Trains theta with X_train and y_train using minimize_fun with different polynomial degree
    (1 to provided max_polynomial_degree). Calculates and return training set and cross-validation error.
    :param X_train:
    :param y_train:
    :param X_cv:
    :param y_cv:
    :param minimize_fun: expected signature minimize_fun(X, y) -> theta
    :param cost_fun: expected signature cost_fun(X, y, theta) -> float
    :param max_polynomial_degree:
    :return: training errors, cross-validation errors
    """
    j_train = []
    j_cv = []

    for polynomial_degree in range(1, max_polynomial_degree):
        poly_features = PolynomialFeatures(polynomial_degree, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train)

        normalizer = feature.FeatureNormalizer(X_train_poly)
        X_train_poly = normalizer.normalize_matrix(X_train_poly)

        X_train_poly = np.hstack((np.ones((X_train_poly.shape[0], 1)), X_train_poly))

        theta = minimize_fun(X_train_poly, y_train)

        j_train.append(cost_fun(X_train_poly, y_train, theta))

        X_cv_poly = poly_features.fit_transform(X_cv)
        X_cv_poly = normalizer.normalize_matrix(X_cv_poly)
        X_cv_poly = np.hstack((np.ones((X_cv_poly.shape[0], 1)), X_cv_poly))

        j_cv.append(cost_fun(X_cv_poly, y_cv, theta))

    return j_train, j_cv


def learning_curves_of_different_lambda(X_train, y_train, X_cv, y_cv, minimize_fun, cost_fun,
                                        regularization_lambdas=None):
    """
    Trains theta with X_train and y_train using minimize_fun with different lambdas. Calculates and return training set
    and cross-validation error.
    :param X_train:
    :param y_train:
    :param X_cv:
    :param y_cv:
    :param minimize_fun: expected signature minimize_fun(X, y, regularization_lambda) -> theta
    :param cost_fun: expected signature cost_fun(X, y, theta) -> float
    :param regularization_lambdas: list of lambdas to try
    :return: training errors, cross-validation errors
    """
    if regularization_lambdas is None:
        regularization_lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    j_train = []
    j_cv = []

    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_cv = np.hstack((np.ones((X_cv.shape[0], 1)), X_cv))

    for regularization_lambda in regularization_lambdas:
        theta = minimize_fun(X_train, y_train, regularization_lambda)

        j_train.append(cost_fun(X_train, y_train, theta))
        j_cv.append(cost_fun(X_cv, y_cv, theta))

    return j_train, j_cv, regularization_lambdas
