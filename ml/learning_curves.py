import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

import ml.feature as feature


class EstimatorPredictor:
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

def learning_curves_of_different_training_set_size(X_train, y_train, X_cv, y_cv, estimator_predictor, cost_fun):
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
        estimator_predictor.fit(X_sub_train, y_sub_train)

        j_train.append(cost_fun(y_sub_train, estimator_predictor.predict(X_sub_train)))
        j_cv.append(cost_fun(y_cv, estimator_predictor.predict(X_cv_with_bias)))

    plt.plot(list(range(1, len(j_train) + 1)), j_train, label="j_train")
    plt.plot(list(range(1, len(j_train) + 1)), j_cv, label="j_cv")
    plt.xlabel("training set size")
    plt.ylabel("error")
    plt.legend()
    plt.show()

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


def learning_curves_on_random_sets(X_train, y_train, X_cv, y_cv, minimize_fun, cost_fun, regularization_lambda=0):

    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_cv = np.hstack((np.ones((X_cv.shape[0], 1)), X_cv))

    rng = np.random.default_rng()

    j_train = {}
    j_cv = {}
    for i in range(100):
        size = rng.integers(1, X_train.shape[0] + 1)
        indices = rng.choice(list(range(X_train.shape[0])), size, False)
        X_train_sub = X_train[indices, :]
        y_train_sub = y_train[indices, :]
        theta = minimize_fun(X_train_sub, y_train_sub, regularization_lambda)

        j_train_for_size = j_train.get(size, np.array([]))
        j_train_for_size = np.append(j_train_for_size, cost_fun(X_train_sub, y_train_sub, theta))
        j_train[size] = j_train_for_size

        X_cv_sub = X_cv[indices, :]
        y_cv_sub = y_cv[indices, :]
        j_cv_for_size = j_cv.get(size, np.array([]))
        j_cv_for_size = np.append(j_cv_for_size, cost_fun(X_cv_sub, y_cv_sub, theta))
        j_cv[size] = j_cv_for_size

    j_train_means = {size: costs.mean() for size, costs in j_train.items()}
    j_cv_means = {size: costs.mean() for size, costs in j_cv.items()}

    j_train_means_sorted_by_size = {}
    for size in sorted(j_train_means):
        j_train_means_sorted_by_size[size] = j_train_means[size]

    j_cv_means_sorted_by_size = {}
    for size in sorted(j_cv_means):
        j_cv_means_sorted_by_size[size] = j_cv_means[size]

    return j_train_means_sorted_by_size, j_cv_means_sorted_by_size