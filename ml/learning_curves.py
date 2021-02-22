import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

import ml.feature as feature


def learning_curves_of_different_training_set_size(X_train, y_train, X_cv, y_cv, estimator_predictor, cost_fun):
    """
    Trains theta with X_train and y_train using minimize_fun with different training size (1 to full).
    Calculates and return training set and cross-validation error.
    :param X_train:
    :param y_train:
    :param X_cv:
    :param y_cv:
    :param estimator_predictor: has methods fit(X, y) and predict(X) -> y
    :param cost_fun: expected signature cost_fun(y_true, y_pred) -> float
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


def learning_curves_of_different_polynomial_degree(X_train, y_train, X_cv, y_cv, estimator_predictor, cost_fun,
                                                   max_polynomial_degree):
    """
    Trains theta with X_train and y_train using minimize_fun with different polynomial degree
    (1 to provided max_polynomial_degree). Calculates and return training set and cross-validation error.
    :param X_train:
    :param y_train:
    :param X_cv:
    :param y_cv:
    :param estimator_predictor: has methods fit(X, y) and predict(X) -> y
    :param cost_fun: expected signature cost_fun(y_true, y_pred) -> float
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

        estimator_predictor.fit(X_train_poly, y_train)

        j_train.append(cost_fun(y_train, estimator_predictor.predict(X_train_poly)))

        X_cv_poly = poly_features.fit_transform(X_cv)
        X_cv_poly = normalizer.normalize_matrix(X_cv_poly)
        X_cv_poly = np.hstack((np.ones((X_cv_poly.shape[0], 1)), X_cv_poly))

        j_cv.append(cost_fun(y_cv, estimator_predictor.predict(X_cv_poly)))

    plt.plot(list(range(1, len(j_train) + 1)), j_train, label="j_train")
    plt.plot(list(range(1, len(j_train) + 1)), j_cv, label="j_cv")
    plt.xlabel("polynomial degree")
    plt.ylabel("error")
    plt.legend()
    plt.show()

    return j_train, j_cv


def learning_curves_of_different_lambda(X_train, y_train, X_cv, y_cv, estimator_predictor_factory, cost_fun,
                                        regularization_lambdas=None):
    """
    Trains theta with X_train and y_train using minimize_fun with different lambdas. Calculates and return training set
    and cross-validation error.
    :param X_train:
    :param y_train:
    :param X_cv:
    :param y_cv:
    :param estimator_predictor_factory: function taking lambda (regularization parameter) producing estimator_predictor,
    has methods fit(X, y) and predict(X) -> y
    :param cost_fun: expected signature cost_fun(y_true, y_pred) -> float
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
        estimator_predictor = estimator_predictor_factory(regularization_lambda)
        estimator_predictor.fit(X_train, y_train)

        j_train.append(cost_fun(y_train, estimator_predictor.predict(X_train)))
        j_cv.append(cost_fun(y_cv, estimator_predictor.predict(X_cv)))

    plt.plot(regularization_lambdas, j_train, label="j_train")
    plt.plot(regularization_lambdas, j_cv, label="j_cv")
    plt.xlabel("lambda")
    plt.ylabel("error")
    plt.legend()
    plt.show()

    return j_train, j_cv, regularization_lambdas


def learning_curves_on_random_sets(X_train, y_train, X_cv, y_cv, estimator_predictor, cost_fun, repeat=100):
    """
    Randomly chooses random number (size = 1 to len(train_set) + 1) of examples from train set snd cv set.
    Then it learns, evaluates and stores costs per size.
    Calculates mean per size and plots the result.
    :param X_train:
    :param y_train:
    :param X_cv:
    :param y_cv:
    :param estimator_predictor: has methods fit(X, y) and predict(X) -> y
    :param cost_fun: expected signature cost_fun(y_true, y_pred) -> float
    :param repeat: defaults to 100
    :return:
    """

    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_cv = np.hstack((np.ones((X_cv.shape[0], 1)), X_cv))

    rng = np.random.default_rng()

    j_train = {}
    j_cv = {}
    for i in range(repeat):
        size = rng.integers(1, X_train.shape[0] + 1)
        indices = rng.choice(list(range(X_train.shape[0])), size, False)
        X_train_sub = X_train[indices, :]
        y_train_sub = y_train[indices, :]
        estimator_predictor.fit(X_train_sub, y_train_sub)

        j_train_for_size = j_train.get(size, np.array([]))
        j_train_for_size = np.append(j_train_for_size, cost_fun(y_train_sub, estimator_predictor.predict(X_train_sub)))
        j_train[size] = j_train_for_size

        X_cv_sub = X_cv[indices, :]
        y_cv_sub = y_cv[indices, :]
        j_cv_for_size = j_cv.get(size, np.array([]))
        j_cv_for_size = np.append(j_cv_for_size, cost_fun(y_cv_sub, estimator_predictor.predict(X_cv_sub)))
        j_cv[size] = j_cv_for_size

    j_train_means = {size: costs.mean() for size, costs in j_train.items()}
    j_cv_means = {size: costs.mean() for size, costs in j_cv.items()}

    j_train_means_sorted_by_size = {}
    for size in sorted(j_train_means):
        j_train_means_sorted_by_size[size] = j_train_means[size]

    j_cv_means_sorted_by_size = {}
    for size in sorted(j_cv_means):
        j_cv_means_sorted_by_size[size] = j_cv_means[size]

    plt.plot(list(j_train_means_sorted_by_size.keys()), list(j_train_means_sorted_by_size.values()), label="j_train_means")
    plt.plot(list(j_cv_means_sorted_by_size.keys()), list(j_cv_means_sorted_by_size.values()), label="j_cv_means")
    plt.xlabel("training set size")
    plt.ylabel("error")
    plt.legend()
    plt.show()

    return j_train_means_sorted_by_size, j_cv_means_sorted_by_size