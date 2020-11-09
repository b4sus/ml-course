import ml.feature as feature
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def learning_curves_of_different_training_set_size(X_train, y_train, X_cv, y_cv, minimize_fun, cost_fun):
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