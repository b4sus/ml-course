import ml.feature as feature
import ml.ml as ml
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import scipy.optimize as op

class Step:
    def apply(self, X):
        pass

    def apply_test(self, X):
        return self.apply(X)

class OneHotEncodeStep(Step):
    def __init__(self, indices):
        self.indices = indices

    def apply(self, X):
        return feature.one_hot_encode(X, self.indices)


class NormalizeStep(Step):
    def apply(self, X):
        self.normalizer = feature.FeatureNormalizer(X)
        return self.normalizer.normalized_x_m

    def apply_test(self, X):
        return self.normalizer.normalize_matrix(X)


class PolynomialStep(Step):
    def __init__(self, degree, *, include_bias, interaction_only):
        self.polynomial_features = PolynomialFeatures(degree, include_bias=include_bias,
                                                      interaction_only=interaction_only)

    def apply(self, X):
        return self.polynomial_features.fit_transform(X)


class ReduceFeaturesWithoutStd(Step):
    def apply(self, X):
        (X, self.non_zero_indices) = feature.reduce_features_without_std(X)
        return X

    def apply_test(self, X):
        return X[:, self.non_zero_indices].reshape((X.shape[0], -1))


class BiasStep(Step):
    def apply(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))


class Pipeline:
    def __init__(self):
        self.steps = []

    def one_hot_encode(self, indices):
        self.steps.append(OneHotEncodeStep(indices))

    def normalize(self):
        self.steps.append(NormalizeStep())

    def polynomial(self, degree, *, include_bias, interaction_only):
        self.steps.append(PolynomialStep(degree, include_bias=include_bias, interaction_only=interaction_only))

    def reduce_features_without_std(self):
        self.steps.append(ReduceFeaturesWithoutStd())

    def bias(self):
        self.steps.append(BiasStep())

    def execute_train(self, X, y, /, *, regularization=0):
        print("Vanilla X({} {}):\n{}".format(*X.shape, X))
        for step in self.steps:
            X = step.apply(X)
            print("After {} X({} {}):\n{}".format(step, *X.shape, X))

        (theta, num_of_evaluations, return_code) = op.fmin_tnc(func=ml.logistic_regression_cost_gradient,
                                                               x0=np.zeros((X.shape[1], 1)),
                                                               args=(X, y, regularization))

        return theta.reshape((X.shape[1], 1)), X

    def process_test(self, X):
        for step in self.steps:
            X = step.apply_test(X)
        return X
