import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.optimize as op
from sklearn.preprocessing import PolynomialFeatures

import ml.feature as feature
import ml.learning_curves as lc
import ml.linear_regression as lire

data = sio.loadmat("ml_course_material/machine-learning-ex5/ex5/ex5data1.mat")

X_train = data["X"]
y_train = data["y"]

X_cv = data["Xval"]
y_cv = data["yval"]

X_test = data["Xtest"]
y_test = data["ytest"]

# plt.plot(X_train, y_train, "rx")
# plt.xlabel("Change in water level")
# plt.ylabel("Water flowing out of the dam")
# plt.show()

cost = lire.linear_regression_cost(np.hstack((np.ones((len(X_train), 1)), X_train)), y_train, np.array([[1], [1]]), 1)

print(f"cost={cost}")

gradient = lire.linear_regression_cost_derivative(np.hstack((np.ones((len(X_train), 1)), X_train)), y_train,
                                                  np.array([[1], [1]]), 1)

print(gradient)

initial_theta = np.zeros((X_train.shape[1] + 1, 1))

X_train_with_bias = np.hstack((np.ones((len(X_train), 1)), X_train))

result = op.minimize(fun=lire.linear_regression_cost_gradient,
                     x0=initial_theta.reshape((-1)),
                     args=(X_train_with_bias, y_train),
                     method="CG",
                     jac=True,
                     options={"maxiter": 400})

print(result)

x_points = list(range(-50, 50))
y_points = [result.x @ np.array([[1], [x_point]]) for x_point in x_points]

plt.figure(0)
plt.plot(X_train, y_train, "rx")
plt.plot(x_points, y_points)
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
plt.show()


def minimize(X, y, regularization_lambda=0):
    initial_theta = np.zeros((X.shape[1], 1))
    result = op.minimize(fun=lire.linear_regression_cost_gradient,
                         x0=initial_theta.reshape((-1)),
                         args=(X, y, regularization_lambda),
                         method="CG",
                         jac=True,
                         options={"maxiter": 400})
    return result.x.reshape((-1, 1))


def cost_lire(X, y, theta):
    return lire.linear_regression_cost(X, y, theta)[0][0]


def with_bias(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


class LinearRegressionEstimatorPredictor:
    def __init__(self, regularization_lambda=0):
        self.regularization_lambda = regularization_lambda

    def fit(self, X, y):
        self.theta = minimize(X, y, self.regularization_lambda)

    def predict(self, X):
        return X @ self.theta


def cost(y, y_pred):
    m = len(y)
    diff = y_pred - y
    return ((diff.T @ diff) / (2 * m))[0][0]


plt.figure(1)
(j_train, j_cv) = lc.learning_curves_of_different_training_set_size(X_train, y_train, X_cv, y_cv,
                                                                    LinearRegressionEstimatorPredictor(), cost)

plt.figure(2)
(j_train, j_cv) = lc.learning_curves_of_different_polynomial_degree(X_train, y_train, X_cv, y_cv,
                                                                    LinearRegressionEstimatorPredictor(), cost, 12)

poly_features = PolynomialFeatures(8, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)

normalizer = feature.FeatureNormalizer(X_train_poly)
X_train_poly = normalizer.normalize_matrix(X_train_poly)
X_cv_poly = poly_features.fit_transform(X_cv)
X_cv_poly = normalizer.normalize_matrix(X_cv_poly)
X_test_poly = poly_features.fit_transform(X_test)
X_test_poly = normalizer.normalize_matrix(X_test_poly)

regularization_lambda = 1

result = op.minimize(fun=lire.linear_regression_cost_gradient,
                     x0=np.zeros(X_train_poly.shape[1] + 1),
                     args=(
                         np.hstack((np.ones((X_train_poly.shape[0], 1)), X_train_poly)), y_train,
                         regularization_lambda),
                     method="CG",
                     jac=True,
                     options={"maxiter": 400})

x_points = np.linspace(-60, 70, 1000).reshape((-1, 1))
X_poly = poly_features.fit_transform(x_points)
X_poly = normalizer.normalize_matrix(X_poly)
X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))

plt.figure(3)
plt.plot(X_train, y_train, "rx")
plt.plot(x_points, X_poly @ result.x)
plt.show()

plt.figure(4)
(j_train, j_cv) = lc.learning_curves_of_different_training_set_size(X_train_poly, y_train, X_cv_poly, y_cv,
                                                                    LinearRegressionEstimatorPredictor(
                                                                        regularization_lambda),
                                                                    cost)

plt.figure(5)
(j_train, j_cv, regularization_lambdas) = lc.learning_curves_of_different_lambda(X_train_poly, y_train, X_cv_poly, y_cv,
                                                                                 LinearRegressionEstimatorPredictor,
                                                                                 cost)


# theta = minimize(with_bias(X_train_poly), y_train, 3)
#
# print(f"cv error: {cost(with_bias(X_cv_poly), y_cv, theta)}")
# print(f"test error: {cost(with_bias(X_test_poly), y_test, theta)}")

(j_train_means, j_cv_means) = lc.learning_curves_on_random_sets(X_train_poly, y_train, X_cv_poly, y_cv,
                                                                LinearRegressionEstimatorPredictor(0.01),
                                                                cost,)
plt.figure(6)
