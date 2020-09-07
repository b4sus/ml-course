import numpy as np
import matplotlib.pyplot as plt
import ml.ml as ml
from sklearn.preprocessing import PolynomialFeatures

data = np.loadtxt("data/ex2data2.txt", delimiter=",")

print(data)

y = data[:, -1].reshape(len(data), 1)

success_indices = np.nonzero(y.flatten() == 1)
fail_indices = np.nonzero(y.flatten() == 0)

X = data[:, :2]

polynomial_features = PolynomialFeatures(6)
polynomial_features.fit(X)
print(polynomial_features.get_feature_names())

X_P = polynomial_features.transform(X)

print(X)
print(X_P)

(theta, costs) = ml.gradient_descent(X_P, y, ml.logistic_regression_cost, ml.logistic_regression_cost_derivative,
                                     regularization=1)

plt.figure(0)
plt.plot(X[success_indices, 0], X[success_indices, 1], "kx")
plt.plot(X[fail_indices, 0], X[fail_indices, 1], "yo")
plt.show(block=False)

plt.figure(1)
plt.plot(costs)
plt.show()