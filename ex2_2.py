import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from sklearn.preprocessing import PolynomialFeatures

import ml.logistic_regression as lore
import ml.predict as predict

data = np.loadtxt("ml_course_material/machine-learning-ex2/ex2/ex2data2.txt", delimiter=",")

print(data)

y = data[:, -1].reshape(len(data), 1)

success_indices = np.nonzero(y.flatten() == 1)
fail_indices = np.nonzero(y.flatten() == 0)

X = data[:, :2]

polynomial_features = PolynomialFeatures(6)
polynomial_features.fit(X)
print(polynomial_features.get_feature_names())

X_P = polynomial_features.transform(X)

# (theta, costs) = ml.gradient_descent(X_P, y, ml.logistic_regression_cost, ml.logistic_regression_cost_derivative, num_iter=100000,
#                                      regularization=0)

(theta, num_of_evaluations, return_code) = op.fmin_tnc(func=lore.logistic_regression_cost_gradient,
                                                       x0=np.zeros((X_P.shape[1], 1)),
                                                       args=(X_P, y, 0))
theta = theta.reshape((X_P.shape[1], 1))

predictions = predict.predict(X_P[:, 1:], theta, lore.logistic_regression_hypothesis)

print(np.mean(predictions == y))

plt.subplot(121)
plt.plot(X[success_indices, 0], X[success_indices, 1], "kx")
plt.plot(X[fail_indices, 0], X[fail_indices, 1], "yo")
plt.show(block=False)

# plt.subplot(122)
# plt.plot(costs)

space_x = np.linspace(-1, 1.5, num=100)
space_y = np.linspace(-1, 1.5, num=100)

Z = np.zeros((len(space_x), len(space_y)))

for i in range(len(space_x)):
    for j in range(len(space_y)):
        x = np.array([[space_x[i], space_y[j]]])
        x_p = polynomial_features.fit_transform(x)
        # Z[i, j] = x_p@theta
        Z[i, j] = predict.predict(x_p[:, 1:], theta, lore.logistic_regression_hypothesis)

plt.subplot(121)
plt.contour(space_x, space_y, Z.T, linewidths=0.5)
plt.show()
