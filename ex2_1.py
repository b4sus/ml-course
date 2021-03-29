import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

import ml.feature as ft
import ml.logistic_regression as lore
import ml.predict as predict

data = np.loadtxt("ml_course_material/machine-learning-ex2/ex2/ex2data1.txt", delimiter=",")

(m, feature_length_with_result) = data.shape

x = data[:, 0:feature_length_with_result - 1].reshape((m, feature_length_with_result - 1))
y = data[:, feature_length_with_result - 1].reshape((m, 1))

success_indices = np.nonzero(y.flatten() == 1)
fail_indices = np.nonzero(y.flatten() == 0)

normalizer = ft.FeatureNormalizer(x)

normalized_x_m = np.hstack((np.ones((m, 1)), normalizer.normalized_x_m))

# (theta, costs) = ml.gradient_descent(normalized_x_m, y, ml.logistic_regression_cost,
#                                      ml.logistic_regression_cost_derivative, alpha=0.01, num_iter=10000)

(theta, num_of_evaluations, return_code) = op.fmin_tnc(func=lore.logistic_regression_cost_gradient, x0=np.zeros((3, 1)),
                                                       args=(np.hstack((np.ones((m, 1)), x)), y))
theta = theta.reshape(len(theta), 1)

plt.figure(0)
plt.subplot(211)
plt.plot(x[success_indices, 0], x[success_indices, 1], "kx")
plt.plot(x[fail_indices, 0], x[fail_indices, 1], "yo")
plt.plot()
plt.subplot(212)
# plt.plot(costs)
plt.show(block=False)

normalized_example = np.hstack((np.array([1]), normalizer.normalize(np.array([45, 85])))).reshape((1, 3))

# print(ml.logistic_regression_hypothesis(normalized_example, theta))
# print(ml.logistic_regression_hypothesis(np.array([[1, 45, 85]]), theta))

predictions = predict.predict(x, theta, lore.logistic_regression_hypothesis, None)

print(np.mean(predictions == y))

space_x = np.linspace(0, 100, num=100)
space_y = np.linspace(0, 100, num=100)

Z = np.zeros((len(space_x), len(space_y)))
for i in range(len(space_x)):
    for j in range(len(space_y)):
        # Z[i, j] = ml.logistic_regression_hypothesis(
        #     np.hstack((np.array([1]), normalizer.normalize(np.array([space_x[i], space_y[j]])))).reshape((1, 3)), theta)
        Z[i, j] = lore.logistic_regression_hypothesis(
            np.hstack((np.array([1]), np.array([space_x[i], space_y[j]]))).reshape((1, 3)), theta)
        # Z[i, j] = predict.predict(np.array([[space_x[i], space_y[j]]]), theta,
        #                           ml.logistic_regression_hypothesis, None)

plt.figure(1)
plt.contourf(space_x, space_y, Z)
plt.show()
