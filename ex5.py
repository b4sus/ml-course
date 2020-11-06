import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.optimize as op

import ml.ml as ml
import ml.learning_curves as lc

data = sio.loadmat("ml_course_solutions/machine-learning-ex5/ex5/ex5data1.mat")

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

cost = ml.linear_regression_cost(np.hstack((np.ones((len(X_train), 1)), X_train)), y_train, np.array([[1], [1]]), 1)

print(f"cost={cost}")

gradient = ml.linear_regression_cost_derivative(np.hstack((np.ones((len(X_train), 1)), X_train)), y_train,
                                                np.array([[1], [1]]), 1)

print(gradient)

initial_theta = np.zeros((X_train.shape[1] + 1, 1))

X_train_with_bias = np.hstack((np.ones((len(X_train), 1)), X_train))

result = op.minimize(fun=ml.linear_regression_cost_gradient,
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


def minimize(X, y):
    initial_theta = np.zeros((X.shape[1], 1))
    result = op.minimize(fun=ml.linear_regression_cost_gradient,
                         x0=initial_theta.reshape((-1)),
                         args=(X, y),
                         method="CG",
                         jac=True,
                         options={"maxiter": 400})
    return result.x.reshape((-1, 1))


def cost(X, y, theta):
    return ml.linear_regression_cost(X, y, theta)[0][0]


(j_train, j_cv) = lc.learning_curves_of_different_training_set_size(X_train, y_train, X_cv, y_cv, minimize, cost)

plt.figure(1)
plt.plot(list(range(1, len(j_train) + 1)), j_train, label="j_train")
plt.plot(list(range(1, len(j_train) + 1)), j_cv, label="j_cv")
plt.xlabel("training set size")
plt.ylabel("error")
plt.show()