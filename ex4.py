import scipy.io as sio
import ml.predict as predict
import ml.ml as ml
import numpy as np

images_mat = sio.loadmat("ml_course_solutions/machine-learning-ex4/ex4/ex4data1.mat")

X = images_mat["X"]
y = images_mat["y"]
# y[y == 10] = 0

Y = np.zeros((5000, 10))

for i in range(len(y)):
    # Y[i, 0 if y[i] == 10 else y[i]] = 1
    Y[i, 9 if y[i] == 10 else y[i] - 1] = 1

weights_mat = sio.loadmat("ml_course_solutions/machine-learning-ex4/ex4/ex4weights.mat")

theta_0 = weights_mat["Theta1"]
theta_1 = weights_mat["Theta2"]

print(theta_0.shape)
print(theta_1.shape)

print(ml.neural_network_cost(X, Y, [theta_0, theta_1]))

Output = ml.feed_forward(X, [theta_0, theta_1])

predictions = np.argmax(Output, axis=1).reshape((X.shape[0], 1))

predictions = predictions + 1

print(np.mean(predictions == y))