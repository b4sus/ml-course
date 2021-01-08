import numpy as np
import scipy.io as sio
import scipy.optimize as op

import ml.neural_network as nn
import ml.utils as utils

images_mat = sio.loadmat("ml_course_material/machine-learning-ex4/ex4/ex4data1.mat")

X = images_mat["X"]
y = images_mat["y"]
# y[y == 10] = 0

Y = np.zeros((5000, 10))

for i in range(len(y)):
    # Y[i, 0 if y[i] == 10 else y[i]] = 1
    Y[i, 9 if y[i] == 10 else y[i] - 1] = 1

weights_mat = sio.loadmat("ml_course_material/machine-learning-ex4/ex4/ex4weights.mat")

theta_0 = weights_mat["Theta1"]
theta_1 = weights_mat["Theta2"]

print(theta_0.shape)
print(theta_1.shape)

print(nn.neural_network_cost(X, Y, [theta_0, theta_1], 1))
print(nn.neural_network_cost_gradient(X, Y, [theta_0, theta_1], 1)[0])

Output = nn.feed_forward(X, [theta_0, theta_1])

predictions = np.argmax(Output, axis=1).reshape((X.shape[0], 1))

predictions = predictions + 1

print(np.mean(predictions == y))

theta_0 = nn.initialize_random_theta((25, 400))
theta_1 = nn.initialize_random_theta((10, 25))

(theta_vec, shapes) = utils.flatten_and_stack([theta_0, theta_1])

result = op.minimize(fun=nn.neural_network_cost_gradient_unrolled,
                     x0=theta_vec.reshape((-1)),
                     args=(X, Y, shapes, 1),
                     method="CG",
                     jac=True,
                     options={"maxiter": 100, "disp": True})

print(result)

Thetas = utils.roll(result.x, shapes)

Output = nn.feed_forward(X, Thetas)

predictions = np.argmax(Output, axis=1).reshape((X.shape[0], 1))

predictions = predictions + 1

print(np.mean(predictions == y))
