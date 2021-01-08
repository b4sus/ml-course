import numpy as np

import ml.utils as utils


def feed_forward(X, Thetas):
    """
    Feeds the inputs X through neural network represented by Theta matrices.
    :param X: inputs: m x n (m examples, n features)
    :param Thetas: Theta matrix for each layer
    :return: matrix of shape m x #neurons_in_output_layer
    """
    (m, n) = X.shape

    A = X.T
    for Theta in Thetas:
        A = np.vstack((np.ones((1, m)), A))
        Z = Theta @ A
        A = utils.sigmoid(Z)

    return A.T


def neural_network_cost(X, Y, Thetas, regularization_lambda=0):
    (m, n) = X.shape

    H_X = feed_forward(X, Thetas)

    total_cost = 0

    for i in range(m):
        y = Y[i].T
        h_x = H_X[i].T
        cost_i = -y * np.log(h_x) - (1 - y) * np.log(1 - h_x)
        total_cost += cost_i.sum()

    regularization = 0
    if (regularization_lambda):
        regularization_sum = 0
        for Theta in Thetas:
            Theta_without_bias = Theta[:, 1:]
            regularization_sum += (Theta_without_bias ** 2).sum((0, 1))
        regularization = regularization_sum

    return total_cost / m + regularization_lambda / (2 * m) * regularization


def neural_network_cost_unrolled(X, Y, theta, shapes, regularization_lambda=0):
    Thetas = utils.roll(theta, shapes)
    return neural_network_cost(X, Y, Thetas, regularization_lambda)


def sigmoid_gradient(z):
    sigmoid_z = utils.sigmoid(z)
    return sigmoid_z * (1 - sigmoid_z)


def neural_network_cost_gradient(X, Y, Thetas, regularization_lambda=0):
    (m, n) = X.shape
    cost = 0
    Deltas = [np.zeros(Theta.shape) for Theta in Thetas]
    for (idx, x) in enumerate(X):
        list_a = [x.reshape((len(x), 1))]
        list_z = []
        for Theta in Thetas:
            list_a[-1] = np.vstack((np.array([1]), list_a[-1]))
            list_z.append(Theta @ list_a[-1])
            list_a.append(utils.sigmoid(list_z[-1]))

        y = Y[idx].reshape((Y.shape[1], 1))
        h_x = list_a.pop()
        cost += (-y * np.log(h_x) - (1 - y) * np.log(1 - h_x)).sum()

        list_z.pop()

        delta_last_layer = h_x - y
        for layer_idx in reversed(range(len(Thetas))):
            Deltas[layer_idx] += delta_last_layer @ list_a.pop().T
            if layer_idx > 0:
                delta_last_layer = (Thetas[layer_idx].T @ delta_last_layer)[1:, :] * sigmoid_gradient(list_z.pop())

    Deltas = [Delta / m for Delta in Deltas]

    cost /= m
    if regularization_lambda:
        regularization_sum = 0
        for Theta in Thetas:
            Theta_without_bias = Theta[:, 1:]
            regularization_sum += (Theta_without_bias ** 2).sum((0, 1))
        cost += regularization_lambda / (2 * m) * regularization_sum

        for idx in range(len(Thetas)):
            Theta_without_bias = Thetas[idx][:, 1:]
            Deltas[idx][:, 1:] += Theta_without_bias * (regularization_lambda / m)

    return cost, Deltas


def neural_network_cost_gradient_unrolled(theta, X, Y, shapes, regularization_lambda=0):
    Thetas = utils.roll(theta, shapes)

    (cost, Deltas) = neural_network_cost_gradient(X, Y, Thetas, regularization_lambda)

    return cost, utils.flatten_and_stack(Deltas)[0].reshape((-1))


def initialize_random_theta(shape, epsilon_init=0.12):
    rng = np.random.default_rng()
    return rng.random((shape[0], shape[1] + 1)) * 2 * epsilon_init - epsilon_init
