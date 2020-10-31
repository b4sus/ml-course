import numpy as np


def linear_regression_cost(x_m, y, theta):
    """
    :param x_m: matrix of size m x n
    :param y: vector of length m (m x 1)
    :param theta: vector of length n (n x 1)
    :return: cost - scalar value
    """
    (m, n) = x_m.shape

    assert theta.shape == (n, 1)

    h_x = x_m @ theta  # h_x -> m x 1
    h_x_minus_y = h_x - y
    return (h_x_minus_y.T @ h_x_minus_y) / (2 * m)


def linear_regression_cost_derivative(x_m, y, theta):
    (m, n) = x_m.shape

    assert theta.shape == (n, 1)

    h_x = x_m @ theta
    return ((h_x - y).T @ x_m).T / m


def logistic_regression_hypothesis(x_m, theta):
    """

    :param x_m: matrix of size m x n
    :param theta: vector of length n (n x 1)
    :return: vector of length m (m x 1)
    """
    (m, n) = x_m.shape

    assert theta.shape == (n, 1)

    z = x_m @ theta

    assert z.shape == (m, 1)

    return sigmoid(z)


def sigmoid(z):
    return 1 / (1 + (np.e ** -z))


def logistic_regression_cost(x_m, y, theta, regularization=0):
    (m, n) = x_m.shape

    assert theta.shape == (n, 1)

    h_x = logistic_regression_hypothesis(x_m, theta)
    costs = -y * np.log(h_x) - (1 - y) * np.log(1 - h_x)
    cost = costs.sum() / m
    if regularization:
        cost = cost + ((theta[1:, :] ** 2).sum() * regularization) / (2 * m)
    return cost


def logistic_regression_cost_derivative(x_m, y, theta, regularization=0):
    (m, n) = x_m.shape

    assert theta.shape == (n, 1)

    h_x = logistic_regression_hypothesis(x_m, theta)
    derivate = ((h_x - y).T @ x_m).T / m

    assert derivate.shape == (n, 1)

    if regularization:
        reg = np.vstack((np.array([[0]]), theta[1:, :]))
        reg = (regularization / m) * reg
        derivate = derivate + reg

    return derivate


def logistic_regression_cost_gradient(theta, x_m, y, regularization=0):
    """
    Combines functions logistic_regression_cost and logistic_regression_cost_derivative.
    :param theta:
    :param x_m:
    :param y:
    :return: tuple with function results of logistic_regression_cost and logistic_regression_cost_derivative
    """
    theta = theta.reshape((-1, 1))
    cost = logistic_regression_cost(x_m, y, theta, regularization)
    gradient = logistic_regression_cost_derivative(x_m, y, theta, regularization).reshape((-1))
    return cost, gradient


def gradient_descent(x_m, y, cost_func, cost_func_derivative, alpha=0.01, num_iter=1000, regularization=0):
    (m, n) = x_m.shape
    theta = np.zeros((n, 1))
    costs = np.empty((num_iter, 1))
    for i in range(num_iter):
        costs[i, 0] = cost_func(x_m, y, theta)
        der = cost_func_derivative(x_m, y, theta, regularization)
        theta = theta - alpha * der
    return theta, costs


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
        A = sigmoid(Z)

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
    Thetas = roll(theta, shapes)
    return neural_network_cost(X, Y, Thetas, regularization_lambda)


def sigmoid_gradient(z):
    sigmoid_z = sigmoid(z)
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
            list_a.append(sigmoid(list_z[-1]))

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
    Thetas = roll(theta, shapes)

    (cost, Deltas) = neural_network_cost_gradient(X, Y, Thetas, regularization_lambda)

    return cost, flatten_and_stack(Deltas)[0].reshape((-1))


def flatten_and_stack(Matrices):
    stack = np.empty((0, 1))
    shapes = []
    for M in Matrices:
        shapes.append(M.shape)
        stack = np.vstack((stack, M.reshape((-1, 1))))
    return stack, shapes


def roll(vector, shapes):
    vector = vector.reshape((-1, 1))
    Matrices = []
    from_idx = 0
    for shape in shapes:
        to_idx = from_idx + shape[0] * shape[1]
        Matrices.append(vector[from_idx:to_idx, :].reshape(shape))
        from_idx = to_idx
    return Matrices


def initialize_random_theta(shape, epsilon_init=0.12):
    rng = np.random.default_rng()
    return rng.random((shape[0], shape[1] + 1)) * 2 * epsilon_init - epsilon_init
