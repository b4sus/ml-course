import numpy as np


def sigmoid(z):
    return 1 / (1 + (np.e ** -z))


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